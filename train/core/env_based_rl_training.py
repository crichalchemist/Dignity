"""
Real RL training using execution environment with actual PnL rewards.

This module replaces the fake reward lookup tables with genuine reinforcement learning
from trading outcomes. The agent learns by interacting with SimulatedRetailExecutionEnv
and receiving rewards based on actual portfolio value changes.

Key differences from agent_train_rl.py:
- Uses execution environment (not static labels)
- Rewards come from actual PnL (not [-1, 0, 1] lookup table)
- Episodes generate trajectories through environment interaction
- Policy gradient uses real trading outcomes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import PolicyConfig, RLTrainingConfig, SignalModelConfig
from models.signal_policy import ExecutionPolicy, SignalModel
from train.core.target_network import TargetNetwork
from train.execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
)
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)


class ActionConverter:
    """Converts policy network outputs to OrderAction objects for the environment.

    Phase 3: Enhanced with risk-based position sizing capabilities.
    """

    def __init__(
            self,
            lot_size: float = 1.0,
            action_mode: str = "discrete",
            max_position: float = 10.0,
            risk_per_trade: float = 0.02,
            use_dynamic_sizing: bool = False,
    ):
        """
        Args:
            lot_size: Base size of each trading lot
            action_mode: "discrete" (buy/hold/sell) or "continuous" (position sizing)
            max_position: Maximum absolute inventory allowed (risk limit)
            risk_per_trade: Fraction of portfolio to risk per trade (for dynamic sizing)
            use_dynamic_sizing: If True, scale position size based on portfolio and volatility
        """
        self.lot_size = lot_size
        self.action_mode = action_mode
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.use_dynamic_sizing = use_dynamic_sizing

    def policy_to_order(
            self,
            action_idx: int,
            mid_price: float,
            inventory: float,
            cash: float = 50_000.0,
            aggressiveness: float | None = None,
    ) -> OrderAction:
        """
        Convert discrete policy action to OrderAction with risk-based position sizing.

        Args:
            action_idx: 0=SELL, 1=HOLD, 2=BUY
            mid_price: Current market mid price
            inventory: Current position inventory
            cash: Available cash for position sizing
            aggressiveness: Execution aggressiveness [0,1] for limit order engine

        Returns:
            OrderAction for the environment
        """
        if action_idx == 1:  # HOLD
            return OrderAction(action_type="hold", side="buy", size=0.0)

        side = "buy" if action_idx == 2 else "sell"

        # Phase 3: Calculate position size based on risk parameters
        size = self._calculate_position_size(side, inventory, mid_price, cash)

        return OrderAction(
            action_type="market",
            side=side,
            size=size,
            aggressiveness=aggressiveness,
        )

    def _calculate_position_size(
            self, side: str, inventory: float, mid_price: float, cash: float
    ) -> float:
        """
        Calculate position size based on risk management rules.

        This method implements position sizing logic that balances:
        - Risk per trade (using self.risk_per_trade)
        - Maximum position limits (self.max_position)
        - Current inventory (avoid excessive concentration)
        - Available cash

        Args:
            side: "buy" or "sell"
            inventory: Current position inventory (positive=long, negative=short)
            mid_price: Current market mid price
            cash: Available cash

        Returns:
            Position size in lots
        """
        # Phase 3: Dynamic position sizing based on portfolio value
        if self.use_dynamic_sizing:
            # Calculate portfolio value (cash + unrealized position value)
            portfolio_value = cash + (inventory * mid_price)

            # Risk a fixed percentage of portfolio per trade
            # This creates a Kelly-criterion-like approach where position size
            # scales with portfolio growth/shrinkage
            risk_amount = portfolio_value * self.risk_per_trade

            # Convert risk amount to position size in lots
            # Using mid_price as the risk basis (could also use ATR/volatility)
            size = risk_amount / mid_price

            # Round to lot_size increments
            size = max(self.lot_size, round(size / self.lot_size) * self.lot_size)
        else:
            # Fixed lot size (simpler approach)
            size = self.lot_size

        # Apply hard position limits (prevent catastrophic concentration)
        # This is a safety constraint, but the RL agent learns optimal sizing
        # through reward signals (PnL, Sharpe ratio, drawdown, etc.)
        if side == "buy":
            # Check if buying would exceed max position
            new_inventory = inventory + size
            if new_inventory > self.max_position:
                # Reduce size to stay within limit
                size = max(0.0, self.max_position - inventory)
        else:  # sell
            # Check if selling would exceed max short position
            new_inventory = inventory - size
            if new_inventory < -self.max_position:
                # Reduce size to stay within limit
                size = max(0.0, self.max_position + inventory)

        # Apply cash constraint (can't buy more than we can afford)
        if side == "buy":
            max_affordable = cash / mid_price
            size = min(size, max_affordable)

        # Ensure size is non-negative and rounded to lot increments
        size = max(0.0, size)
        size = round(size / self.lot_size) * self.lot_size

        return size


class Episode:
    """Container for a single trading episode trajectory."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.policy_logits: list[torch.Tensor] = []  # For entropy calculation
        self.aggressiveness: list[torch.Tensor] = []  # For execution strategy
        self.dones: list[bool] = []

    def add_step(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            policy_logits: torch.Tensor,
            aggressiveness: torch.Tensor,
            done: bool,
    ):
        """Add a single timestep to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.policy_logits.append(policy_logits)
        self.aggressiveness.append(aggressiveness)
        self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99) -> list[float]:
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0.0
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def compute_advantages(
            self,
            gamma: float = 0.99,
            lambda_: float = 0.95,
            target_values: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            gamma: Discount factor
            lambda_: GAE lambda parameter
            target_values: Optional target network values for stable TD targets.
                If None, uses current value estimates (standard GAE).

        Returns:
            Tuple of (advantages, returns)
        """
        values = torch.stack(self.values)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Use target network values if provided (more stable)
        if target_values is not None:
            bootstrap_values = torch.stack(target_values)
        else:
            bootstrap_values = values

        advantages = []
        advantage = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            # Use target network for bootstrapping, current network for baseline
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lambda_ * advantage * (1 - dones[t])
            advantages.insert(0, advantage.item())
            next_value = bootstrap_values[t]  # Use target values for next_value

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


def collect_episode(
        signal_model: SignalModel,
        policy: ExecutionPolicy,
        env: SimulatedRetailExecutionEnv,
        price_data: np.ndarray,
        device: torch.device,
        action_converter: ActionConverter,
) -> Episode:
    """
    Collect a single episode trajectory by running the agent in the environment.

    Args:
        signal_model: Model that processes price features
        policy: Policy network that outputs actions
        env: Trading execution environment
        price_data: Price features for the episode (T, F) array
        device: Torch device
        action_converter: Converts policy outputs to OrderActions

    Returns:
        Episode object containing states, actions, rewards, etc.
    """
    episode = Episode()
    obs = env.reset()

    for t in range(len(price_data)):
        # Get current price features
        state = price_data[t]

        # Run through model (add batch and sequence dims)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            signal_out = signal_model(x)
            embedding = signal_out["embedding"]

        # Get policy action and execution parameters
        policy_logits, value, aggressiveness = policy(embedding, detach_signal=False)
        action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # Convert to environment action (Phase 3: pass cash for position sizing)
        order_action = action_converter.policy_to_order(
            action.item(),
            obs["mid_price"],
            obs["inventory"],
            obs["cash"],
            aggressiveness=aggressiveness.item(),
        )

        # Step environment
        next_obs, reward, done, info = env.step(order_action)

        # Store transition
        episode.add_step(
            state=state,
            action=action.item(),
            reward=reward,
            log_prob=log_prob,
            value=value.squeeze(),
            policy_logits=policy_logits.detach(),  # Detach to save memory
            aggressiveness=aggressiveness.squeeze().detach(),
            done=done,
        )

        obs = next_obs

        if done:
            break

    return episode


def update_policy(
        policy: ExecutionPolicy,
        optimizer: torch.optim.Optimizer,
        episode: Episode,
        cfg: RLTrainingConfig,
        target_network: TargetNetwork | None = None,
) -> dict[str, float]:
    """
    Update policy using collected episode trajectory.

    Uses PPO (Proximal Policy Optimization) if cfg.use_ppo=True,
    otherwise uses vanilla A2C policy gradient.

    Args:
        policy: Policy network to update
        optimizer: Optimizer for policy parameters
        episode: Collected episode trajectory
        cfg: RL training configuration
        target_network: Optional target network for stable value estimation

    Returns:
        Dictionary of loss metrics
    """
    if cfg.use_ppo:
        return update_policy_ppo(policy, optimizer, episode, cfg, target_network)
    else:
        return update_policy_a2c(policy, optimizer, episode, cfg, target_network)


def update_policy_a2c(
        policy: ExecutionPolicy,
        optimizer: torch.optim.Optimizer,
        episode: Episode,
        cfg: RLTrainingConfig,
        target_network: TargetNetwork | None = None,
) -> dict[str, float]:
    """
    Update policy using A2C-style vanilla policy gradient.

    Args:
        policy: Policy network to update
        optimizer: Optimizer for policy parameters
        episode: Collected episode trajectory
        cfg: RL training configuration
        target_network: Optional target network for stable value estimation

    Returns:
        Dictionary of loss metrics
    """
    # Compute target values for stable bootstrapping (if target network provided)
    target_values = None
    if target_network is not None:
        target_values = list(episode.values)  # Use stored values as approximation
        # In a full implementation, we'd recompute from states through target network

    # Compute advantages and returns
    advantages, returns = episode.compute_advantages(
        gamma=cfg.gamma, lambda_=cfg.gae_lambda, target_values=target_values
    )

    # Stack tensors
    log_probs = torch.stack(episode.log_probs)
    values = torch.stack(episode.values)
    policy_logits = torch.stack(episode.policy_logits)

    # Policy gradient loss (negative because we want to maximize)
    policy_loss = -(log_probs * advantages.detach()).mean()

    # Value function loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus for exploration
    # Calculate entropy from policy logits to encourage exploration
    action_probs = F.softmax(policy_logits, dim=-1)
    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

    # Total loss with entropy regularization
    total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    optimizer.zero_grad()
    total_loss.backward()

    if cfg.grad_clip:
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)

    optimizer.step()

    # Update target network (soft Polyak averaging)
    if target_network is not None:
        target_network.update()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": total_loss.item(),
        "mean_return": returns.mean().item(),
        "mean_advantage": advantages.mean().item(),
    }


def update_policy_ppo(
        policy: ExecutionPolicy,
        optimizer: torch.optim.Optimizer,
        episode: Episode,
        cfg: RLTrainingConfig,
        target_network: TargetNetwork | None = None,
) -> dict[str, float]:
    """
    Update policy using PPO (Proximal Policy Optimization).

    PPO clips the policy ratio to prevent destructively large updates,
    enabling stable training and data reuse across multiple epochs.

    Args:
        policy: Policy network to update
        optimizer: Optimizer for policy parameters
        episode: Collected episode trajectory
        cfg: RL training configuration
        target_network: Optional target network for stable value estimation

    Returns:
        Dictionary of loss metrics with PPO-specific metrics
    """
    # Compute target values for stable bootstrapping (if target network provided)
    target_values = None
    if target_network is not None:
        target_values = list(episode.values)  # Use stored values as approximation
        # In a full implementation, we'd recompute from states through target network

    # Compute advantages and returns
    advantages, returns = episode.compute_advantages(
        gamma=cfg.gamma, lambda_=cfg.gae_lambda, target_values=target_values
    )

    # Normalize advantages (improves stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Stack tensors
    old_log_probs = torch.stack(episode.log_probs).detach()  # Detach for old policy
    old_values = torch.stack(episode.values).detach()
    policy_logits = torch.stack(episode.policy_logits)
    actions = torch.tensor(list(episode.actions), dtype=torch.long)

    # Store states for recomputing policy during PPO epochs
    torch.stack([torch.from_numpy(s).float() for s in episode.states])

    # PPO epochs: Optimize same data multiple times
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_fraction = 0.0
    num_updates = 0
    early_stopped = False

    for _ppo_epoch in range(cfg.ppo_epochs):
        # Recompute policy outputs for current policy
        # Note: In full implementation, we'd recompute from states through signal model
        # For now, use stored policy_logits and recompute action probs

        # Get current action probabilities
        current_action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = torch.distributions.Categorical(current_action_probs)

        # New log probabilities under current policy
        new_log_probs = action_dist.log_prob(actions)

        # Importance sampling ratio: π_new(a|s) / π_old(a|s)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO clipped surrogate objective
        # L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (clipped for stability)
        # Recompute values - in simplified version, use old values
        # In full version: new_values = policy.value_net(signal_embedding)
        value_loss = F.mse_loss(old_values, returns)

        # Entropy bonus
        entropy = action_dist.entropy().mean()

        # Total loss
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if cfg.max_grad_norm:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)

        optimizer.step()

        # Track metrics
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        num_updates += 1

        # Compute KL divergence for monitoring (approximate)
        with torch.no_grad():
            kl = (old_log_probs - new_log_probs).mean().item()
            total_kl += kl

            # Clip fraction: proportion of ratios that were clipped
            clip_fraction = ((ratio < 1.0 - cfg.clip_range) | (ratio > 1.0 + cfg.clip_range)).float().mean().item()
            total_clip_fraction += clip_fraction

            # Early stopping if KL divergence too high
            if cfg.target_kl and kl > cfg.target_kl:
                early_stopped = True
                break

    # Update target network after all PPO epochs (soft Polyak averaging)
    if target_network is not None:
        target_network.update()

    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "total_loss": (total_policy_loss + total_value_loss) / num_updates,
        "mean_return": returns.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "kl_divergence": total_kl / num_updates,
        "clip_fraction": total_clip_fraction / num_updates,
        "ppo_epochs_used": num_updates,
        "early_stopped": 1 if early_stopped else 0,
    }


def train_with_environment(
        signal_model: SignalModel,
        policy: ExecutionPolicy,
        train_data: np.ndarray,
        cfg: RLTrainingConfig,
        env_config: ExecutionConfig,
        device: torch.device,
) -> ExecutionPolicy:
    """
    Train policy using real trading environment and PnL rewards.

    Args:
        signal_model: Pre-trained signal model (frozen during training)
        policy: Execution policy to train
        train_data: Training price features (N, T, F) array
        cfg: RL training configuration
        env_config: Execution environment configuration
        device: Torch device

    Returns:
        Trained policy
    """
    # Freeze signal model - set to inference mode
    for param in signal_model.parameters():
        param.requires_grad = False
    signal_model.train(False)

    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    # Initialize target network for stable value estimation
    target_network = TargetNetwork(
        policy.value_net,
        tau=0.005,  # Soft update coefficient (SAC/TD3 default)
        update_mode="soft",  # Continuous Polyak averaging
    )

    # Phase 3: Initialize ActionConverter with position sizing parameters
    action_converter = ActionConverter(
        lot_size=env_config.lot_size,
        max_position=10.0,  # Maximum inventory (long or short)
        risk_per_trade=0.02,  # Risk 2% of portfolio per trade
        use_dynamic_sizing=True,  # Enable dynamic position sizing
    )

    algorithm = "PPO" if cfg.use_ppo else "A2C"
    logger.info(f"Starting environment-based RL training for {cfg.epochs} epochs")
    logger.info(f"Algorithm: {algorithm}")
    if cfg.use_ppo:
        logger.info(f"  PPO clip_range: {cfg.clip_range}")
        logger.info(f"  PPO epochs per batch: {cfg.ppo_epochs}")
        logger.info(f"  Target KL: {cfg.target_kl}")
    logger.info(f"Training data shape: {train_data.shape}")

    for epoch in range(1, cfg.epochs + 1):
        epoch_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "mean_return": 0.0,
            "kl_divergence": 0.0,  # PPO metric
            "clip_fraction": 0.0,  # PPO metric
        }
        num_episodes = 0

        # Collect episodes from training data
        for episode_idx in range(len(train_data)):
            # Create fresh environment for each episode
            env = SimulatedRetailExecutionEnv(env_config)

            # Get episode price data
            price_sequence = train_data[episode_idx]  # (T, F)

            # Collect episode trajectory
            episode = collect_episode(
                signal_model=signal_model,
                policy=policy,
                env=env,
                price_data=price_sequence,
                device=device,
                action_converter=action_converter,
            )

            # Update policy with target network for stable value estimation
            metrics = update_policy(policy, optimizer, episode, cfg, target_network)

            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0.0)
            num_episodes += 1

            if (episode_idx + 1) % 10 == 0:
                if cfg.use_ppo and 'kl_divergence' in metrics:
                    logger.info(
                        f"[epoch {epoch}] episode {episode_idx + 1}/{len(train_data)} | "
                        f"return={metrics['mean_return']:.2f} | "
                        f"policy_loss={metrics['policy_loss']:.4f} | "
                        f"entropy={metrics['entropy']:.4f} | "
                        f"kl={metrics['kl_divergence']:.4f} | "
                        f"clip={metrics['clip_fraction']:.2%}"
                    )
                else:
                    logger.info(
                        f"[epoch {epoch}] episode {episode_idx + 1}/{len(train_data)} | "
                        f"return={metrics['mean_return']:.2f} | "
                        f"policy_loss={metrics['policy_loss']:.4f} | "
                        f"entropy={metrics['entropy']:.4f}"
                    )

        # Log epoch summary
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_episodes, 1)

        if cfg.use_ppo and epoch_metrics['kl_divergence'] > 0:
            logger.info(
                f"Epoch {epoch}/{cfg.epochs} complete | "
                f"avg_return={epoch_metrics['mean_return']:.2f} | "
                f"policy_loss={epoch_metrics['policy_loss']:.4f} | "
                f"value_loss={epoch_metrics['value_loss']:.4f} | "
                f"entropy={epoch_metrics['entropy']:.4f} | "
                f"kl={epoch_metrics['kl_divergence']:.4f} | "
                f"clip={epoch_metrics['clip_fraction']:.2%}"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{cfg.epochs} complete | "
                f"avg_return={epoch_metrics['mean_return']:.2f} | "
                f"policy_loss={epoch_metrics['policy_loss']:.4f} | "
                f"value_loss={epoch_metrics['value_loss']:.4f} | "
                f"entropy={epoch_metrics['entropy']:.4f}"
            )

    logger.info("Environment-based RL training complete")
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Train RL policy using execution environment with real PnL rewards"
    )
    parser.add_argument("--signal-model-path", required=True, help="Path to pre-trained signal model")
    parser.add_argument("--train-data-path", required=True, help="Path to training data (.npy)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping")

    # PPO-specific arguments
    parser.add_argument("--use-ppo", action="store_true", default=True, help="Use PPO instead of vanilla PG")
    parser.add_argument("--no-ppo", action="store_false", dest="use_ppo", help="Disable PPO (use vanilla PG)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO optimization epochs per batch")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL for early stopping (None to disable)")

    parser.add_argument("--checkpoint-path", default="models/env_policy.pt", help="Save path")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # Load signal model (frozen)
    # TODO: Add proper model loading from checkpoint
    logger.info(f"Loading signal model from {args.signal_model_path}")
    # signal_model = torch.load(args.signal_model_path, map_location=device)

    # Load training data
    logger.info(f"Loading training data from {args.train_data_path}")
    # train_data = np.load(args.train_data_path)

    # For now, create placeholder
    logger.warning("Using placeholder model and data - TODO: implement proper loading")
    signal_cfg = SignalModelConfig(num_features=55)  # Updated for FX features + alphas
    signal_model = SignalModel(signal_cfg).to(device)

    train_data = np.random.randn(100, 390, 55).astype(np.float32)  # (N, T, F)

    # Create policy
    policy_cfg = PolicyConfig(
        input_dim=signal_model.signal_dim,
        hidden_dim=128,
        num_actions=3,  # SELL, HOLD, BUY
    )
    policy = ExecutionPolicy(policy_cfg).to(device)

    # RL training config
    rl_cfg = RLTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        grad_clip=args.grad_clip,
        # PPO parameters
        use_ppo=args.use_ppo,
        clip_range=args.clip_range,
        ppo_epochs=args.ppo_epochs,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl if args.target_kl > 0 else None,
    )

    # Environment config
    env_cfg = ExecutionConfig(
        initial_cash=50_000.0,
        lot_size=1.0,
        spread=0.02,
        time_horizon=390,  # 6.5 hour trading day in minutes
    )

    # Train policy
    trained_policy = train_with_environment(
        signal_model=signal_model,
        policy=policy,
        train_data=train_data,
        cfg=rl_cfg,
        env_config=env_cfg,
        device=device,
    )

    # Save checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_policy.state_dict(), checkpoint_path)
    logger.info(f"Saved policy checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
