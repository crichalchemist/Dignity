"""SAC-based training loop for FX trading.

Implements off-policy training with experience replay for sample-efficient learning.

Key differences from PPO training:
- Off-policy: Reuses past experiences from replay buffer
- Multiple updates per step: 1-4 gradient updates per environment step
- Warmup period: Collects initial random experiences before training
- Continuous actions: Outputs aggressiveness directly (no discretization)

Architecture:
    Signal Model (frozen) → SAC Policy → Environment → Replay Buffer → SAC Updates
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, "/Volumes/Containers/Sequence")

from config.config import SignalModelConfig
from execution.limit_order_engine import LimitOrderConfig
from execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv
from models.signal_policy import SignalModel
from rl.agents.sac_agent import SACAgent
from rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """Configuration for SAC training.

    Attributes:
        buffer_size: Replay buffer capacity
        batch_size: Batch size for SAC updates
        updates_per_step: Number of gradient updates per environment step
        warmup_steps: Random exploration steps before training
        target_entropy: Target entropy for auto-tuning (if None, uses -action_dim)
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        alpha: Initial entropy temperature
        auto_entropy_tuning: Whether to auto-tune alpha
        use_prioritized_replay: Use prioritized experience replay
        prioritized_alpha: Prioritization exponent (0=uniform, 1=full)
        prioritized_beta_start: Initial importance sampling correction
        prioritized_beta_frames: Frames to anneal beta to 1.0
    """

    buffer_size: int = 100_000
    batch_size: int = 256
    updates_per_step: int = 1
    warmup_steps: int = 1_000
    target_entropy: float | None = None
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_entropy_tuning: bool = True
    use_prioritized_replay: bool = False
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_frames: int = 100_000


def train_with_sac(
        signal_model: SignalModel,
        sac_config: SACConfig,
        signal_cfg: SignalModelConfig,
        env_config: ExecutionConfig,
        train_data: np.ndarray,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        log_frequency: int = 10,
        checkpoint_dir: str | None = None,
        device: str = "cpu",
        sequence_length: int = 100,
) -> dict:
    """Train SAC agent on FX trading task.

    Args:
        signal_model: Pretrained signal model (frozen)
        sac_config: SAC training configuration
        signal_cfg: Signal model configuration
        env_config: Execution environment configuration
        train_data: Training data [num_steps, num_features]
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        log_frequency: Log metrics every N episodes
        checkpoint_dir: Directory to save checkpoints
        device: Device to run on
        sequence_length: Length of price sequences to feed to signal model

    Returns:
        Dictionary of training metrics
    """
    logger.info("Initializing SAC training...")

    # Validate train_data
    if len(train_data.shape) != 2:
        raise ValueError(f"train_data must be 2D [num_steps, num_features], got {train_data.shape}")
    if train_data.shape[1] != signal_cfg.num_features:
        raise ValueError(
            f"train_data features ({train_data.shape[1]}) != signal_cfg.num_features ({signal_cfg.num_features})")

    # Setup
    signal_model.to(device)
    signal_model.eval()  # Set to eval mode
    for param in signal_model.parameters():
        param.requires_grad = False  # Freeze signal model

    # Create SAC agent
    # State is the signal model embedding, not raw features
    state_dim = signal_model.signal_dim  # Get actual embedding dimension
    action_dim = 1  # Aggressiveness parameter

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=sac_config.lr,
        gamma=sac_config.gamma,
        tau=sac_config.tau,
        alpha=sac_config.alpha,
        auto_entropy_tuning=sac_config.auto_entropy_tuning,
        target_entropy=sac_config.target_entropy,
        device=device,
    )

    # Create replay buffer
    if sac_config.use_prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=sac_config.buffer_size,
            alpha=sac_config.prioritized_alpha,
            beta_start=sac_config.prioritized_beta_start,
            beta_frames=sac_config.prioritized_beta_frames,
        )
        logger.info(f"Using prioritized replay buffer (α={sac_config.prioritized_alpha})")
    else:
        replay_buffer = ReplayBuffer(capacity=sac_config.buffer_size)
        logger.info("Using uniform replay buffer")

    # Create environment (will use train_data internally)
    env = SimulatedRetailExecutionEnv(config=env_config, seed=42)

    logger.info("Training SAC agent:")
    logger.info(f"  - State dim: {state_dim} (signal embedding)")
    logger.info(f"  - Action dim: {action_dim}")
    logger.info(f"  - Buffer size: {sac_config.buffer_size}")
    logger.info(f"  - Batch size: {sac_config.batch_size}")
    logger.info(f"  - Updates per step: {sac_config.updates_per_step}")
    logger.info(f"  - Warmup steps: {sac_config.warmup_steps}")
    logger.info(f"  - Sequence length: {sequence_length}")
    logger.info(f"  - Training data: {train_data.shape}")

    # Training loop
    import torch

    total_steps = 0
    episode_rewards = []
    episode_lengths = []

    # Calculate max episode start index
    max_start_idx = len(train_data) - sequence_length - max_steps_per_episode
    if max_start_idx <= 0:
        raise ValueError(
            f"train_data too short ({len(train_data)}) for sequence_length ({sequence_length}) + max_steps ({max_steps_per_episode})")

    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_length = 0

        # Random starting position in train_data for this episode
        episode_start_idx = np.random.randint(0, max_start_idx)
        current_idx = episode_start_idx

        for _step in range(max_steps_per_episode):
            # Extract price sequence from train_data
            # [sequence_length, num_features]
            price_sequence = train_data[current_idx: current_idx + sequence_length]

            # Process through signal model to get state embedding
            with torch.no_grad():
                price_tensor = torch.from_numpy(price_sequence).float().unsqueeze(0).to(
                    device)  # [1, seq_len, features]
                signal_output = signal_model(price_tensor)
                state_embedding = signal_output["embedding"].squeeze(0).cpu().numpy()  # [state_dim]

                # Get direction from signal model if it has direction head
                if "aux" in signal_output and "direction_logits" in signal_output["aux"]:
                    direction_logits = signal_output["aux"]["direction_logits"].squeeze(0).cpu().numpy()  # [3]
                    # Sample direction: 0=SELL, 1=HOLD, 2=BUY
                    direction = int(np.argmax(direction_logits))
                else:
                    # Fallback: random direction if no direction head
                    direction = np.random.choice([0, 2])  # Skip HOLD for now

            state = state_embedding

            # Select aggressiveness action from SAC
            if total_steps < sac_config.warmup_steps:
                # Random exploration during warmup
                action = np.random.uniform(-1, 1, size=action_dim)
            else:
                # Sample from policy
                action = agent.select_action(state, evaluate=False)

            # Convert continuous action to order action
            # action is aggressiveness in [-1, 1], map to [0, 1]
            aggressiveness = (action[0] + 1.0) / 2.0  # Map [-1,1] to [0,1]

            # Create order action
            from execution.simulated_retail_env import OrderAction

            # Skip HOLD actions (direction == 1) - only BUY or SELL
            if direction == 1:
                # For HOLD, just advance time without trading
                next_obs = obs
                reward = 0.0
                done = False
                info = {}
            else:
                order_action = OrderAction(
                    action_type="market",
                    side="buy" if direction == 2 else "sell",
                    size=1.0,
                    aggressiveness=aggressiveness,
                )

                # Execute in environment
                next_obs, reward, done, info = env.step(order_action)

            # Get next state
            next_idx = current_idx + 1
            if next_idx + sequence_length >= len(train_data):
                # Reached end of data
                done = True
                next_state = state  # Reuse current state
            else:
                # Extract next price sequence
                next_price_sequence = train_data[next_idx: next_idx + sequence_length]

                with torch.no_grad():
                    next_price_tensor = torch.from_numpy(next_price_sequence).float().unsqueeze(0).to(device)
                    next_signal_output = signal_model(next_price_tensor)
                    next_state = next_signal_output["embedding"].squeeze(0).cpu().numpy()

            # Store in replay buffer
            replay_buffer.add(state, action[0], reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1
            current_idx = next_idx

            # Update SAC agent
            if len(replay_buffer) > sac_config.batch_size and total_steps >= sac_config.warmup_steps:
                for _ in range(sac_config.updates_per_step):
                    agent.update(replay_buffer, sac_config.batch_size)

            if done:
                break

        # Log episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-log_frequency:])
            avg_length = np.mean(episode_lengths[-log_frequency:])

            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Length: {avg_length:.0f} | "
                f"Buffer: {len(replay_buffer)} | "
                f"Alpha: {agent.alpha:.3f}"
            )

            # Save checkpoint
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / f"sac_checkpoint_ep{episode + 1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("SAC training complete!")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_buffer_size": len(replay_buffer),
        "total_steps": total_steps,
    }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Testing SAC Training Loop...\n")

    # Create configurations
    signal_cfg = SignalModelConfig(num_features=76)
    signal_model = SignalModel(signal_cfg)

    sac_config = SACConfig(
        buffer_size=10_000,
        batch_size=64,
        updates_per_step=1,
        warmup_steps=100,
        auto_entropy_tuning=True,
    )

    env_config = ExecutionConfig(
        use_limit_order_engine=True,
        limit_order_config=LimitOrderConfig(),
        initial_cash=50_000.0,
    )

    # Dummy training data (need enough for sequences)
    train_data = np.random.randn(5000, 76)

    # Run short training
    results = train_with_sac(
        signal_model=signal_model,
        sac_config=sac_config,
        signal_cfg=signal_cfg,
        env_config=env_config,
        train_data=train_data,
        num_episodes=10,
        max_steps_per_episode=100,
        log_frequency=5,
        checkpoint_dir=None,
        device="cpu",
        sequence_length=100,  # Price sequence length
    )

    print("\n" + "=" * 60)
    print("✅ SAC Training Test Complete!")
    print("=" * 60)
    print("Results:")
    print(f"  - Episodes: {len(results['episode_rewards'])}")
    print(f"  - Total steps: {results['total_steps']}")
    print(f"  - Final buffer size: {results['final_buffer_size']}")
    print(f"  - Avg reward (last 5): {np.mean(results['episode_rewards'][-5:]):.2f}")
    print(f"  - Avg length (last 5): {np.mean(results['episode_lengths'][-5:]):.0f}")
