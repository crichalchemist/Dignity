"""Train RL trading policy on prepared historical FX data.

This script loads prepared numpy sequences and trains the execution policy using
real RL with telemetry integration and GPU cluster support.

Usage:
    python scripts/train_on_historical_data.py --data-dir data/prepared --epochs 100 --batch-size 32
    python scripts/train_on_historical_data.py --config configs/production.yaml --gpu
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import PolicyConfig, RLTrainingConfig, SignalModelConfig
from execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv
from models.signal_policy import ExecutionPolicy
from train.core.env_based_rl_training import ActionConverter
from utils.logger import get_logger
from utils.seed import set_seed
from utils.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class HistoricalDataLoader:
    """Loads and manages prepared historical FX data for training."""

    def __init__(self, data_dir: Path, sequence_length: int = 390):
        """
        Args:
            data_dir: Directory containing prepared .npy files
            sequence_length: Expected sequence length per episode
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = sorted(glob.glob(str(data_dir / "*.npy")))

        if not self.data_files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        logger.info(f"Found {len(self.data_files)} data files")
        self._load_metadata()
        self._build_file_index()

    def _load_metadata(self):
        """Load metadata about the dataset."""
        # Load first file to get shape info
        sample = np.load(self.data_files[0])
        self.num_features = sample.shape[2]  # (sequences, timesteps, features)

        # Count total sequences across all files
        self.total_sequences = sum(
            np.load(f).shape[0] for f in self.data_files
        )

        logger.info(f"Dataset: {self.total_sequences:,} sequences")
        logger.info(f"Shape: ({self.sequence_length} steps, {self.num_features} features)")

    def _build_file_index(self):
        """Build index mapping global sequence IDs to (file_idx, position_in_file).

        This enables efficient batch loading without loading all files.

        Example:
            File 0: 903 sequences  → global indices 0-902
            File 1: 918 sequences  → global indices 903-1820
            File 2: 954 sequences  → global indices 1821-2774

            cumulative_positions = [0, 903, 1821, 2775, ...]
        """
        self.file_sizes = []  # Number of sequences in each file
        self.cumulative_positions = [0]  # Cumulative count: [0, size0, size0+size1, ...]

        for file_path in self.data_files:
            # Load just to check size (lazy - we don't keep the data)
            data = np.load(file_path)
            file_size = data.shape[0]

            self.file_sizes.append(file_size)
            # Cumulative sum: next position = last position + this file's size
            self.cumulative_positions.append(
                self.cumulative_positions[-1] + file_size
            )

        logger.info(f"Built file index: {len(self.data_files)} files, "
                    f"{self.total_sequences:,} total sequences")

    def load_all(self) -> np.ndarray:
        """Load all data into memory.

        Returns:
            Array of shape (total_sequences, sequence_length, num_features)
        """
        logger.info("Loading all data into memory...")

        data_arrays = [np.load(f) for f in self.data_files]
        combined = np.concatenate(data_arrays, axis=0)

        logger.info(f"Loaded {combined.shape[0]:,} sequences ({combined.nbytes / 1e6:.1f} MB)")
        return combined

    def load_batch(self, batch_size: int, shuffle: bool = True) -> np.ndarray:
        """Load a random batch of sequences efficiently.

        This method only loads the files that contain sequences in the batch,
        not all 15 files.

        Args:
            batch_size: Number of sequences to load
            shuffle: Whether to randomly sample sequences

        Returns:
            Array of shape (batch_size, sequence_length, num_features)
        """
        # Step 1: Pick which sequences we want (global indices)
        if shuffle:
            # Random sampling without replacement
            global_indices = np.random.choice(
                self.total_sequences,
                size=batch_size,
                replace=False
            )
        else:
            # Sequential: take first batch_size sequences
            global_indices = np.arange(batch_size)

        # Step 2: Map each global index to (file_idx, position_within_file)
        batch_sequences = []

        for global_idx in global_indices:
            # Find which file this index belongs to
            # We want: cumulative_positions[file_idx] <= global_idx < cumulative_positions[file_idx+1]
            file_idx = np.searchsorted(self.cumulative_positions, global_idx, side='right') - 1

            # Calculate position within that file
            position_in_file = global_idx - self.cumulative_positions[file_idx]

            # Step 3: Load only this file (numpy caches it if accessed multiple times)
            file_data = np.load(self.data_files[file_idx])

            # Step 4: Extract the specific sequence
            sequence = file_data[position_in_file]
            batch_sequences.append(sequence)

        # Step 5: Stack into batch array
        return np.stack(batch_sequences, axis=0)


class RLTrainer:
    """Trains RL policy on historical data with telemetry integration."""

    def __init__(
            self,
            policy: ExecutionPolicy,
            exec_config: ExecutionConfig,
            rl_config: RLTrainingConfig,
            action_converter: ActionConverter,
            device: str = "cpu",
    ):
        """
        Args:
            policy: Execution policy network
            exec_config: Execution environment configuration
            rl_config: RL training configuration
            action_converter: Converts policy actions to orders
            device: Training device ("cpu", "cuda", "mps")
        """
        self.policy = policy.to(device)
        self.exec_config = exec_config
        self.rl_config = rl_config
        self.action_converter = action_converter
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=rl_config.learning_rate
        )

        logger.info(f"Initialized trainer on device: {device}")

    def train_episode(self, price_data: np.ndarray, seed: int = 42) -> dict:
        """Train on a single episode (price sequence).

        Args:
            price_data: Array of shape (sequence_length, num_features)
                       Features: [open, high, low, close, volume, returns, log_returns]
            seed: Random seed for environment

        Returns:
            Dict with episode statistics
        """
        with tracer.start_as_current_span("train_episode") as span:
            # Create environment with this price data
            env = SimulatedRetailExecutionEnv(self.exec_config, seed=seed)
            obs = env.reset()

            # Episode storage
            log_probs = []
            rewards = []

            # Run episode
            for step in range(len(price_data)):
                # Convert to tensor
                obs_tensor = self._obs_to_tensor(obs, price_data[step])

                # Forward pass through policy (returns policy_logits, value)
                logits, value = self.policy(obs_tensor.unsqueeze(0))  # (1, 3), (1, 1)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                # Convert to order and execute
                order = self.action_converter.policy_to_order(
                    action.item(),
                    obs["mid_price"],
                    obs["inventory"],
                    obs["cash"]
                )
                obs, reward, done, info = env.step(order)

                # Store for learning
                log_probs.append(dist.log_prob(action))
                rewards.append(reward)

                if done:
                    break

            # Compute returns and update policy
            returns = self._compute_returns(rewards)
            policy_loss = self._compute_policy_loss(log_probs, returns)

            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            # Episode stats
            stats = {
                "total_reward": sum(rewards),
                "episode_length": len(rewards),
                "final_portfolio_value": obs["portfolio_value"],
                "total_return": (obs[
                                     "portfolio_value"] - self.exec_config.initial_cash) / self.exec_config.initial_cash,
                "policy_loss": policy_loss.item(),
            }

            # Add to telemetry span
            for key, value in stats.items():
                span.set_attribute(key, value)

            return stats

    def _obs_to_tensor(self, obs: dict, price_features: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor for policy network.

        Args:
            obs: Environment observation dict
            price_features: Current price features [open, high, low, close, volume, returns, log_returns]

        Returns:
            Tensor of shape (feature_dim,)
        """
        # Combine environment state with price features
        features = np.array([
            obs["inventory"],
            obs["cash"] / 50_000.0,  # Normalize cash
            obs["portfolio_value"] / 50_000.0,  # Normalize portfolio value
            obs["mid_price"],
            *price_features  # OHLC + volume + returns
        ], dtype=np.float32)

        return torch.from_numpy(features).to(self.device)

    def _compute_returns(self, rewards: list[float]) -> torch.Tensor:
        """Compute discounted returns for policy gradient.

        Args:
            rewards: List of rewards from episode

        Returns:
            Tensor of discounted returns
        """
        returns = []
        R = 0.0

        for r in reversed(rewards):
            R = r + self.rl_config.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def _compute_policy_loss(
            self, log_probs: list[torch.Tensor], returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss with entropy regularization.

        Args:
            log_probs: Log probabilities of actions taken
            returns: Discounted returns

        Returns:
            Policy loss tensor
        """
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * returns).mean()

        # Entropy regularization (encourage exploration)
        # TODO: Add entropy term

        return policy_loss


def main():
    parser = argparse.ArgumentParser(description="Train RL policy on historical FX data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prepared"),
        help="Directory with prepared .npy files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=100,
        help="Number of episodes per epoch"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device selection
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    elif args.gpu and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"

    logger.info("=" * 80)
    logger.info("RL TRAINING ON HISTORICAL DATA")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Episodes per epoch: {args.episodes_per_epoch}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Load data
    data_loader = HistoricalDataLoader(args.data_dir)
    all_data = data_loader.load_all()

    # Create configurations
    SignalModelConfig(
        num_features=11,
        # inventory, cash, portfolio_value, mid_price, open, high, low, close, volume, returns, log_returns
        hidden_size_lstm=64,
    )

    policy_config = PolicyConfig(
        input_dim=11,  # Match num_features
        num_actions=3,  # SELL, HOLD, BUY
    )

    exec_config = ExecutionConfig(
        initial_cash=50_000.0,
        commission_pct=0.0001,
        spread=0.00015,
        time_horizon=390,
        reward_type="incremental_pnl",  # From Phase 4
    )

    rl_config = RLTrainingConfig(
        epochs=args.epochs,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
    )

    # Create policy and trainer
    policy = ExecutionPolicy(policy_config)
    action_converter = ActionConverter(
        risk_per_trade=0.02,
        max_position=10.0,
        use_dynamic_sizing=True,
    )

    trainer = RLTrainer(policy, exec_config, rl_config, action_converter, device=device)

    # Training loop
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EPOCH {epoch + 1}/{args.epochs}")
        logger.info(f"{'=' * 80}")

        epoch_stats = {
            "total_reward": 0.0,
            "total_return": 0.0,
            "policy_loss": 0.0,
        }

        # Sample episodes for this epoch
        episode_indices = np.random.choice(len(all_data), size=args.episodes_per_epoch, replace=False)

        for i, idx in enumerate(episode_indices):
            price_data = all_data[idx]  # (390, 7)

            stats = trainer.train_episode(price_data, seed=args.seed + epoch * 1000 + i)

            epoch_stats["total_reward"] += stats["total_reward"]
            epoch_stats["total_return"] += stats["total_return"]
            epoch_stats["policy_loss"] += stats["policy_loss"]

            if (i + 1) % 20 == 0:
                logger.info(f"  Episode {i + 1}/{args.episodes_per_epoch} | "
                            f"Reward: {stats['total_reward']:.2f} | "
                            f"Return: {stats['total_return']:.2%} | "
                            f"Loss: {stats['policy_loss']:.4f}")

        # Epoch summary
        avg_reward = epoch_stats["total_reward"] / args.episodes_per_epoch
        avg_return = epoch_stats["total_return"] / args.episodes_per_epoch
        avg_loss = epoch_stats["policy_loss"] / args.episodes_per_epoch

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Avg Reward: {avg_reward:.2f}")
        logger.info(f"  Avg Return: {avg_return:.2%}")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.output_dir / f"policy_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "avg_return": avg_return,
            }, checkpoint_path)
            logger.info(f"  Saved checkpoint: {checkpoint_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
