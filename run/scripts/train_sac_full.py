"""Full SAC training pipeline with all enhancements.

This script integrates all Week 1-3 components:
- Multi-timeframe features (76 dimensions)
- Frozen signal model for state embeddings
- SAC agent with experience replay
- Limit order engine with learned aggressiveness
- Target networks and automatic entropy tuning

Usage:
    python scripts/train_sac_full.py --pair EURUSD --episodes 200
    python scripts/train_sac_full.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/")

# Import directly from module to bypass __init__.py with Python version issues
import importlib.util

from config.config import SignalModelConfig
from execution.limit_order_engine import LimitOrderConfig
from execution.simulated_retail_env import ExecutionConfig
from models.signal_policy import SignalModel

spec = importlib.util.spec_from_file_location("sac_training", "/Volumes/Containers/Sequence/train/core/sac_training.py")
sac_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sac_training)
SACConfig = sac_training.SACConfig
train_with_sac = sac_training.train_with_sac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def load_training_data(pair: str, data_dir: str = "data/prepared") -> np.ndarray:
    """Load prepared training data for a currency pair.

    Args:
        pair: Currency pair (e.g., "EURUSD")
        data_dir: Directory containing prepared .npy files

    Returns:
        Training data array [num_steps, num_features]
    """
    data_path = Path(data_dir) / pair.lower()

    # Look for prepared files in pair subdirectory
    pattern = "*.npy"
    files = sorted(data_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No prepared data found for {pair} in {data_path} (pattern: {pattern})")

    logger.info(f"Found {len(files)} files for {pair}")

    # Load and concatenate all files
    data_arrays = []
    for file in files:
        logger.info(f"Loading {file.name}...")
        data = np.load(file)
        data_arrays.append(data)

    # Concatenate along time axis
    train_data = np.concatenate(data_arrays, axis=0)

    logger.info(f"Loaded training data: {train_data.shape}")

    return train_data


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent on FX data")

    # Data arguments
    parser.add_argument("--pair", type=str, required=True, help="Currency pair (e.g., EURUSD)")
    parser.add_argument("--data-dir", type=str, default="data/prepared", help="Directory with prepared data")

    # Model arguments
    parser.add_argument("--num-features", type=int, default=76, help="Number of input features")
    parser.add_argument("--sequence-length", type=int, default=100, help="Price sequence length")
    parser.add_argument("--use-direction-head", action="store_true", default=True,
                        help="Use signal model direction head")

    # Training arguments
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for SAC updates")
    parser.add_argument("--updates-per-step", type=int, default=2, help="Gradient updates per env step")
    parser.add_argument("--warmup-steps", type=int, default=1_000, help="Random exploration steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Execution arguments
    parser.add_argument("--initial-cash", type=float, default=50_000.0, help="Initial cash")
    parser.add_argument("--use-limit-orders", action="store_true", default=True, help="Use limit order engine")
    parser.add_argument("--use-prioritized-replay", action="store_true", help="Use prioritized experience replay")

    # Logging and checkpoints
    parser.add_argument("--log-frequency", type=int, default=10, help="Log every N episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sac", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")

    args = parser.parse_args()

    # Print configuration
    logger.info("=" * 70)
    logger.info("FULL SAC TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Currency Pair: {args.pair}")
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Features: {args.num_features} (multi-timeframe)")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Buffer Size: {args.buffer_size:,}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Updates/Step: {args.updates_per_step}")
    logger.info(f"Limit Orders: {args.use_limit_orders}")
    logger.info(f"Prioritized Replay: {args.use_prioritized_replay}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

    # Load training data
    logger.info("\nLoading training data...")
    try:
        train_data = load_training_data(args.pair, args.data_dir)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please run data preparation first:")
        logger.error("  python scripts/validate_training_data.py data/histdata/EURUSD_*.csv")
        return 1

    # Validate data dimensions
    if train_data.shape[1] != args.num_features:
        logger.error(f"Data has {train_data.shape[1]} features, expected {args.num_features}")
        logger.error("Update --num-features or reprocess data with multi-timeframe features")
        return 1

    # Create configurations
    logger.info("\nInitializing models...")

    signal_cfg = SignalModelConfig(
        num_features=args.num_features,
        use_direction_head=args.use_direction_head,
    )

    signal_model = SignalModel(signal_cfg)
    logger.info(f"✓ Signal model created (embedding_dim={signal_model.signal_dim})")

    sac_config = SACConfig(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        gamma=args.gamma,
        auto_entropy_tuning=True,
        use_prioritized_replay=args.use_prioritized_replay,
    )

    env_config = ExecutionConfig(
        use_limit_order_engine=args.use_limit_orders,
        limit_order_config=LimitOrderConfig() if args.use_limit_orders else None,
        initial_cash=args.initial_cash,
    )

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.pair.lower()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    logger.info("\nStarting SAC training...")
    logger.info("=" * 70)

    results = train_with_sac(
        signal_model=signal_model,
        sac_config=sac_config,
        signal_cfg=signal_cfg,
        env_config=env_config,
        train_data=train_data,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        log_frequency=args.log_frequency,
        checkpoint_dir=str(checkpoint_dir),
        device=args.device,
        sequence_length=args.sequence_length,
    )

    # Print final results
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total Episodes: {len(results['episode_rewards'])}")
    logger.info(f"Total Steps: {results['total_steps']}")
    logger.info(f"Final Buffer Size: {results['final_buffer_size']}")
    logger.info(f"Mean Reward (all): {np.mean(results['episode_rewards']):.2f}")
    logger.info(f"Mean Reward (last 20): {np.mean(results['episode_rewards'][-20:]):.2f}")
    logger.info(f"Mean Length (all): {np.mean(results['episode_lengths']):.0f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
