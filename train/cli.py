"""Command-line interface for training Dignity models."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DignityConfig
from train.loop import _selection_metric, fit, prepare_data


def main(argv: list[str] | None = None) -> int:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Dignity models")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args(argv)

    # Load config
    print(f"Loading config from {args.config}")
    config = DignityConfig.from_yaml(args.config)
    print(config)

    print("\nGenerating synthetic data...")
    df, data = prepare_data(config)

    if data.epsilon_spent:
        spent = ", ".join(f"{k} {v:.3f}" for k, v in data.epsilon_spent.items())
        print(f"Privacy: ε spent per split: {spent}")

    print(
        f"Train/Val/Test windows: {len(data.train.X)}/{len(data.val.X)}/"
        f"{len(data.test.X)}"
    )

    print(f"\nStarting training for {config.train.epochs} epochs...")
    result = fit(
        config,
        data,
        seed=config.seed,
        run_dir=config.train.checkpoint_dir,
        resume_from=args.resume,
    )

    print(
        f"best {_selection_metric(config)} {result.best_metric:.4f} "
        f"-> {result.best_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
