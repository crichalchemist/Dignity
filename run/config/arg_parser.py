"""Shared argument parser factories for training, evaluation, and pipeline scripts.

This module provides reusable functions to add common argument groups to ArgumentParser
instances, reducing code duplication and ensuring consistency across entry points.
"""

import argparse

from config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DC_THRESHOLD,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_WEIGHT_DECAY,
)


def add_data_preparation_args(parser: argparse.ArgumentParser) -> None:
    """Add common data preparation arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    parser.add_argument(
        "--years", default=None, help="Comma-separated years to include (default: all available)"
    )
    parser.add_argument(
        "--input-root", default="output_central", help="Root containing Central-time zips"
    )
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Prediction horizon")
    parser.add_argument(
        "--lookahead-window", type=int, default=None, help="Lookahead for auxiliary targets"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top-K future returns/prices predictions"
    )
    parser.add_argument(
        "--predict-sell-now", action="store_true", help="Enable sell-now auxiliary head"
    )
    parser.add_argument(
        "--task-type", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)


def add_feature_engineering_args(parser: argparse.ArgumentParser) -> None:
    """Add common feature engineering arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--feature-groups", default="all", help="Comma-separated feature groups to include or 'all'"
    )
    parser.add_argument(
        "--exclude-feature-groups", default=None, help="Comma-separated feature groups to drop"
    )
    parser.add_argument(
        "--sma-windows", default="10,20,50", help="Comma-separated SMA window lengths"
    )
    parser.add_argument("--ema-windows", default="10,20,50", help="Comma-separated EMA spans")
    parser.add_argument("--rsi-window", type=int, default=14, help="Window length for RSI")
    parser.add_argument(
        "--bollinger-window", type=int, default=20, help="Window length for Bollinger bands"
    )
    parser.add_argument(
        "--bollinger-num-std",
        type=float,
        default=2.0,
        help="Std dev multiplier for Bollinger bands",
    )
    parser.add_argument("--atr-window", type=int, default=14, help="Window length for ATR")
    parser.add_argument(
        "--short-vol-window", type=int, default=10, help="Short window for volatility clustering"
    )
    parser.add_argument(
        "--long-vol-window", type=int, default=50, help="Long window for volatility clustering"
    )
    parser.add_argument(
        "--spread-windows", default="20", help="Comma-separated windows for normalized spread stats"
    )
    parser.add_argument(
        "--imbalance-smoothing",
        type=int,
        default=5,
        help="Rolling mean window for wick/body imbalance",
    )


def add_intrinsic_time_args(parser: argparse.ArgumentParser) -> None:
    """Add intrinsic time (directional-change) arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--intrinsic-time",
        action="store_true",
        help="Convert minute bars to intrinsic-time bars via directional-change events.",
    )
    parser.add_argument(
        "--dc-threshold-up",
        type=float,
        default=DEFAULT_DC_THRESHOLD,
        help=f"Fractional increase needed to flag an upward directional change (e.g., {DEFAULT_DC_THRESHOLD}=0.1%).",
    )
    parser.add_argument(
        "--dc-threshold-down",
        type=float,
        default=None,
        help="Fractional decrease needed to flag a downward directional change. Defaults to dc-threshold-up.",
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--device", default="cuda")


def add_dataloader_args(parser: argparse.ArgumentParser) -> None:
    """Add DataLoader-specific arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader workers for parallel loading",
    )
    parser.add_argument(
        "--pin-memory", action="store_true", default=True, help="Pin DataLoader memory for GPU"
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help="DataLoader prefetch factor",
    )


def add_checkpoint_args(
        parser: argparse.ArgumentParser, include_signal_policy: bool = False
) -> None:
    """Add checkpoint path arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
        include_signal_policy: If True, add signal and policy checkpoint arguments
    """
    parser.add_argument(
        "--checkpoint-path", default="models/best_model.pt", help="Path to model checkpoint"
    )
    if include_signal_policy:
        parser.add_argument(
            "--signal-checkpoint-path",
            default="models/signal_{pair}.pt",
            help="Path to signal checkpoint (format string {pair} supported)",
        )
        parser.add_argument(
            "--policy-checkpoint-path",
            default="models/policy_{pair}.pt",
            help="Path to policy checkpoint (format string {pair} supported)",
        )


def add_risk_args(parser: argparse.ArgumentParser) -> None:
    """Add risk management arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--disable-risk",
        action="store_true",
        help="Disable risk manager gating during training/evaluation.",
    )


def add_auxiliary_head_weights(parser: argparse.ArgumentParser) -> None:
    """Add auxiliary head weight arguments for multi-task learning.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument("--max-return-weight", type=float, default=1.0)
    parser.add_argument("--topk-return-weight", type=float, default=1.0)
    parser.add_argument("--topk-price-weight", type=float, default=1.0)
    parser.add_argument("--sell-now-weight", type=float, default=1.0)


def add_rl_training_args(parser: argparse.ArgumentParser) -> None:
    """Add reinforcement learning specific training arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--pretrain-epochs", type=int, default=5, help="epochs for signal pretraining"
    )
    parser.add_argument(
        "--policy-epochs", type=int, default=5, help="epochs for execution policy training"
    )
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--detach-signal", action="store_true", help="freeze signal encoder during policy training"
    )


def add_amp_args(parser: argparse.ArgumentParser) -> None:
    """Add automatic mixed precision (AMP) training arguments.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--use-amp", action="store_true", help="Enable mixed precision (AMP) training for GPU"
    )
