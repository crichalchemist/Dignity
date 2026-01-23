"""Run hyperparameter tuning to find optimal trading configurations.

This script performs systematic hyperparameter search to optimize:
1. Position sizing (risk_per_trade, max_position)
2. Reward engineering (reward_type, penalties)
3. Risk management (drawdown limits)

Usage:
    python scripts/run_hyperparameter_tuning.py --mode random --samples 50 --episodes 10
    python scripts/run_hyperparameter_tuning.py --mode grid --max-configs 30

Results are saved to tuning_results/ directory with detailed metrics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import RLTrainingConfig
from execution.simulated_retail_env import ExecutionConfig
from train.hyperparameter_tuning import HyperparameterGrid, HyperparameterTuner
from utils.logger import get_logger

logger = get_logger(__name__)


def create_conservative_grid() -> HyperparameterGrid:
    """Conservative search grid for production use."""
    return HyperparameterGrid(
        risk_per_trade=[0.01, 0.015, 0.02],
        max_position=[5.0, 10.0],
        use_dynamic_sizing=[True],
        reward_type=["incremental_pnl", "cost_aware"],
        cost_penalty_weight=[0.0, 1.0, 3.0],
        drawdown_penalty_weight=[500.0, 1000.0],
        max_drawdown_pct=[0.15, 0.20],
        enable_stop_loss=[False],
        learning_rate=[1e-4, 3e-4],
        gamma=[0.97, 0.99],
        entropy_coef=[0.01],
    )


def create_aggressive_grid() -> HyperparameterGrid:
    """Aggressive search grid for research/exploration."""
    return HyperparameterGrid(
        risk_per_trade=[0.015, 0.02, 0.025, 0.03],
        max_position=[10.0, 15.0, 20.0],
        use_dynamic_sizing=[True],
        reward_type=["incremental_pnl", "cost_aware", "sharpe"],
        cost_penalty_weight=[0.0, 3.0, 5.0],
        drawdown_penalty_weight=[0.0, 1000.0, 2000.0],
        max_drawdown_pct=[0.20, 0.25, 0.30],
        enable_stop_loss=[False],
        learning_rate=[1e-4, 3e-4, 1e-3],
        gamma=[0.95, 0.97, 0.99],
        entropy_coef=[0.001, 0.01, 0.05],
    )


def create_position_sizing_grid() -> HyperparameterGrid:
    """Focused grid for position sizing optimization only."""
    return HyperparameterGrid(
        risk_per_trade=[0.01, 0.012, 0.015, 0.017, 0.02, 0.022, 0.025, 0.03],
        max_position=[5.0, 7.5, 10.0, 12.5, 15.0],
        use_dynamic_sizing=[True],
        reward_type=["incremental_pnl"],  # Fix reward type
        cost_penalty_weight=[0.0],
        drawdown_penalty_weight=[1000.0],  # Fix penalty
        max_drawdown_pct=[0.20],  # Fix risk limit
        enable_stop_loss=[False],
        learning_rate=[3e-4],  # Fix RL params
        gamma=[0.99],
        entropy_coef=[0.01],
    )


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for RL trading system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "grid"],
        default="random",
        help="Search mode: random or grid search",
    )
    parser.add_argument(
        "--grid-type",
        type=str,
        choices=["conservative", "aggressive", "position_sizing"],
        default="conservative",
        help="Predefined grid type to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of random samples (for random search)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to test (for grid search)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per configuration",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=390,
        help="Length of each episode in steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Directory to save tuning results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Select grid
    if args.grid_type == "conservative":
        grid = create_conservative_grid()
    elif args.grid_type == "aggressive":
        grid = create_aggressive_grid()
    elif args.grid_type == "position_sizing":
        grid = create_position_sizing_grid()
    else:
        raise ValueError(f"Unknown grid type: {args.grid_type}")

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Grid type: {args.grid_type}")
    logger.info(f"Grid size: {grid.get_grid_size()} configurations")
    logger.info(f"Episodes per config: {args.episodes}")
    logger.info(f"Episode length: {args.episode_length} steps")
    logger.info("=" * 80)

    # Base configurations
    base_exec_config = ExecutionConfig(
        initial_cash=50_000.0,
        commission_pct=0.0001,  # 1 basis point (typical FX)
        spread=0.00015,  # 1.5 pips for EUR/USD
        variable_spread=True,
        time_horizon=args.episode_length,
    )

    base_rl_config = RLTrainingConfig(
        epochs=5,
        learning_rate=3e-4,
        entropy_coef=0.01,
        gamma=0.99,
    )

    # Create tuner
    tuner = HyperparameterTuner(base_exec_config, base_rl_config, output_dir=args.output_dir)

    # Run tuning
    if args.mode == "random":
        logger.info(f"Running random search with {args.samples} samples...")
        results = tuner.random_search(
            grid,
            n_samples=args.samples,
            n_episodes=args.episodes,
            episode_length=args.episode_length,
            seed=args.seed,
        )
    else:  # grid search
        logger.info("Running grid search...")
        results = tuner.grid_search(
            grid,
            n_episodes=args.episodes,
            episode_length=args.episode_length,
            max_configs=args.max_configs,
        )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Tested {len(results)} configurations")

    # Print top 5 by risk-adjusted return
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS (Risk-Adjusted Return)")
    logger.info("=" * 80)

    top_5_risk_adj = tuner.get_top_configs(n=5, metric="risk_adjusted_return")
    for i, result in enumerate(top_5_risk_adj, 1):
        logger.info(f"\n[{i}] Config:")
        for k, v in result.config.items():
            logger.info(f"      {k}: {v}")
        logger.info(f"    Return: {result.total_return:.2%}")
        logger.info(f"    Sharpe: {result.sharpe_ratio:.3f}")
        logger.info(f"    Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"    Risk-Adjusted Return: {result.risk_adjusted_return:.3f}")
        logger.info(f"    Avg Trades/Episode: {result.avg_trade_size:.1f}")

    # Print top 5 by total return
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS (Total Return)")
    logger.info("=" * 80)

    top_5_return = tuner.get_top_configs(n=5, metric="total_return")
    for i, result in enumerate(top_5_return, 1):
        logger.info(f"\n[{i}] Return: {result.total_return:.2%} | Sharpe: {result.sharpe_ratio:.3f}")
        logger.info(f"    risk_per_trade: {result.config['risk_per_trade']}")
        logger.info(f"    max_position: {result.config['max_position']}")
        logger.info(f"    reward_type: {result.config['reward_type']}")

    # Print top 5 by Sharpe ratio
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS (Sharpe Ratio)")
    logger.info("=" * 80)

    top_5_sharpe = tuner.get_top_configs(n=5, metric="sharpe_ratio")
    for i, result in enumerate(top_5_sharpe, 1):
        logger.info(f"\n[{i}] Sharpe: {result.sharpe_ratio:.3f} | Return: {result.total_return:.2%}")
        logger.info(f"    risk_per_trade: {result.config['risk_per_trade']}")
        logger.info(f"    drawdown_penalty_weight: {result.config['drawdown_penalty_weight']}")
        logger.info(f"    max_drawdown_pct: {result.config['max_drawdown_pct']}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Results saved to: {tuner.output_dir / 'tuning_results.json'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
