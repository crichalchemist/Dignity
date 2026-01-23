"""Hyperparameter tuning framework for RL trading system.

This module provides tools for systematic hyperparameter optimization across
position sizing, reward engineering, and RL training parameters.

Features:
- Grid search and random search strategies
- Multi-objective optimization (return vs. risk)
- Parallel execution of tuning runs
- Results tracking and visualization
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

import itertools
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from config.config import RLTrainingConfig
from execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv
from train.core.env_based_rl_training import ActionConverter
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HyperparameterGrid:
    """Defines hyperparameter search space."""

    # Position sizing parameters
    risk_per_trade: list[float] = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.025, 0.03])
    max_position: list[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0])
    use_dynamic_sizing: list[bool] = field(default_factory=lambda: [True])

    # Reward engineering parameters
    reward_type: list[str] = field(default_factory=lambda: ["incremental_pnl", "cost_aware", "sharpe"])
    cost_penalty_weight: list[float] = field(default_factory=lambda: [0.0, 1.0, 3.0, 5.0])
    drawdown_penalty_weight: list[float] = field(default_factory=lambda: [0.0, 500.0, 1000.0, 2000.0])

    # Risk management parameters
    max_drawdown_pct: list[float] = field(default_factory=lambda: [0.15, 0.20, 0.25])
    enable_stop_loss: list[bool] = field(default_factory=lambda: [False])
    stop_loss_pct: list[float] = field(default_factory=lambda: [0.02])

    # RL training parameters
    learning_rate: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    gamma: list[float] = field(default_factory=lambda: [0.95, 0.97, 0.99])
    entropy_coef: list[float] = field(default_factory=lambda: [0.001, 0.01, 0.05])

    def get_grid_size(self) -> int:
        """Calculate total number of configurations in grid."""
        sizes = []
        for _field_name, field_value in asdict(self).items():
            if isinstance(field_value, list):
                sizes.append(len(field_value))
        return int(np.prod(sizes))

    def sample_random(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Sample N random configurations from the grid.

        Args:
            n: Number of configurations to sample
            seed: Random seed for reproducibility

        Returns:
            List of configuration dictionaries
        """
        if seed is not None:
            random.seed(seed)

        configs = []
        for _ in range(n):
            config = {}
            for field_name, field_value in asdict(self).items():
                if isinstance(field_value, list) and len(field_value) > 0:
                    config[field_name] = random.choice(field_value)
            configs.append(config)

        return configs


@dataclass
class TuningResult:
    """Results from a single hyperparameter configuration."""

    config: dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_trade_size: float
    total_commission: float
    total_spread: float
    final_portfolio_value: float
    episode_length: int

    @property
    def risk_adjusted_return(self) -> float:
        """Return penalized by risk (simple metric)."""
        return self.total_return - (self.max_drawdown * 0.5)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config,
            "metrics": {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "total_trades": self.total_trades,
                "avg_trade_size": self.avg_trade_size,
                "total_commission": self.total_commission,
                "total_spread": self.total_spread,
                "final_portfolio_value": self.final_portfolio_value,
                "episode_length": self.episode_length,
                "risk_adjusted_return": self.risk_adjusted_return,
            },
        }


class HyperparameterTuner:
    """Framework for hyperparameter optimization."""

    def __init__(
            self,
            base_execution_config: ExecutionConfig,
            base_rl_config: RLTrainingConfig,
            output_dir: str = "tuning_results",
    ):
        """
        Args:
            base_execution_config: Base execution environment configuration
            base_rl_config: Base RL training configuration
            output_dir: Directory to save tuning results
        """
        self.base_execution_config = base_execution_config
        self.base_rl_config = base_rl_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[TuningResult] = []

    def grid_search(
            self,
            grid: HyperparameterGrid,
            n_episodes: int = 10,
            episode_length: int = 390,
            max_configs: int | None = None,
    ) -> list[TuningResult]:
        """Run grid search over hyperparameter space.

        Args:
            grid: Hyperparameter grid defining search space
            n_episodes: Number of episodes to run per configuration
            episode_length: Length of each episode in steps
            max_configs: Maximum number of configurations to test (for partial grid search)

        Returns:
            List of tuning results sorted by risk-adjusted return
        """
        logger.info("Starting grid search...")
        logger.info(f"Grid size: {grid.get_grid_size()} configurations")

        # Generate all configurations
        configs = self._generate_grid_configs(grid)

        if max_configs is not None and len(configs) > max_configs:
            logger.info(f"Limiting to {max_configs} random configurations from grid")
            random.shuffle(configs)
            configs = configs[:max_configs]

        logger.info(f"Testing {len(configs)} configurations")

        # Evaluate each configuration
        for i, config in enumerate(configs):
            logger.info(f"[{i + 1}/{len(configs)}] Testing config: {config}")
            result = self._evaluate_config(config, n_episodes, episode_length)
            self.results.append(result)

            # Save intermediate results
            if (i + 1) % 10 == 0:
                self._save_results()

        # Save final results
        self._save_results()

        # Sort by risk-adjusted return
        self.results.sort(key=lambda r: r.risk_adjusted_return, reverse=True)

        return self.results

    def random_search(
            self,
            grid: HyperparameterGrid,
            n_samples: int = 50,
            n_episodes: int = 10,
            episode_length: int = 390,
            seed: int | None = None,
    ) -> list[TuningResult]:
        """Run random search over hyperparameter space.

        Args:
            grid: Hyperparameter grid defining search space
            n_samples: Number of random configurations to sample
            n_episodes: Number of episodes to run per configuration
            episode_length: Length of each episode in steps
            seed: Random seed for reproducibility

        Returns:
            List of tuning results sorted by risk-adjusted return
        """
        logger.info(f"Starting random search with {n_samples} samples...")

        # Sample random configurations
        configs = grid.sample_random(n_samples, seed=seed)

        # Evaluate each configuration
        for i, config in enumerate(configs):
            logger.info(f"[{i + 1}/{n_samples}] Testing config: {config}")
            result = self._evaluate_config(config, n_episodes, episode_length)
            self.results.append(result)

            # Save intermediate results
            if (i + 1) % 10 == 0:
                self._save_results()

        # Save final results
        self._save_results()

        # Sort by risk-adjusted return
        self.results.sort(key=lambda r: r.risk_adjusted_return, reverse=True)

        return self.results

    def _generate_grid_configs(self, grid: HyperparameterGrid) -> list[dict[str, Any]]:
        """Generate all configurations from grid."""
        grid_dict = asdict(grid)
        keys = list(grid_dict.keys())
        values = [grid_dict[k] for k in keys]

        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination, strict=False))
            configs.append(config)

        return configs

    def _evaluate_config(
            self, config: dict[str, Any], n_episodes: int, episode_length: int
    ) -> TuningResult:
        """Evaluate a single hyperparameter configuration.

        Args:
            config: Hyperparameter configuration
            n_episodes: Number of episodes to run
            episode_length: Length of each episode

        Returns:
            TuningResult with performance metrics
        """
        # Create execution config with hyperparameters
        exec_config = ExecutionConfig(
            initial_cash=self.base_execution_config.initial_cash,
            time_horizon=episode_length,
            commission_pct=self.base_execution_config.commission_pct,
            spread=self.base_execution_config.spread,
            variable_spread=self.base_execution_config.variable_spread,
            reward_type=config.get("reward_type", "incremental_pnl"),
            cost_penalty_weight=config.get("cost_penalty_weight", 0.0),
            drawdown_penalty_weight=config.get("drawdown_penalty_weight", 0.0),
            max_drawdown_pct=config.get("max_drawdown_pct", 0.20),
            enable_stop_loss=config.get("enable_stop_loss", False),
            stop_loss_pct=config.get("stop_loss_pct", 0.02),
            enable_drawdown_limit=True,
        )

        # Create action converter with position sizing params
        action_converter = ActionConverter(
            risk_per_trade=config.get("risk_per_trade", 0.02),
            max_position=config.get("max_position", 10.0),
            use_dynamic_sizing=config.get("use_dynamic_sizing", True),
        )

        # Run episodes
        episode_returns = []
        episode_drawdowns = []
        total_trades = 0
        total_commission = 0.0
        total_spread = 0.0
        final_values = []

        for ep in range(n_episodes):
            env = SimulatedRetailExecutionEnv(exec_config, seed=42 + ep)
            obs = env.reset()

            # Simple policy: random actions for now (replace with trained policy later)
            for _step in range(episode_length):
                # Random policy for tuning (not optimal, just for hyperparameter comparison)
                action_idx = random.choice([0, 1, 2])  # SELL, HOLD, BUY
                order = action_converter.policy_to_order(
                    action_idx, obs["mid_price"], obs["inventory"], obs["cash"]
                )
                obs, reward, done, info = env.step(order)

                if done:
                    break

            # Collect episode statistics
            episode_return = (obs["portfolio_value"] - exec_config.initial_cash) / exec_config.initial_cash
            episode_returns.append(episode_return)

            # Calculate drawdown
            drawdown = (env._peak_portfolio_value - obs["portfolio_value"]) / env._peak_portfolio_value
            episode_drawdowns.append(drawdown)

            total_trades += len(env._fill_events)
            total_commission += env._commission_paid
            total_spread += env._spread_paid
            final_values.append(obs["portfolio_value"])

        # Calculate aggregate metrics
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        max_drawdown = np.max(episode_drawdowns)
        avg_trade_size = total_trades / n_episodes if n_episodes > 0 else 0

        return TuningResult(
            config=config,
            total_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            avg_trade_size=avg_trade_size,
            total_commission=total_commission / n_episodes,
            total_spread=total_spread / n_episodes,
            final_portfolio_value=np.mean(final_values),
            episode_length=episode_length,
        )

    def _save_results(self) -> None:
        """Save tuning results to JSON file."""
        results_dict = {
            "base_config": {
                "execution": asdict(self.base_execution_config),
                "rl_training": asdict(self.base_rl_config),
            },
            "results": [r.to_dict() for r in self.results],
        }

        output_file = self.output_dir / "tuning_results.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Saved results to {output_file}")

    def get_top_configs(self, n: int = 5, metric: str = "risk_adjusted_return") -> list[TuningResult]:
        """Get top N configurations by specified metric.

        Args:
            n: Number of top configurations to return
            metric: Metric to sort by ("risk_adjusted_return", "total_return", "sharpe_ratio")

        Returns:
            Top N tuning results
        """
        if metric == "risk_adjusted_return":
            sorted_results = sorted(self.results, key=lambda r: r.risk_adjusted_return, reverse=True)
        elif metric == "total_return":
            sorted_results = sorted(self.results, key=lambda r: r.total_return, reverse=True)
        elif metric == "sharpe_ratio":
            sorted_results = sorted(self.results, key=lambda r: r.sharpe_ratio, reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted_results[:n]


def main():
    """Example hyperparameter tuning run."""
    # Base configurations
    base_exec_config = ExecutionConfig(
        initial_cash=50_000.0,
        commission_pct=0.0001,
        spread=0.00015,
        variable_spread=True,
        time_horizon=390,
    )

    base_rl_config = RLTrainingConfig(
        epochs=5,
        learning_rate=3e-4,
        entropy_coef=0.01,
        gamma=0.99,
    )

    # Define search grid (smaller for demo)
    grid = HyperparameterGrid(
        risk_per_trade=[0.01, 0.02, 0.03],
        max_position=[5.0, 10.0, 15.0],
        reward_type=["incremental_pnl", "cost_aware"],
        cost_penalty_weight=[0.0, 3.0],
        drawdown_penalty_weight=[0.0, 1000.0],
    )

    logger.info(f"Grid size: {grid.get_grid_size()} configurations")

    # Create tuner
    tuner = HyperparameterTuner(base_exec_config, base_rl_config)

    # Run random search (faster for large grids)
    tuner.random_search(grid, n_samples=20, n_episodes=5, episode_length=100, seed=42)

    # Print top 5 configurations
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS BY RISK-ADJUSTED RETURN:")
    logger.info("=" * 80)

    top_5 = tuner.get_top_configs(n=5)
    for i, result in enumerate(top_5, 1):
        logger.info(f"\n[{i}] Config: {result.config}")
        logger.info(f"    Return: {result.total_return:.2%}")
        logger.info(f"    Sharpe: {result.sharpe_ratio:.3f}")
        logger.info(f"    Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"    Risk-Adjusted Return: {result.risk_adjusted_return:.3f}")


if __name__ == "__main__":
    main()
