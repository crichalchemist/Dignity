"""
Bayesian hyperparameter optimization for RL trading system.

This module implements intelligent hyperparameter search using Gaussian Process
surrogate models and acquisition functions (Expected Improvement, UCB).

Key advantages over grid/random search:
- Learns from previous trials to suggest promising configurations
- Converges faster to optimal hyperparameters
- Provides hyperparameter importance analysis
- Supports parallel evaluation and early stopping
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

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from skopt import dump, gp_minimize
    from skopt.plots import plot_convergence
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Run: pip install scikit-optimize")

from config.config import RLTrainingConfig
from execution.simulated_retail_env import ExecutionConfig
from train.hyperparameter_tuning import HyperparameterTuner
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BayesianSearchSpace:
    """Defines hyperparameter search space for Bayesian optimization.

    Uses scikit-optimize's dimension types:
    - Real: Continuous values (e.g., learning rate)
    - Integer: Discrete integers (e.g., batch size)
    - Categorical: Discrete categories (e.g., reward type)
    """

    # Position sizing (continuous)
    risk_per_trade: tuple[float, float] = field(default=(0.005, 0.05))
    max_position: tuple[float, float] = field(default=(5.0, 20.0))

    # Reward engineering (categorical + continuous)
    reward_type: list[str] = field(default_factory=lambda: ["incremental_pnl", "cost_aware", "sharpe"])
    cost_penalty_weight: tuple[float, float] = field(default=(0.0, 10.0))
    drawdown_penalty_weight: tuple[float, float] = field(default=(0.0, 3000.0))

    # Risk management (continuous)
    max_drawdown_pct: tuple[float, float] = field(default=(0.10, 0.30))

    # RL training parameters (continuous, log-scale for learning rate)
    learning_rate_log: tuple[float, float] = field(default=(-5, -3))  # 10^-5 to 10^-3
    gamma: tuple[float, float] = field(default=(0.90, 0.999))
    entropy_coef_log: tuple[float, float] = field(default=(-4, -1))  # 10^-4 to 10^-1

    # PPO-specific parameters
    clip_range: tuple[float, float] = field(default=(0.1, 0.3))
    ppo_epochs: tuple[int, int] = field(default=(2, 8))

    def to_skopt_dimensions(self) -> list[Any]:
        """Convert to scikit-optimize dimension objects."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        dimensions = [
            Real(*self.risk_per_trade, name='risk_per_trade'),
            Real(*self.max_position, name='max_position'),
            Categorical(self.reward_type, name='reward_type'),
            Real(*self.cost_penalty_weight, name='cost_penalty_weight'),
            Real(*self.drawdown_penalty_weight, name='drawdown_penalty_weight'),
            Real(*self.max_drawdown_pct, name='max_drawdown_pct'),
            Real(*self.learning_rate_log, name='learning_rate_log'),
            Real(*self.gamma, name='gamma'),
            Real(*self.entropy_coef_log, name='entropy_coef_log'),
            Real(*self.clip_range, name='clip_range'),
            Integer(*self.ppo_epochs, name='ppo_epochs'),
        ]

        return dimensions

    def get_param_names(self) -> list[str]:
        """Get ordered list of parameter names."""
        return [
            'risk_per_trade', 'max_position', 'reward_type',
            'cost_penalty_weight', 'drawdown_penalty_weight', 'max_drawdown_pct',
            'learning_rate_log', 'gamma', 'entropy_coef_log',
            'clip_range', 'ppo_epochs'
        ]


class BayesianHyperparameterTuner:
    """Bayesian optimization for hyperparameter tuning."""

    def __init__(
            self,
            base_execution_config: ExecutionConfig,
            base_rl_config: RLTrainingConfig,
            output_dir: str = "bayesian_tuning_results",
    ):
        """
        Args:
            base_execution_config: Base execution environment configuration
            base_rl_config: Base RL training configuration
            output_dir: Directory to save tuning results
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )

        self.base_execution_config = base_execution_config
        self.base_rl_config = base_rl_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use HyperparameterTuner for evaluation
        self.evaluator = HyperparameterTuner(
            base_execution_config,
            base_rl_config,
            str(self.output_dir)
        )

        self.optimization_result = None
        self.trial_history = []

    def optimize(
            self,
            search_space: BayesianSearchSpace,
            n_calls: int = 50,
            n_episodes: int = 10,
            episode_length: int = 390,
            n_initial_points: int = 10,
            acq_func: str = "EI",  # Expected Improvement
            random_state: int | None = 42,
            verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run Bayesian optimization.

        Args:
            search_space: Search space definition
            n_calls: Number of optimization iterations
            n_episodes: Episodes per configuration evaluation
            episode_length: Steps per episode
            n_initial_points: Random points before Bayesian optimization starts
            acq_func: Acquisition function ('EI', 'LCB', 'PI')
            random_state: Random seed
            verbose: Print progress

        Returns:
            Dictionary with best configuration and optimization results
        """
        logger.info(f"Starting Bayesian optimization with {n_calls} iterations")
        logger.info(f"Acquisition function: {acq_func}")
        logger.info(f"Initial random points: {n_initial_points}")

        dimensions = search_space.to_skopt_dimensions()
        param_names = search_space.get_param_names()

        # Create objective function
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            """Objective function to minimize (negative Sharpe ratio)."""
            # Convert log-scale parameters
            params['learning_rate'] = 10 ** params.pop('learning_rate_log')
            params['entropy_coef'] = 10 ** params.pop('entropy_coef_log')

            # Add fixed parameters
            params['use_dynamic_sizing'] = True
            params['enable_stop_loss'] = False
            params['stop_loss_pct'] = 0.02

            if verbose:
                logger.info(f"Trial {len(self.trial_history) + 1}/{n_calls}")
                logger.info(f"  Config: {params}")

            # Evaluate configuration
            result = self.evaluator._evaluate_config(params, n_episodes, episode_length)

            # Store trial result
            self.trial_history.append({
                'trial': len(self.trial_history) + 1,
                'config': params.copy(),
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'risk_adjusted_return': result.risk_adjusted_return,
            })

            # Save intermediate results
            if len(self.trial_history) % 5 == 0:
                self._save_trial_history()

            if verbose:
                logger.info(f"  Sharpe: {result.sharpe_ratio:.3f}")
                logger.info(f"  Return: {result.total_return:.2%}")
                logger.info(f"  Drawdown: {result.max_drawdown:.2%}")

            # Return negative Sharpe (we minimize)
            return -result.sharpe_ratio

        # Run Bayesian optimization
        logger.info("Running Gaussian Process optimization...")
        self.optimization_result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=random_state,
            verbose=verbose,
        )

        # Extract best configuration
        best_params_list = self.optimization_result.x
        best_params = dict(zip(param_names, best_params_list, strict=False))

        # Convert log-scale parameters
        best_params['learning_rate'] = 10 ** best_params.pop('learning_rate_log')
        best_params['entropy_coef'] = 10 ** best_params.pop('entropy_coef_log')

        best_score = -self.optimization_result.fun  # Convert back to positive Sharpe

        logger.info("\n" + "=" * 80)
        logger.info("BAYESIAN OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Sharpe Ratio: {best_score:.3f}")
        logger.info(f"Best Configuration: {best_params}")

        # Save final results
        self._save_optimization_result(best_params, best_score)
        self._save_trial_history()

        return {
            'best_config': best_params,
            'best_sharpe': best_score,
            'n_trials': n_calls,
            'optimization_result': self.optimization_result,
        }

    def get_hyperparameter_importance(self) -> dict[str, float]:
        """
        Compute hyperparameter importance using fANOVA-like analysis.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.optimization_result is None:
            raise ValueError("Run optimize() first")

        # Simple importance: variance of objective across parameter values
        # More sophisticated: use fANOVA from scikit-optimize
        param_names = BayesianSearchSpace().get_param_names()
        importance = {}

        # Extract parameter values and scores from trials
        X = np.array([[trial['config'][p] for p in param_names if p in trial['config']]
                      for trial in self.trial_history])
        y = np.array([trial['sharpe_ratio'] for trial in self.trial_history])

        # Compute correlation-based importance
        for i, param in enumerate(param_names):
            if i < X.shape[1]:
                # Categorical parameters need special handling
                if param == 'reward_type':
                    continue
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                importance[param] = abs(correlation) if not np.isnan(correlation) else 0.0

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def plot_convergence(self, save_path: str | None = None):
        """Plot convergence of optimization."""
        if self.optimization_result is None:
            raise ValueError("Run optimize() first")

        try:
            import matplotlib.pyplot as plt
            plot_convergence(self.optimization_result)
            plt.ylabel("Negative Sharpe Ratio (minimize)")

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved convergence plot to {save_path}")
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib not available for plotting")

    def _save_trial_history(self):
        """Save trial history to JSON."""
        output_file = self.output_dir / "trial_history.json"
        with open(output_file, 'w') as f:
            json.dump(self.trial_history, f, indent=2, default=str)
        logger.info(f"Saved trial history to {output_file}")

    def _save_optimization_result(self, best_config: dict[str, Any], best_score: float):
        """Save optimization results."""
        result_dict = {
            'best_config': best_config,
            'best_sharpe_ratio': best_score,
            'n_trials': len(self.trial_history),
            'convergence': [-y for y in self.optimization_result.func_vals],  # Convert to positive
        }

        # Add hyperparameter importance
        try:
            importance = self.get_hyperparameter_importance()
            result_dict['hyperparameter_importance'] = importance
        except Exception as e:
            logger.warning(f"Could not compute hyperparameter importance: {e}")

        output_file = self.output_dir / "optimization_result.json"
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        logger.info(f"Saved optimization result to {output_file}")

        # Save skopt result object for later analysis
        dump(self.optimization_result, str(self.output_dir / "skopt_result.pkl"))


def main():
    """Example Bayesian optimization run."""
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
        use_ppo=True,
    )

    # Define search space
    search_space = BayesianSearchSpace(
        risk_per_trade=(0.01, 0.03),
        max_position=(5.0, 15.0),
        reward_type=["incremental_pnl", "cost_aware", "sharpe"],
        learning_rate_log=(-4.5, -3.5),  # 3e-5 to 3e-4
        gamma=(0.95, 0.99),
        entropy_coef_log=(-3, -1),  # 0.001 to 0.1
    )

    # Create Bayesian tuner
    tuner = BayesianHyperparameterTuner(base_exec_config, base_rl_config)

    # Run optimization
    tuner.optimize(
        search_space=search_space,
        n_calls=30,  # Total optimization iterations
        n_episodes=5,  # Episodes per config
        episode_length=100,  # Shorter for demo
        n_initial_points=10,  # Random initialization
        acq_func="EI",  # Expected Improvement
        verbose=True,
    )

    logger.info("\n" + "=" * 80)
    logger.info("HYPERPARAMETER IMPORTANCE:")
    logger.info("=" * 80)
    importance = tuner.get_hyperparameter_importance()
    for param, score in importance.items():
        logger.info(f"{param:30s}: {score:.3f}")

    # Plot convergence
    tuner.plot_convergence(save_path="bayesian_tuning_results/convergence.png")


if __name__ == "__main__":
    main()
