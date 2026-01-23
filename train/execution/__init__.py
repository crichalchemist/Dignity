"""Trading execution environments.

Provides simulation environments for RL training and backtesting.

Modules:
    backtesting_env: Deterministic historical replay environment
    simulated_retail_env: Stochastic retail execution simulation
"""

from train.execution.simulated_retail_env import SimulatedRetailExecutionEnv

# Optional import for backtesting environment (requires backtesting library)
try:
    from train.execution.backtesting_env import BacktestingRetailExecutionEnv

    __all__ = [
        "BacktestingRetailExecutionEnv",
        "SimulatedRetailExecutionEnv",
    ]
except ImportError:
    # backtesting library not installed, only export simulated env
    __all__ = [
        "SimulatedRetailExecutionEnv",
    ]
