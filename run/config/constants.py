"""Centralized constants for the Sequence trading system.

This module consolidates all constants from:
- Training configuration
- Feature engineering
- Execution and backtesting
"""

# -----------------------------------------------------------------------------
# Training Constants
# -----------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 10
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 4

# -----------------------------------------------------------------------------
# Feature Engineering Constants
# -----------------------------------------------------------------------------
MAX_THRESHOLD_VALUE = 1.0
MIN_THRESHOLD_VALUE = 0.0
DEFAULT_DC_THRESHOLD = 0.001

# -----------------------------------------------------------------------------
# Execution and Backtesting Constants
# -----------------------------------------------------------------------------
DEFAULT_BACKTEST_CASH = 10000
DEFAULT_COMMISSION_RATE = 0.001
MIN_COMMISSION_RATE = 0.0001
MAX_COMMISSION_RATE = 0.01
