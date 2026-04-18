"""Dignity backtesting module — pre-inference cascade signals → backtesting.py simulation."""

from .runner import BacktestConfig, align_signals, prepare_ohlcv, run_backtest
from .strategy import ACTION_BUY, ACTION_HOLD, ACTION_SELL, DignityStrategy

__all__ = [
    "DignityStrategy",
    "BacktestConfig",
    "run_backtest",
    "prepare_ohlcv",
    "align_signals",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
]
