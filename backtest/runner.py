"""BacktestRunner — orchestrates data + cascade signals → backtesting.py simulation.

Design: cascade inference runs in batch BEFORE Backtest.run(). Signals are
aligned to OHLCV bars (with warmup padding), injected as class-level state
on DignityStrategy, then backtesting.py reads them bar-by-bar via I().

This keeps the model completely outside the hot path of backtesting.py's
event loop — no per-bar inference, no coupling to the training loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

from .strategy import ACTION_HOLD, DignityStrategy

# ---------------------------------------------------------------------------
# Backtest gate thresholds — machine-enforced via validate_backtest_results()
# ---------------------------------------------------------------------------

BACKTEST_MIN_ARR = 0.15  # 15% annualized return (fraction, not %)
BACKTEST_MIN_SHARPE = 1.0  # Sharpe ratio
BACKTEST_MAX_DRAWDOWN = 0.20  # 20% max drawdown (fraction, not %)
BACKTEST_MIN_WIN_RATE = 0.52  # 52% win rate (fraction, not %)
BACKTEST_MAX_GATE_TRIGGER = 0.10  # risk gate fires on ≤10% of bars


class BacktestGateError(Exception):
    """Raised when backtest results do not meet minimum gate thresholds.

    All failures are reported together so the caller can see every gap at once,
    not just the first one hit.
    """


# Required OHLCV columns (backtesting.py convention — Title case)
_REQUIRED_COLS = ("Open", "High", "Low", "Close", "Volume")
_COL_MAP = {c.lower(): c for c in _REQUIRED_COLS}


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a backtesting.py simulation run.

    Defaults reflect realistic forex spot conditions:
    - 0.02% commission (typical ECN fee per side)
    - 0.01% spread (1 pip on EUR/USD)
    - No leverage (margin=1.0) for conservative initial evaluation
    """

    cash: float = 10_000.0
    commission: float = 0.0002
    spread: float = 0.0001
    margin: float = 1.0
    max_drawdown: float = 0.05
    position_size: float = 0.95
    trade_on_close: bool = True
    exclusive_orders: bool = True  # each new signal closes prior trade
    finalize_trades: bool = True  # close open trades at end so # Trades counts them


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename lowercase OHLCV columns to Title case required by backtesting.py.

    Accepts either case; raises ValueError if required columns are missing.
    """
    # Build rename map only for columns that need renaming
    rename = {src: dst for src, dst in _COL_MAP.items() if src in df.columns}
    result = df.rename(columns=rename)

    missing = [c for c in _REQUIRED_COLS if c not in result.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing}")

    return result


def align_signals(
    signals: dict[str, np.ndarray],
    n_bars: int,
    seq_len: int = 100,
) -> dict[str, np.ndarray]:
    """Pad cascade signal arrays to match OHLCV bar count.

    The cascade model outputs one prediction per sequence window. With stride=1
    and seq_len=100, the first valid prediction corresponds to bar 99 (0-indexed).
    Bars 0..seq_len-2 receive neutral warmup values so the signal array length
    equals n_bars exactly.

    Warmup padding:
        action  → ACTION_HOLD (0) — no trades during warmup
        var     → 0.0 — risk gate open (but action=HOLD prevents trading)
        alpha   → 0.0 — neutral
        regime  → 0.0 — calm

    Args:
        signals: Dict of pre-computed arrays, each of length n_bars - seq_len + 1.
        n_bars:  Total number of OHLCV bars.
        seq_len: Sequence window length used during inference.

    Raises:
        ValueError: If signal array length doesn't match expected n_bars - seq_len + 1.
    """
    warmup = seq_len - 1
    expected_len = n_bars - warmup

    neutral_values: dict[str, float] = {
        "action": float(ACTION_HOLD),
        "var": 0.0,
        "alpha": 0.0,
        "regime": 0.0,
    }

    aligned: dict[str, np.ndarray] = {}
    for key, arr in signals.items():
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr) != expected_len:
            raise ValueError(
                f"Signal '{key}' has length {len(arr)}, expected {expected_len} "
                f"(n_bars={n_bars} - seq_len={seq_len} + 1)."
            )
        neutral = neutral_values.get(key, 0.0)
        padding = np.full(warmup, neutral, dtype=np.float64)
        aligned[key] = np.concatenate([padding, arr])

    return aligned


def run_backtest(
    ohlcv: pd.DataFrame,
    signals: dict[str, np.ndarray],
    config: BacktestConfig = BacktestConfig(),
    plot: bool = False,
    plot_path: str | None = None,
) -> pd.Series:
    """Run a backtesting.py simulation driven by pre-computed cascade signals.

    Args:
        ohlcv:     OHLCV DataFrame (lowercase or Title case columns).
                   Length must equal len(signals["action"]).
        signals:   Pre-computed signal arrays aligned to OHLCV bars.
                   Required keys: "action", "var", "alpha", "regime".
        config:    Simulation parameters (cash, commission, etc.).
        plot:      If True, generate interactive Bokeh HTML plot.
        plot_path: File path for the HTML plot. Defaults to "backtest.html".

    Returns:
        pd.Series of performance statistics from backtesting.py run().
    """
    ohlcv_titled = prepare_ohlcv(ohlcv)

    # Inject pre-computed signals as class-level state on DignityStrategy.
    # backtesting.py calls Strategy.init() once, which wraps these via I().
    DignityStrategy._signals = {k: np.asarray(v, dtype=np.float64) for k, v in signals.items()}

    # Stamp optimizable parameters onto the class so bt.optimize() can vary them
    DignityStrategy.max_drawdown = config.max_drawdown
    DignityStrategy.position_size = config.position_size

    bt = Backtest(
        ohlcv_titled,
        DignityStrategy,
        cash=config.cash,
        commission=config.commission,
        spread=config.spread,
        margin=config.margin,
        trade_on_close=config.trade_on_close,
        exclusive_orders=config.exclusive_orders,
        finalize_trades=config.finalize_trades,
    )

    stats = bt.run()

    if plot:
        path = plot_path or "backtest.html"
        bt.plot(filename=path, open_browser=False)

    return stats


def compute_gate_metrics(
    stats: pd.Series,
    signals: dict[str, np.ndarray],
    config: BacktestConfig,
) -> dict[str, float]:
    """Extract gate-relevant metrics from backtesting.py stats and aligned signals.

    Bridges backtesting.py's key naming and percentage conventions to the
    plain fractions that validate_backtest_results() expects.

    Returns:
        dict with keys: arr, sharpe, max_drawdown, win_rate, gate_trigger_rate
    """
    ann_return = stats.get("Return (Ann.) [%]", 0.0)
    sharpe = stats.get("Sharpe Ratio", 0.0)
    max_dd = abs(stats.get("Max. Drawdown [%]", 0.0))
    win_rate = stats.get("Win Rate [%]", 0.0)

    # backtesting.py returns NaN for Sharpe / Win Rate when there are no trades
    return {
        "arr": float(ann_return) / 100.0 if not pd.isna(ann_return) else 0.0,
        "sharpe": float(sharpe) if not pd.isna(sharpe) else 0.0,
        "max_drawdown": float(max_dd) / 100.0,
        "win_rate": float(win_rate) / 100.0 if not pd.isna(win_rate) else 0.0,
        "gate_trigger_rate": _gate_trigger_rate(signals, config.max_drawdown),
    }


def _gate_trigger_rate(signals: dict[str, np.ndarray], max_drawdown: float) -> float:
    """Fraction of bars where VaR exceeded max_drawdown (risk gate fired)."""
    var = np.asarray(signals.get("var", []), dtype=np.float64)
    return float((var > max_drawdown).mean()) if len(var) > 0 else 0.0


def validate_backtest_results(metrics: dict[str, float]) -> None:
    """Raise BacktestGateError if any gate threshold is not met.

    All failures are collected before raising so the caller sees every gap.

    Args:
        metrics: dict from compute_gate_metrics() with float values for:
            arr, sharpe, max_drawdown, win_rate, gate_trigger_rate
    """
    failures: list[str] = []

    if metrics.get("arr", 0.0) < BACKTEST_MIN_ARR:
        failures.append(f"ARR {metrics['arr']:.1%} < required {BACKTEST_MIN_ARR:.0%}")
    if metrics.get("sharpe", 0.0) < BACKTEST_MIN_SHARPE:
        failures.append(f"Sharpe {metrics['sharpe']:.2f} < required {BACKTEST_MIN_SHARPE:.1f}")
    if metrics.get("max_drawdown", 1.0) > BACKTEST_MAX_DRAWDOWN:
        failures.append(
            f"Max drawdown {metrics['max_drawdown']:.1%} > allowed {BACKTEST_MAX_DRAWDOWN:.0%}"
        )
    if metrics.get("win_rate", 0.0) < BACKTEST_MIN_WIN_RATE:
        failures.append(
            f"Win rate {metrics['win_rate']:.1%} < required {BACKTEST_MIN_WIN_RATE:.0%}"
        )
    if metrics.get("gate_trigger_rate", 1.0) > BACKTEST_MAX_GATE_TRIGGER:
        failures.append(
            f"Gate trigger rate {metrics['gate_trigger_rate']:.1%} > allowed {BACKTEST_MAX_GATE_TRIGGER:.0%}"
        )

    if failures:
        raise BacktestGateError(
            "Backtest gate failed — all must pass before paper trading:\n"
            + "\n".join(f"  • {f}" for f in failures)
        )


def write_backtest_report(
    metrics: dict[str, float],
    config_path: str | Path,
    checkpoint_path: str | Path,
    data_date_range: tuple[str, str],
    output_dir: str | Path = "reports",
) -> Path:
    """Write a JSON audit artifact after a backtest gate pass.

    Creates an immutable record of what was tested and how it performed,
    required before paper trading begins (Section 5 prerequisite).

    Args:
        metrics:         dict from compute_gate_metrics()
        config_path:     YAML config used for training
        checkpoint_path: Model checkpoint evaluated
        data_date_range: (start_date, end_date) ISO date strings for test data
        output_dir:      Output directory (created if absent)

    Returns:
        Path to the written JSON file
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = date.today().strftime("%Y%m%d")
    report_path = out_dir / f"backtest_report_{stamp}.json"

    report = {
        "metrics": metrics,
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "data_date_range": list(data_date_range),
        "generated": date.today().isoformat(),
    }

    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
