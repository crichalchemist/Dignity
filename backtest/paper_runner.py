"""Paper trading soak infrastructure (Section 5).

Pure helper functions are fully unit-testable without MetaApi credentials.
The async run_paper_loop() requires a live connection — all safety logic is
extracted into testable helpers so the loop itself is a thin coordinator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Soak gate thresholds
# ---------------------------------------------------------------------------

SOAK_MAX_DAILY_DRAWDOWN = 0.05         # circuit breaker fires at or below this
SOAK_ALERT_DAILY_DRAWDOWN = 0.03      # alert fires at or below this (no circuit break)
SOAK_MAX_REGIME_CONCENTRATION = 0.70   # no single regime > 70% of bars
SOAK_GATE_RATE_TOLERANCE = 0.05       # gate trigger rate within ±5% of backtest rate
SOAK_MIN_CALENDAR_DAYS = 30

_ACTION_STRINGS: dict[int, str] = {0: "HOLD", 1: "BUY", 2: "SELL"}
_LOG_ROTATE_BYTES = 10 * 1024 * 1024
_LOG_ROTATE_BACKUPS = 5


# ---------------------------------------------------------------------------
# Soak config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SoakConfig:
    """Configuration for a paper trading soak run."""

    model_path: str
    metaapi_token: str
    account_id: str
    symbol: str = "EURUSD"
    max_drawdown: float = 0.05
    max_position_size: float = 0.1
    backtest_gate_rate: float = 0.0   # from Section 4 report
    seq_len: int = 100
    input_size: int = 32
    log_dir: str = "reports"


# ---------------------------------------------------------------------------
# Pure helpers — testable without MetaApi or model
# ---------------------------------------------------------------------------

def append_bar_log(log_path: Path, entry: dict) -> None:
    """Append one JSON line to the soak JSONL log."""
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def write_alert(alerts_path: Path, message: str) -> None:
    """Append a timestamped entry to alerts.log."""
    ts = datetime.utcnow().isoformat(timespec="seconds")
    with open(alerts_path, "a") as f:
        f.write(f"[{ts}] {message}\n")


def write_lock(lock_path: Path, reason: str) -> None:
    """Write soak_breaker.lock. Manual deletion required before soak can resume."""
    ts = datetime.utcnow().isoformat(timespec="seconds")
    lock_path.write_text(f"{ts} — {reason}\n")


def check_daily_drawdown(log_entries: list[dict]) -> float:
    """Return the worst (most negative) single-day P&L sum as a fraction.

    Groups simulated_pnl by calendar day (YYYY-MM-DD prefix of timestamp).
    Returns 0.0 when there are no entries.
    """
    if not log_entries:
        return 0.0
    daily: dict[str, float] = {}
    for e in log_entries:
        day = str(e.get("timestamp", ""))[:10]
        daily[day] = daily.get(day, 0.0) + float(e.get("simulated_pnl", 0.0))
    return min(daily.values()) if daily else 0.0


def evaluate_soak_gate(
    log_entries: list[dict],
    backtest_gate_rate: float,
    min_days: int = SOAK_MIN_CALENDAR_DAYS,
) -> dict[str, bool]:
    """Evaluate all soak gate criteria against accumulated log entries.

    Args:
        log_entries:        Parsed JSONL lines from paper_trading_log.jsonl.
        backtest_gate_rate: Gate trigger rate from Section 4 report (fraction).
        min_days:           Minimum calendar days required (default 30).

    Returns:
        Dict mapping criterion name → bool, plus an "all_passed" key.
        Criteria: days_elapsed, daily_drawdown, regime_concentration,
                  gate_trigger_rate, zero_executor_errors.
    """
    empty = {
        "days_elapsed": False,
        "daily_drawdown": False,
        "regime_concentration": False,
        "gate_trigger_rate": False,
        "zero_executor_errors": False,
        "all_passed": False,
    }
    if not log_entries:
        return empty

    n = len(log_entries)

    # Calendar days
    dates = {str(e.get("timestamp", ""))[:10] for e in log_entries}
    days_ok = len(dates) >= min_days

    # Worst daily drawdown
    worst = check_daily_drawdown(log_entries)
    drawdown_ok = worst >= -SOAK_MAX_DAILY_DRAWDOWN

    # Regime concentration — no single regime dominates
    regimes = [int(e.get("regime", 0)) for e in log_entries]
    max_regime_frac = max(regimes.count(r) for r in set(regimes)) / n if n > 0 else 1.0
    regime_ok = max_regime_frac <= SOAK_MAX_REGIME_CONCENTRATION

    # Gate trigger rate vs backtest baseline
    blocked = sum(1 for e in log_entries if not e.get("gate_passed", True))
    gate_rate = blocked / n if n > 0 else 0.0
    gate_rate_ok = abs(gate_rate - backtest_gate_rate) <= SOAK_GATE_RATE_TOLERANCE

    # Executor errors logged as action="ERROR"
    errors_ok = sum(1 for e in log_entries if e.get("action") == "ERROR") == 0

    results: dict[str, bool] = {
        "days_elapsed": days_ok,
        "daily_drawdown": drawdown_ok,
        "regime_concentration": regime_ok,
        "gate_trigger_rate": gate_rate_ok,
        "zero_executor_errors": errors_ok,
    }
    results["all_passed"] = all(results.values())
    return results


def bars_to_tensor(bar_buffer: list[pd.Series], input_size: int) -> torch.Tensor:
    """Convert a rolling bar buffer to a [1, seq_len, input_size] model input.

    Extracts OHLCV arrays, computes the 32-feature signal set via
    SignalProcessor.process_sequence, and stacks into a float32 tensor.

    Args:
        bar_buffer: List of pd.Series, each with open/high/low/close/volume keys.
        input_size: Must equal the number of features returned by process_sequence.

    Returns:
        torch.Tensor of shape [1, len(bar_buffer), input_size].
    """
    from core.signals import SignalProcessor

    closes = np.array([b["close"] for b in bar_buffer], dtype=np.float64)
    highs = np.array([b["high"] for b in bar_buffer], dtype=np.float64)
    lows = np.array([b["low"] for b in bar_buffer], dtype=np.float64)
    volumes = np.array([b["volume"] for b in bar_buffer], dtype=np.float64)

    signals = SignalProcessor.process_sequence(
        volumes=volumes,
        prices=closes,
        high=highs,
        low=lows,
    )
    matrix = np.stack(list(signals.values()), axis=1)  # [seq_len, features]
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, features]


def _send_macos_alert(message: str) -> None:
    """Fire a macOS notification via osascript (best-effort, non-blocking)."""
    os.system(
        f"osascript -e 'display notification \"{message}\" with title \"Dignity Soak\"'"
    )


def _setup_rotating_logger(log_path: Path) -> logging.Logger:
    name = f"paper_soak.{log_path.stem}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path, maxBytes=_LOG_ROTATE_BYTES, backupCount=_LOG_ROTATE_BACKUPS
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Async run loop
# ---------------------------------------------------------------------------

async def run_paper_loop(config: SoakConfig) -> None:
    """Long-running paper trading soak loop.

    Runs until SIGTERM or the daily drawdown circuit breaker fires.
    Each bar: stream → signals → model → gate → executor → JSONL log.

    Circuit breaker: daily drawdown > SOAK_MAX_DAILY_DRAWDOWN writes
    reports/soak_breaker.lock and exits. Manual deletion required to resume.
    """
    from core.execution import check_risk_gate
    from data.source.metaapi import MetaApiExecutor, MetaApiSource
    from models.dignity import Dignity

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    soak_log = log_dir / "paper_trading_log.jsonl"
    alerts_log = log_dir / "alerts.log"
    lock_path = log_dir / "soak_breaker.lock"

    if lock_path.exists():
        raise RuntimeError(
            f"Soak breaker lock exists at {lock_path}. "
            "Delete it manually before resuming."
        )

    _setup_rotating_logger(soak_log)

    model = Dignity(task="cascade", input_size=config.input_size)
    ckpt = torch.load(config.model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    src = MetaApiSource(
        token=config.metaapi_token,
        account_id=config.account_id,
        symbol=config.symbol,
    )
    executor = MetaApiExecutor(
        token=config.metaapi_token,
        account_id=config.account_id,
        symbol=config.symbol,
        max_drawdown=config.max_drawdown,
        max_position_size=config.max_position_size,
        paper=True,
    )

    await src.connect()

    _shutdown = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, _shutdown.set)

    bar_buffer: list[pd.Series] = []
    daily_pnl: dict[str, float] = {}

    try:
        async for bar in src.stream():
            if _shutdown.is_set():
                break

            bar_buffer.append(bar)
            if len(bar_buffer) < config.seq_len:
                continue
            bar_buffer = bar_buffer[-config.seq_len:]

            action_str = "HOLD"
            var_est = 0.0
            alpha = 0.0
            regime = 0
            gate_passed = True
            simulated_pnl = 0.0

            try:
                x = bars_to_tensor(bar_buffer, config.input_size)
                with torch.no_grad():
                    outputs = model(x)

                var_est = float(outputs["var_estimate"].item())
                alpha = float(outputs["alpha_score"].item())
                regime = int(outputs["regime_probs"].argmax().item())
                action_idx = int(outputs["action_logits"].argmax().item())
                position_size = float(outputs["position_limit"].item())

                gate = check_risk_gate(
                    var_est, position_size,
                    config.max_drawdown, config.max_position_size,
                )
                gate_passed = gate.allowed

                if gate.allowed and action_idx != 0:
                    order = await executor.execute(action_idx, position_size, var_est)
                    simulated_pnl = float(order.get("simulated_pnl", 0.0)) if order else 0.0
                    action_str = _ACTION_STRINGS.get(action_idx, "HOLD")
                elif not gate.allowed:
                    action_str = "BLOCKED"

            except (RuntimeError, NameError) as exc:
                action_str = "ERROR"
                gate_passed = False
                write_alert(alerts_log, f"executor error: {exc}")

            ts = bar.name.isoformat() if hasattr(bar.name, "isoformat") else str(bar.name)
            day = ts[:10]
            daily_pnl[day] = daily_pnl.get(day, 0.0) + simulated_pnl

            append_bar_log(soak_log, {
                "timestamp": ts,
                "action": action_str,
                "regime": regime,
                "var_estimate": var_est,
                "alpha_score": alpha,
                "gate_passed": gate_passed,
                "simulated_pnl": simulated_pnl,
            })

            day_dd = daily_pnl[day]
            if day_dd < -SOAK_ALERT_DAILY_DRAWDOWN:
                msg = f"daily drawdown {day_dd:.2%} on {day}"
                write_alert(alerts_log, msg)
                _send_macos_alert(msg)

            if day_dd < -SOAK_MAX_DAILY_DRAWDOWN:
                write_lock(lock_path, f"daily drawdown {day_dd:.2%} on {day}")
                break

    finally:
        await src.disconnect()
