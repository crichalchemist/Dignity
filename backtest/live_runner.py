"""Live trading run loop (Section 6).

Activated only after go_live.sh checklist passes all gates.
Structurally mirrors paper_runner.py but uses paper=False on the executor,
applies the MAX_POSITION_FRACTION hard cap, and uses a 7-day rolling
circuit breaker instead of the soak's daily breaker.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest.paper_runner import (
    _send_macos_alert,
    _setup_rotating_logger,
    append_bar_log,
    bars_to_tensor,
    write_alert,
    write_lock,
)
from core.execution import MAX_POSITION_FRACTION, apply_live_position_cap, check_risk_gate

# ---------------------------------------------------------------------------
# Live circuit breaker threshold
# ---------------------------------------------------------------------------

LIVE_CIRCUIT_BREAKER_DRAWDOWN = 0.08   # 7-day realized drawdown ceiling
LIVE_ALERT_DRAWDOWN = 0.04             # alert at 4% before circuit breaker fires

_ACTION_STRINGS: dict[int, str] = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ---------------------------------------------------------------------------
# Live config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveConfig:
    """Configuration for a live trading run."""

    model_path: str
    metaapi_token: str
    account_id: str
    symbol: str = "EURUSD"
    max_drawdown: float = 0.05
    max_position_size: float = 0.1
    seq_len: int = 100
    input_size: int = 32
    log_dir: str = "reports"


# ---------------------------------------------------------------------------
# Pure helpers — testable without MetaApi or model
# ---------------------------------------------------------------------------

def compute_rolling_drawdown(log_entries: list[dict], days: int = 7) -> float:
    """Sum realized_pnl over the most recent `days` calendar days.

    Returns a negative value when the rolling window is in drawdown.
    Returns 0.0 when there are no entries or fewer than `days` days of data.
    """
    if not log_entries:
        return 0.0
    daily: dict[str, float] = {}
    for e in log_entries:
        day = str(e.get("timestamp", ""))[:10]
        daily[day] = daily.get(day, 0.0) + float(e.get("realized_pnl", 0.0))
    sorted_days = sorted(daily.keys())
    recent = sorted_days[-days:]
    return sum(daily[d] for d in recent)


# ---------------------------------------------------------------------------
# Async run loop
# ---------------------------------------------------------------------------

async def run_live_loop(config: LiveConfig) -> None:
    """Live trading run loop.

    Requires go_live.sh checklist to have passed before invocation.
    Applies MAX_POSITION_FRACTION hard cap on every order.
    Circuit breaker fires when 7-day realized drawdown exceeds
    LIVE_CIRCUIT_BREAKER_DRAWDOWN — writes circuit_breaker.lock and exits.

    Not unit-tested (requires live MetaApi and real credentials).
    """
    from data.source.metaapi import MetaApiExecutor, MetaApiSource
    from models.dignity import Dignity

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    live_log = log_dir / "live_trading_log.jsonl"
    alerts_log = log_dir / "alerts.log"
    lock_path = log_dir / "circuit_breaker.lock"

    if lock_path.exists():
        raise RuntimeError(
            f"Circuit breaker lock exists at {lock_path}. "
            "Delete it manually to resume live trading."
        )

    _setup_rotating_logger(live_log)

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
        paper=False,
    )

    await src.connect()
    await executor.connect()

    _shutdown = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, _shutdown.set)

    bar_buffer: list[pd.Series] = []
    recent_entries: list[dict] = []  # rolling window for circuit breaker

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
            realized_pnl = 0.0

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
                    # Fetch account balance for hard cap — live only
                    account_info = await src._connection.get_account_information()
                    balance = float(account_info.get("balance", 10_000.0))
                    capped_size = apply_live_position_cap(gate.adjusted_size, balance)

                    order = await executor.execute(action_idx, capped_size, var_est)
                    realized_pnl = float(order.get("profit", 0.0)) if order else 0.0
                    action_str = _ACTION_STRINGS.get(action_idx, "HOLD")
                elif not gate.allowed:
                    action_str = "BLOCKED"

            except (RuntimeError, NameError) as exc:
                action_str = "ERROR"
                gate_passed = False
                write_alert(alerts_log, f"live executor error: {exc}")

            ts = bar.name.isoformat() if hasattr(bar.name, "isoformat") else str(bar.name)

            entry = {
                "timestamp": ts,
                "action": action_str,
                "regime": regime,
                "var_estimate": var_est,
                "alpha_score": alpha,
                "gate_passed": gate_passed,
                "realized_pnl": realized_pnl,
            }
            append_bar_log(live_log, entry)
            recent_entries.append(entry)

            # Rolling 7-day circuit breaker
            rolling_dd = compute_rolling_drawdown(recent_entries, days=7)
            if rolling_dd < -LIVE_ALERT_DRAWDOWN:
                msg = f"7-day drawdown {rolling_dd:.2%}"
                write_alert(alerts_log, msg)
                _send_macos_alert(msg)

            if rolling_dd < -LIVE_CIRCUIT_BREAKER_DRAWDOWN:
                write_lock(lock_path, f"7-day drawdown {rolling_dd:.2%}")
                break

    finally:
        await src.disconnect()
