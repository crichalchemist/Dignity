"""Deterministic risk gate — no model dependency.

Sits between PolicyHead and the MetaApi executor. All decisions are pure
functions of scalar floats so they're testable without a running model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    """Immutable gate verdict returned by check_risk_gate."""

    allowed: bool
    reason: str          # "ok" | "drawdown_exceeded"
    adjusted_size: float  # 0.0 when blocked; min(requested, max) when allowed


def check_risk_gate(
    var_estimate: float,
    position_size: float,
    max_drawdown: float,
    max_position_size: float,
) -> GateDecision:
    """Stateless hard-stop gate.

    Blocks execution when the model's VaR estimate exceeds the configured
    maximum drawdown threshold. When allowed, caps position size at the
    instrument maximum (prevents oversizing regardless of model confidence).

    Args:
        var_estimate: RiskHead's var_estimate output — fraction of portfolio
                      at risk (sigmoid in [0, 1]).
        position_size: Requested lot size from PolicyHead.
        max_drawdown: Maximum tolerable drawdown fraction (e.g. 0.05 = 5%).
        max_position_size: Hard ceiling on lot size.

    Returns:
        GateDecision with allowed=False and adjusted_size=0.0 if var exceeds
        threshold; otherwise allowed=True with adjusted_size capped at max.
    """
    if var_estimate > max_drawdown:
        return GateDecision(allowed=False, reason="drawdown_exceeded", adjusted_size=0.0)

    capped = min(float(position_size), float(max_position_size))
    return GateDecision(allowed=True, reason="ok", adjusted_size=capped)
