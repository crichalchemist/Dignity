"""
Risk management utilities to gate trading actions before order entry.

The risk manager operates on classification logits or regression outputs and
returns adjusted outputs when gates are tripped (e.g., by excessive drawdown,
position limits, high volatility, spreads, or no-trade windows). Logging is
included to make it easy to diagnose why actions were blocked.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised when torch isn't installed
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class RiskConfig:
    """Configuration for pre-trade risk gates.

    Attributes:
        enabled: Toggle all risk checks on/off.
        max_drawdown_pct: Maximum acceptable drawdown before forcing flat.
        max_positions: Maximum simultaneous non-flat positions.
        volatility_threshold: Volatility level (std-dev proxy) to throttle trades.
        no_trade_hours: List of (start_hour, end_hour) tuples (24h) where entries are blocked.
        max_spread: Spread threshold beyond which new entries are blocked.
        flat_class_index: Index of the "flat"/"hold" class in classification logits.
        throttle_factor: Scaling applied to regression outputs when throttling.
    """

    enabled: bool = True
    max_drawdown_pct: float = 0.2
    max_positions: int = 3
    volatility_threshold: float = 0.02
    no_trade_hours: list[tuple[int, int]] = field(default_factory=list)
    max_spread: float = 0.0002
    flat_class_index: int = 1
    throttle_factor: float = 0.5


class RiskManager:
    """Stateful risk manager used by training and inference loops."""

    def __init__(self, cfg: RiskConfig | None = None):
        self.cfg = cfg or RiskConfig()
        self.current_equity: float | None = None
        self.peak_equity: float | None = None
        self.open_positions: int = 0

    def update_equity(self, equity: float) -> None:
        """Update equity marks used for drawdown checks."""

        if equity < 0:
            logger.warning("Ignoring negative equity input: %s", equity)
            return
        self.current_equity = equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

    # --- Gate helpers -------------------------------------------------
    def _in_no_trade_window(self, timestamp: datetime | None) -> bool:
        if not timestamp or not self.cfg.no_trade_hours:
            return False
        hour = timestamp.hour
        return any(start <= hour <= end for start, end in self.cfg.no_trade_hours)

    def _drawdown_exceeded(self) -> bool:
        if self.current_equity is None or self.peak_equity is None:
            return False
        drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1e-9)
        return drawdown >= self.cfg.max_drawdown_pct

    def _active_gates(self, context: dict[str, Any] | None) -> list[str]:
        if not self.cfg.enabled:
            return []
        context = context or {}
        reasons: list[str] = []
        if self._drawdown_exceeded():
            reasons.append("max_drawdown")
        if self.cfg.max_positions and self.open_positions >= self.cfg.max_positions:
            reasons.append("position_limit")
        volatility = context.get("volatility")
        if volatility is not None and volatility > self.cfg.volatility_threshold:
            reasons.append("volatility_throttle")
        spread = context.get("spread")
        if spread is not None and spread > self.cfg.max_spread:
            reasons.append("spread_too_wide")
        timestamp = context.get("timestamp")
        if isinstance(timestamp, datetime) and self._in_no_trade_window(timestamp):
            reasons.append("no_trade_window")
        return reasons

    # --- Public API ----------------------------------------------------
    def apply_classification_logits(
            self, logits: torch.Tensor, context: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        """Mask or override logits based on risk gates.

        When a gate is active, all logits are suppressed except for the flat
        class, which is promoted to steer the policy away from new entries.
        """
        if not TORCH_AVAILABLE:
            raise ModuleNotFoundError("torch is required for apply_classification_logits")

        reasons = self._active_gates(context)
        if not reasons:
            return logits, []

        adjusted = logits.clone()
        adjusted = adjusted - adjusted.max(dim=1, keepdim=True).values - 1e3
        flat_idx = min(self.cfg.flat_class_index, adjusted.shape[1] - 1)
        adjusted[:, flat_idx] = 0.0
        return adjusted, reasons

    def apply_regression_output(
            self, preds: torch.Tensor, context: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        """Throttle regression outputs when gates are active."""
        if not TORCH_AVAILABLE:
            raise ModuleNotFoundError("torch is required for apply_regression_output")

        reasons = self._active_gates(context)
        if not reasons:
            return preds, []
        return preds * self.cfg.throttle_factor, reasons

    def record_actions(self, actions: torch.Tensor) -> None:
        """Update position count based on non-flat decisions."""
        if not TORCH_AVAILABLE:
            raise ModuleNotFoundError("torch is required for record_actions")

        if actions.ndim == 0:
            non_flat = int(actions.item() != self.cfg.flat_class_index)
        else:
            non_flat = int(torch.count_nonzero(actions != self.cfg.flat_class_index).item())
        self.open_positions = min(non_flat, self.cfg.max_positions)

    def log_events(self, reasons: Sequence[str], prefix: str = "") -> None:
        if not reasons:
            return
        tag = f"{prefix} " if prefix else ""
        logger.info("%sRisk gates triggered: %s", tag, ", ".join(reasons))

    # --- Convenience ---------------------------------------------------
    def build_context(
            self,
            x: torch.Tensor | None = None,
            timestamp: datetime | None = None,
            spread: float | None = None,
    ) -> dict[str, Any]:
        """Derive a minimal context dict from optional inputs."""
        if x is not None and not TORCH_AVAILABLE:
            raise ModuleNotFoundError("torch is required for build_context when x is provided")

        context: dict[str, Any] = {}
        if timestamp:
            context["timestamp"] = timestamp
        if spread is not None:
            context["spread"] = spread
        if x is not None:
            with torch.no_grad():
                vol = torch.std(x, dim=1, unbiased=False).mean().item()
            context["volatility"] = vol
        return context


__all__ = ["RiskConfig", "RiskManager"]
