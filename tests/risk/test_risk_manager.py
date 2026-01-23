"""Tests for risk/risk_manager.py - risk management and position limits."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.mark.unit
class TestRiskConfig:
    """Unit tests for RiskConfig initialization."""

    def test_risk_config_defaults(self):
        """Test default RiskConfig values."""
        from risk.risk_manager import RiskConfig

        cfg = RiskConfig()

        assert cfg.enabled == True
        assert cfg.max_drawdown_pct == 0.2
        assert cfg.max_positions == 3
        assert cfg.volatility_threshold == 0.02
        assert cfg.max_spread == 0.0002
        assert cfg.flat_class_index == 1
        assert cfg.throttle_factor == 0.5

    def test_risk_config_custom_values(self):
        """Test RiskConfig with custom values."""
        from risk.risk_manager import RiskConfig

        cfg = RiskConfig(
            enabled=False,
            max_drawdown_pct=0.1,
            max_positions=5,
            volatility_threshold=0.03,
            max_spread=0.0005,
        )

        assert cfg.enabled == False
        assert cfg.max_drawdown_pct == 0.1
        assert cfg.max_positions == 5
        assert cfg.volatility_threshold == 0.03
        assert cfg.max_spread == 0.0005

    def test_risk_config_no_trade_hours(self):
        """Test RiskConfig with no-trade hours."""
        from risk.risk_manager import RiskConfig

        no_trade_hours = [(0, 5), (21, 23)]  # Avoid early morning and late night
        cfg = RiskConfig(no_trade_hours=no_trade_hours)

        assert cfg.no_trade_hours == no_trade_hours


@pytest.mark.unit
class TestRiskManagerInitialization:
    """Unit tests for RiskManager initialization."""

    def test_risk_manager_with_default_config(self):
        """Test RiskManager with default config."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()

        assert rm.cfg is not None
        assert rm.current_equity is None
        assert rm.peak_equity is None
        assert rm.open_positions == 0

    def test_risk_manager_with_custom_config(self):
        """Test RiskManager with custom config."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_drawdown_pct=0.15, max_positions=2)
        rm = RiskManager(cfg)

        assert rm.cfg.max_drawdown_pct == 0.15
        assert rm.cfg.max_positions == 2

    def test_risk_manager_disabled(self):
        """Test RiskManager with disabled config."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(enabled=False)
        rm = RiskManager(cfg)

        assert rm.cfg.enabled == False


@pytest.mark.unit
class TestEquityTracking:
    """Unit tests for equity tracking and peak equity."""

    def test_update_equity_single_value(self):
        """Test updating equity once."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()
        rm.update_equity(100.0)

        assert rm.current_equity == 100.0
        assert rm.peak_equity == 100.0

    def test_update_equity_increasing(self):
        """Test peak equity tracking with increasing values."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()

        for equity in [100.0, 110.0, 120.0, 115.0]:
            rm.update_equity(equity)

        assert rm.current_equity == 115.0
        assert rm.peak_equity == 120.0

    def test_update_equity_decreasing(self):
        """Test equity tracking with decreasing values."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()

        rm.update_equity(100.0)
        rm.update_equity(90.0)
        rm.update_equity(80.0)

        assert rm.current_equity == 80.0
        assert rm.peak_equity == 100.0

    def test_update_equity_negative_ignored(self):
        """Test that negative equity is ignored."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()
        rm.update_equity(100.0)
        rm.update_equity(-50.0)  # Should be ignored

        assert rm.current_equity == 100.0
        assert rm.peak_equity == 100.0


@pytest.mark.unit
class TestDrawdownDetection:
    """Unit tests for drawdown detection."""

    def test_drawdown_not_exceeded_initial(self):
        """Test no drawdown on single equity update."""
        from risk.risk_manager import RiskManager

        rm = RiskManager()
        rm.update_equity(100.0)

        assert rm._drawdown_exceeded() == False

    def test_drawdown_exceeded(self):
        """Test drawdown detection when limit is breached."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_drawdown_pct=0.2)  # 20% max drawdown
        rm = RiskManager(cfg)

        rm.update_equity(100.0)  # Peak
        rm.update_equity(79.0)   # 21% drawdown - exceeds limit

        assert rm._drawdown_exceeded() == True

    def test_drawdown_at_limit(self):
        """Test drawdown at exact limit."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_drawdown_pct=0.2)
        rm = RiskManager(cfg)

        rm.update_equity(100.0)
        rm.update_equity(80.0)   # Exactly 20% drawdown

        assert rm._drawdown_exceeded() == True

    def test_drawdown_below_limit(self):
        """Test drawdown below limit."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_drawdown_pct=0.2)
        rm = RiskManager(cfg)

        rm.update_equity(100.0)
        rm.update_equity(81.0)   # 19% drawdown - below limit

        assert rm._drawdown_exceeded() == False

    def test_drawdown_with_multiple_peaks(self):
        """Test drawdown calculation with multiple peaks."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_drawdown_pct=0.2)
        rm = RiskManager(cfg)

        rm.update_equity(100.0)
        rm.update_equity(110.0)  # New peak
        rm.update_equity(89.0)   # 19.1% from new peak

        assert rm._drawdown_exceeded() == False


@pytest.mark.unit
class TestPositionLimits:
    """Unit tests for position limit checking."""

    def test_position_limit_not_exceeded(self):
        """Test position limit check when below limit."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_positions=3)
        rm = RiskManager(cfg)

        rm.open_positions = 2

        active_gates = rm._active_gates({})

        assert "position_limit" not in active_gates

    def test_position_limit_exceeded(self):
        """Test position limit check when at limit."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_positions=3)
        rm = RiskManager(cfg)

        rm.open_positions = 3

        active_gates = rm._active_gates({})

        assert "position_limit" in active_gates

    def test_position_limit_zero_positions(self):
        """Test position limit with zero open positions."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_positions=3)
        rm = RiskManager(cfg)

        rm.open_positions = 0

        active_gates = rm._active_gates({})

        assert "position_limit" not in active_gates


@pytest.mark.unit
class TestNoTradeWindow:
    """Unit tests for no-trade window detection."""

    def test_in_no_trade_window_yes(self):
        """Test detection of no-trade window."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(no_trade_hours=[(22, 23), (0, 5)])
        rm = RiskManager(cfg)

        # Hour 3 is in no-trade window
        timestamp = datetime(2024, 1, 1, 3, 0, 0)

        assert rm._in_no_trade_window(timestamp) == True

    def test_in_no_trade_window_no(self):
        """Test outside of no-trade window."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(no_trade_hours=[(22, 23), (0, 5)])
        rm = RiskManager(cfg)

        # Hour 12 is outside no-trade window
        timestamp = datetime(2024, 1, 1, 12, 0, 0)

        assert rm._in_no_trade_window(timestamp) == False

    def test_in_no_trade_window_boundary(self):
        """Test boundary conditions of no-trade window."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(no_trade_hours=[(0, 5)])
        rm = RiskManager(cfg)

        # Test hour 0 (start)
        assert rm._in_no_trade_window(datetime(2024, 1, 1, 0, 0, 0)) == True

        # Test hour 5 (end)
        assert rm._in_no_trade_window(datetime(2024, 1, 1, 5, 0, 0)) == True

        # Test hour 6 (just outside)
        assert rm._in_no_trade_window(datetime(2024, 1, 1, 6, 0, 0)) == False

    def test_in_no_trade_window_no_timestamp(self):
        """Test with no timestamp."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(no_trade_hours=[(0, 5)])
        rm = RiskManager(cfg)

        assert rm._in_no_trade_window(None) == False


@pytest.mark.unit
class TestVolatilityThrottle:
    """Unit tests for volatility-based throttling."""

    def test_volatility_throttle_active(self):
        """Test volatility throttle activation."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(volatility_threshold=0.02)
        rm = RiskManager(cfg)

        context = {"volatility": 0.03}  # Above threshold
        active_gates = rm._active_gates(context)

        assert "volatility_throttle" in active_gates

    def test_volatility_throttle_inactive(self):
        """Test volatility below threshold."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(volatility_threshold=0.02)
        rm = RiskManager(cfg)

        context = {"volatility": 0.01}  # Below threshold
        active_gates = rm._active_gates(context)

        assert "volatility_throttle" not in active_gates

    def test_volatility_throttle_at_threshold(self):
        """Test volatility at exact threshold."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(volatility_threshold=0.02)
        rm = RiskManager(cfg)

        context = {"volatility": 0.02}  # At threshold
        active_gates = rm._active_gates(context)

        # Should not trigger at exact threshold (> not >=)
        assert "volatility_throttle" not in active_gates

    def test_volatility_throttle_none_context(self):
        """Test with no volatility in context."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(volatility_threshold=0.02)
        rm = RiskManager(cfg)

        context = {}
        active_gates = rm._active_gates(context)

        assert "volatility_throttle" not in active_gates


@pytest.mark.unit
class TestSpreadCheck:
    """Unit tests for spread-based position limiting."""

    def test_spread_too_wide(self):
        """Test spread limit detection."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_spread=0.0002)
        rm = RiskManager(cfg)

        context = {"spread": 0.0003}  # Above limit
        active_gates = rm._active_gates(context)

        assert "spread_too_wide" in active_gates

    def test_spread_within_limit(self):
        """Test spread within acceptable range."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_spread=0.0002)
        rm = RiskManager(cfg)

        context = {"spread": 0.0001}  # Below limit
        active_gates = rm._active_gates(context)

        assert "spread_too_wide" not in active_gates

    def test_spread_at_limit(self):
        """Test spread at exact limit."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(max_spread=0.0002)
        rm = RiskManager(cfg)

        context = {"spread": 0.0002}  # At limit
        active_gates = rm._active_gates(context)

        # Should trigger at exact limit (>= comparison)
        assert "spread_too_wide" in active_gates


@pytest.mark.unit
class TestActiveGates:
    """Unit tests for gate aggregation."""

    def test_active_gates_multiple(self):
        """Test multiple gates activating simultaneously."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(
            enabled=True,
            max_drawdown_pct=0.2,
            volatility_threshold=0.02,
            max_spread=0.0002,
        )
        rm = RiskManager(cfg)

        # Trigger multiple gates
        rm.update_equity(100.0)
        rm.update_equity(75.0)  # 25% drawdown

        context = {
            "volatility": 0.03,
            "spread": 0.0003,
        }

        active_gates = rm._active_gates(context)

        # Should have at least drawdown
        assert "max_drawdown" in active_gates
        # Should have volatility
        assert "volatility_throttle" in active_gates
        # Should have spread
        assert "spread_too_wide" in active_gates

    def test_active_gates_disabled(self):
        """Test that no gates activate when disabled."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig(enabled=False)
        rm = RiskManager(cfg)

        rm.update_equity(100.0)
        rm.update_equity(50.0)  # Would trigger drawdown

        context = {"volatility": 0.5, "spread": 0.1}
        active_gates = rm._active_gates(context)

        # No gates should be active when disabled
        assert len(active_gates) == 0

    def test_active_gates_empty_context(self):
        """Test with empty context."""
        from risk.risk_manager import RiskConfig, RiskManager

        cfg = RiskConfig()
        rm = RiskManager(cfg)

        active_gates = rm._active_gates({})

        # Should return list (may be empty if no gates triggered)
        assert isinstance(active_gates, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
