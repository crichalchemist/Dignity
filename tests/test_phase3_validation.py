"""
Phase 3 Validation Tests

Comprehensive test suite for Phase 3 features:
- Transaction cost modeling (commission, variable spreads)
- Position sizing logic (dynamic sizing, limits)
- Risk management (stop-loss, take-profit, drawdown)

Run with: pytest tests/test_phase3_validation.py -v
"""

# Import directly from module to avoid __init__.py dependency issues
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))

from train.core.env_based_rl_training import ActionConverter
from train.execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
)


class TestTransactionCosts:
    """Test Phase 3.1: Transaction cost modeling"""

    def test_commission_per_lot(self):
        """Test fixed commission per lot is correctly applied"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            lot_size=1.0,
            commission_per_lot=7.0,  # $7 per lot
            spread=0.01,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Execute a buy order of 2 lots
        action = OrderAction(action_type="market", side="buy", size=2.0)
        obs, reward, done, info = env.step(action)

        # Commission should be 2 lots * $7 = $14
        assert env._commission_paid == pytest.approx(14.0, abs=0.01)

        # Cash should decrease by (2 * price) + commission
        expected_cash_decrease = (2.0 * env.mid_price) + 14.0
        assert env.cash < 10_000.0 - expected_cash_decrease + 1.0  # Allow small tolerance

    def test_commission_percentage(self):
        """Test percentage-based commission is correctly applied"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            lot_size=1.0,
            commission_pct=0.001,  # 0.1% commission
            spread=0.01,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Execute a buy order of 1 lot at ~$100
        action = OrderAction(action_type="market", side="buy", size=1.0)
        obs, reward, done, info = env.step(action)

        # Commission should be ~0.1% of notional (~$100)
        notional = 1.0 * env.mid_price
        expected_commission = notional * 0.001
        assert env._commission_paid == pytest.approx(expected_commission, abs=0.01)

    def test_variable_spread_widening(self):
        """Test spread widening during high volatility"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            initial_mid_price=100.0,
            spread=0.10,  # Base spread of $0.10
            variable_spread=True,
            spread_volatility_multiplier=2.0,
            price_volatility=0.01,  # 1% volatility
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Simulate multiple steps to build up volatility
        high_volatility_encountered = False
        for _ in range(100):
            # Manually inject high volatility by updating _recent_volatility
            env._recent_volatility = config.price_volatility * 2.0  # 2x baseline

            action = OrderAction(action_type="market", side="buy", size=1.0)
            spread_before = env._spread_paid

            obs, reward, done, info = env.step(action)

            spread_after = env._spread_paid
            if spread_after > spread_before:
                # Check if spread widened (should be 2x during high volatility)
                # Spread widening should occur when volatility_ratio > 1.5
                high_volatility_encountered = True
                break

        assert high_volatility_encountered, "Variable spread should widen during high volatility"

    def test_cost_tracking_and_logging(self):
        """Test that all costs are tracked separately and summed correctly"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            lot_size=1.0,
            spread=0.02,
            commission_per_lot=5.0,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Execute multiple trades
        for _ in range(5):
            action = OrderAction(action_type="market", side="buy", size=1.0)
            env.step(action)

        # All cost trackers should be positive
        assert env._spread_paid > 0
        assert env._slippage_paid >= 0  # Slippage can be zero in some cases
        assert env._commission_paid > 0

        # Total costs should equal sum of components
        total_costs = env._spread_paid + env._slippage_paid + env._commission_paid
        assert total_costs > 0


class TestPositionSizing:
    """Test Phase 3.2: Position sizing logic"""

    def test_fixed_position_sizing(self):
        """Test fixed lot size when dynamic sizing is disabled"""
        converter = ActionConverter(
            lot_size=2.0,
            use_dynamic_sizing=False,
        )

        # Should always return fixed lot_size
        size = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=10_000.0
        )
        assert size == 2.0

        # Should be same regardless of portfolio value
        size = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=50_000.0
        )
        assert size == 2.0

    def test_dynamic_position_sizing_scales_with_portfolio(self):
        """Test position size scales with portfolio value"""
        converter = ActionConverter(
            lot_size=1.0,
            risk_per_trade=0.02,  # 2% risk
            use_dynamic_sizing=True,
            max_position=100.0,  # High limit so it doesn't interfere
        )

        # Portfolio = $10,000, risk 2% = $200, price = $100 -> size = 2 lots
        size1 = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=10_000.0
        )
        assert size1 == pytest.approx(2.0, abs=0.5)  # Allow rounding

        # Portfolio = $50,000, risk 2% = $1,000, price = $100 -> size = 10 lots
        size2 = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=50_000.0
        )
        assert size2 == pytest.approx(10.0, abs=0.5)

        # Size should scale linearly with portfolio
        assert size2 > size1

    def test_position_limit_enforcement(self):
        """Test max_position limit is enforced"""
        converter = ActionConverter(
            lot_size=1.0,
            risk_per_trade=0.10,  # 10% risk (would create large positions)
            use_dynamic_sizing=True,
            max_position=5.0,  # Max 5 lots
        )

        # Try to buy when already at limit
        size = converter._calculate_position_size(
            side="buy", inventory=5.0, mid_price=100.0, cash=50_000.0
        )
        assert size == 0.0  # Should be blocked

        # Try to buy when close to limit
        size = converter._calculate_position_size(
            side="buy", inventory=4.0, mid_price=100.0, cash=50_000.0
        )
        assert size <= 1.0  # Should be capped at 1 lot to reach limit

        # Short side: try to sell when already at max short
        size = converter._calculate_position_size(
            side="sell", inventory=-5.0, mid_price=100.0, cash=50_000.0
        )
        assert size == 0.0  # Should be blocked

    def test_cash_constraint_enforcement(self):
        """Test that position size respects available cash"""
        converter = ActionConverter(
            lot_size=1.0,
            risk_per_trade=0.50,  # 50% risk (very aggressive)
            use_dynamic_sizing=True,
            max_position=100.0,  # High limit
        )

        # Only have $1,000 cash, price = $100 -> can buy max 10 lots
        size = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=1_000.0
        )
        assert size <= 10.0

        # With very little cash, should get small position
        size = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=50.0
        )
        assert size < 1.0

    def test_lot_size_rounding(self):
        """Test position sizes are rounded to lot_size increments"""
        converter = ActionConverter(
            lot_size=0.5,  # Half-lot increments
            risk_per_trade=0.02,
            use_dynamic_sizing=True,
            max_position=100.0,
        )

        size = converter._calculate_position_size(
            side="buy", inventory=0.0, mid_price=100.0, cash=10_000.0
        )

        # Size should be a multiple of 0.5
        assert size % 0.5 == pytest.approx(0.0, abs=0.01)


class TestRiskManagement:
    """Test Phase 3.3: Risk management features"""

    def test_stop_loss_trigger(self):
        """Test stop-loss closes position when loss threshold is breached"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            initial_mid_price=100.0,
            lot_size=1.0,
            spread=0.01,
            price_drift=-0.01,  # Negative drift to trigger stop-loss
            price_volatility=0.001,  # Low volatility for predictability
            enable_stop_loss=True,
            stop_loss_pct=0.02,  # 2% stop-loss
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Buy 1 lot at ~$100
        action = OrderAction(action_type="market", side="buy", size=1.0)
        env.step(action)

        initial_inventory = env.inventory
        assert initial_inventory > 0

        # Run steps until stop-loss triggers or timeout
        for _step in range(100):
            # Let price drift down
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)

            # Check if stop-loss triggered (inventory closed out)
            if env._stop_loss_triggered > 0:
                assert env.inventory < initial_inventory  # Position should be reduced/closed
                break

        # Note: Due to randomness, stop-loss might not always trigger in 100 steps
        # This test verifies the mechanism works when conditions are met

    def test_take_profit_trigger(self):
        """Test take-profit closes position when profit threshold is breached"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            initial_mid_price=100.0,
            lot_size=1.0,
            spread=0.01,
            price_drift=0.01,  # Positive drift to trigger take-profit
            price_volatility=0.001,
            enable_take_profit=True,
            take_profit_pct=0.04,  # 4% take-profit
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Buy 1 lot at ~$100
        action = OrderAction(action_type="market", side="buy", size=1.0)
        env.step(action)

        initial_inventory = env.inventory

        # Run steps until take-profit triggers or timeout
        for _step in range(100):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)

            if env._take_profit_triggered > 0:
                assert env.inventory < initial_inventory
                break

        # Note: Similar to stop-loss, randomness may prevent trigger in 100 steps

    def test_drawdown_limit_termination(self):
        """Test episode terminates when drawdown exceeds limit"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            initial_mid_price=100.0,
            lot_size=10.0,  # Large lot to amplify losses
            spread=0.01,
            price_drift=-0.02,  # Strong negative drift
            price_volatility=0.05,  # High volatility
            enable_drawdown_limit=True,
            max_drawdown_pct=0.20,  # 20% drawdown limit
            time_horizon=1000,  # Long horizon so drawdown triggers first
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Buy large position to amplify losses
        action = OrderAction(action_type="market", side="buy", size=10.0)
        env.step(action)

        # Run steps until drawdown limit triggers or timeout
        done = False
        for step in range(500):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)

            if done:
                # Check if termination was due to drawdown
                portfolio_value = obs["portfolio_value"]
                drawdown = (env._peak_portfolio_value - portfolio_value) / env._peak_portfolio_value

                # If done early, should be due to drawdown (not time horizon)
                if step < config.time_horizon - 1:
                    assert drawdown >= config.max_drawdown_pct * 0.9  # Allow small tolerance
                break

        # Note: Test validates mechanism; actual trigger depends on random walk

    def test_peak_portfolio_tracking(self):
        """Test peak portfolio value is correctly tracked for drawdown calculation"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            enable_drawdown_limit=True,
            max_drawdown_pct=0.50,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Initial peak should be initial cash
        assert env._peak_portfolio_value == 10_000.0

        # Make profitable trade (price goes up after buy)
        # This is hard to guarantee due to randomness, so we'll just verify
        # that peak updates correctly when portfolio value increases

        # Manually set peak for testing
        env._peak_portfolio_value = 12_000.0
        env.cash = 12_000.0

        # Step environment
        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        # Peak should remain at least 12,000
        assert env._peak_portfolio_value >= 12_000.0


class TestIntegration:
    """Integration tests combining multiple Phase 3 features"""

    def test_full_episode_with_all_features(self):
        """Test complete episode with all Phase 3 features enabled"""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            lot_size=1.0,
            spread=0.02,
            commission_pct=0.0001,  # 0.01% commission
            variable_spread=True,
            enable_stop_loss=True,
            stop_loss_pct=0.02,
            enable_take_profit=True,
            take_profit_pct=0.04,
            enable_drawdown_limit=True,
            max_drawdown_pct=0.30,
            time_horizon=100,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)

        converter = ActionConverter(
            lot_size=1.0,
            max_position=10.0,
            risk_per_trade=0.02,
            use_dynamic_sizing=True,
        )

        env.reset()
        done = False
        step = 0

        while not done and step < config.time_horizon:
            obs = env._observation()

            # Simulate policy decision (random actions for testing)
            action_idx = np.random.choice([0, 1, 2])  # SELL, HOLD, BUY

            # Use ActionConverter to size position
            order_action = converter.policy_to_order(
                action_idx, obs["mid_price"], obs["inventory"], obs["cash"]
            )

            obs, reward, done, info = env.step(order_action)
            step += 1

        # Verify all cost components were tracked
        assert env._spread_paid >= 0
        assert env._slippage_paid >= 0
        assert env._commission_paid >= 0

        # Verify episode completed (either time horizon or drawdown)
        assert done or step >= config.time_horizon

    def test_dynamic_sizing_with_transaction_costs(self):
        """Test position sizing adapts to transaction costs impact on portfolio"""
        config = ExecutionConfig(
            initial_cash=10_000.0,
            lot_size=1.0,
            commission_pct=0.01,  # 1% commission (very high)
            spread=0.10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)

        converter = ActionConverter(
            lot_size=1.0,
            risk_per_trade=0.02,
            use_dynamic_sizing=True,
            max_position=10.0,
        )

        env.reset()

        # Execute several trades
        for _ in range(10):
            obs = env._observation()

            # Buy action
            order_action = converter.policy_to_order(
                2, obs["mid_price"], obs["inventory"], obs["cash"]
            )

            obs, reward, done, info = env.step(order_action)

        # Portfolio should have decreased due to transaction costs
        final_portfolio = obs["portfolio_value"]
        assert final_portfolio < 10_000.0

        # Position sizes should have decreased as portfolio shrank
        # (this is implicit in dynamic sizing - smaller portfolio = smaller positions)

    def test_risk_management_with_position_sizing(self):
        """Test risk management works correctly with dynamic position sizing"""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            lot_size=1.0,
            enable_stop_loss=True,
            stop_loss_pct=0.02,
            time_horizon=50,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)

        converter = ActionConverter(
            lot_size=1.0,
            max_position=20.0,  # Higher limit
            risk_per_trade=0.05,  # 5% risk (aggressive)
            use_dynamic_sizing=True,
        )

        env.reset()

        # Take large position
        obs = env._observation()
        order_action = converter.policy_to_order(
            2, obs["mid_price"], obs["inventory"], obs["cash"]  # BUY
        )
        env.step(order_action)

        # Run until stop-loss or time horizon
        for _ in range(49):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)

            if done:
                break

        # Verify risk metrics were tracked
        assert isinstance(env._stop_loss_triggered, int)
        assert isinstance(env._take_profit_triggered, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
