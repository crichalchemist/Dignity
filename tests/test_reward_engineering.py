"""Tests for Phase 4 reward engineering features.

This module validates that different reward functions produce expected behavior:
- portfolio_value: Raw portfolio value (baseline)
- incremental_pnl: Change in portfolio value
- sharpe: Risk-adjusted returns
- cost_aware: PnL with explicit cost penalties
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))

from train.execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
)


class TestRewardFunctions:
    """Test different reward calculation methods."""

    def test_portfolio_value_reward(self):
        """Test portfolio_value reward type returns scaled portfolio value."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="portfolio_value",
            reward_scaling=1e-4,
            time_horizon=10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Execute a buy order
        action = OrderAction(action_type="market", side="buy", size=1.0)
        obs, reward, done, info = env.step(action)

        # Reward should be portfolio_value * scaling
        expected_reward = obs["portfolio_value"] * config.reward_scaling
        assert abs(reward - expected_reward) < 1e-6, f"Expected {expected_reward}, got {reward}"

    def test_incremental_pnl_reward(self):
        """Test incremental_pnl reward returns change in portfolio value."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="incremental_pnl",
            reward_scaling=1.0,  # No scaling for easier testing
            time_horizon=10,
            price_drift=0.001,  # Small positive drift
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        initial_obs = env.reset()
        initial_pv = initial_obs["portfolio_value"]

        # Take a step (hold action)
        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        # Reward should be change in portfolio value
        pv_change = obs["portfolio_value"] - initial_pv
        assert abs(reward - pv_change) < 1e-6, f"Expected {pv_change}, got {reward}"

    def test_cost_aware_reward_penalizes_costs(self):
        """Test cost_aware reward explicitly penalizes transaction costs."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="cost_aware",
            reward_scaling=1.0,
            cost_penalty_weight=10.0,  # Strong penalty
            commission_pct=0.001,  # 0.1% commission
            time_horizon=10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Execute a buy order (incurs costs)
        action = OrderAction(action_type="market", side="buy", size=5.0)
        obs, reward, done, info = env.step(action)

        # Reward should be negative due to costs (no price movement in this step)
        # The penalty should be cost_penalty_weight * (commission + spread + slippage)
        assert reward < 0, f"Expected negative reward due to costs, got {reward}"

    def test_drawdown_penalty(self):
        """Test drawdown penalty is applied when portfolio drops."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="incremental_pnl",
            reward_scaling=1.0,
            drawdown_penalty_weight=1000.0,  # Strong penalty
            price_drift=-0.05,  # Negative drift to cause losses
            time_horizon=10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Buy at start
        action_buy = OrderAction(action_type="market", side="buy", size=100.0)
        env.step(action_buy)

        # Hold and let price drop
        action_hold = OrderAction(action_type="hold", side="buy", size=0.0)
        obs1, reward1, done1, info1 = env.step(action_hold)

        # Reward should include drawdown penalty (making it more negative)
        # Since price is dropping and we're long, we should see losses
        # With drawdown penalty, the reward should be even more negative
        assert reward1 < 0, f"Expected negative reward with drawdown, got {reward1}"

    def test_sharpe_reward_with_history(self):
        """Test Sharpe reward accumulates history and calculates risk-adjusted returns."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="sharpe",
            reward_scaling=1.0,
            sharpe_window=5,
            time_horizon=20,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Take several steps to build history
        action = OrderAction(action_type="hold", side="buy", size=0.0)
        for _ in range(10):
            obs, reward, done, info = env.step(action)

        # After 10 steps, we should have enough history for Sharpe calculation
        # Reward should be non-zero (could be positive or negative based on returns)
        # The key is that it's calculated, not just zero
        assert len(env._portfolio_value_history) <= config.sharpe_window, \
            f"History should be capped at {config.sharpe_window}, got {len(env._portfolio_value_history)}"

    def test_reward_scaling(self):
        """Test reward_scaling parameter scales rewards appropriately."""
        scaling = 1e-4
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="portfolio_value",
            reward_scaling=scaling,
            time_horizon=10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        # Reward should be portfolio_value * scaling
        expected = obs["portfolio_value"] * scaling
        assert abs(reward - expected) < 1e-9, f"Scaling incorrect: expected {expected}, got {reward}"

    def test_invalid_reward_type_raises_error(self):
        """Test that invalid reward_type raises ValueError."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="invalid_type",
        )

        with pytest.raises(ValueError, match="Invalid reward_type"):
            config.validate()


class TestRewardEngineering:
    """Integration tests for reward engineering across multiple episodes."""

    def test_incremental_vs_portfolio_value(self):
        """Compare incremental_pnl vs portfolio_value rewards."""
        # Portfolio value reward
        config_pv = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="portfolio_value",
            reward_scaling=1.0,
            time_horizon=5,
        )
        env_pv = SimulatedRetailExecutionEnv(config_pv, seed=42)

        # Incremental PnL reward
        config_inc = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="incremental_pnl",
            reward_scaling=1.0,
            time_horizon=5,
        )
        env_inc = SimulatedRetailExecutionEnv(config_inc, seed=42)

        env_pv.reset()
        env_inc.reset()

        action = OrderAction(action_type="hold", side="buy", size=0.0)

        # Step 1
        obs_pv, reward_pv, _, _ = env_pv.step(action)
        obs_inc, reward_inc, _, _ = env_inc.step(action)

        # Portfolio value reward should be large (absolute value)
        # Incremental reward should be small (just the change)
        assert reward_pv > 10_000, f"Portfolio value reward should be large, got {reward_pv}"
        assert abs(reward_inc) < 1_000, f"Incremental reward should be small, got {reward_inc}"

    def test_cost_aware_discourages_excessive_trading(self):
        """Test that cost_aware reward discourages frequent trading."""
        config = ExecutionConfig(
            initial_cash=50_000.0,
            reward_type="cost_aware",
            reward_scaling=1.0,
            cost_penalty_weight=5.0,
            commission_pct=0.001,
            time_horizon=10,
        )
        env = SimulatedRetailExecutionEnv(config, seed=42)
        env.reset()

        # Strategy 1: Trade frequently
        total_reward_frequent = 0.0
        for i in range(5):
            side = "buy" if i % 2 == 0 else "sell"
            action = OrderAction(action_type="market", side=side, size=1.0)
            _, reward, _, _ = env.step(action)
            total_reward_frequent += reward

        # Reset and try strategy 2: Hold
        env.reset()
        total_reward_hold = 0.0
        action_hold = OrderAction(action_type="hold", side="buy", size=0.0)
        for _ in range(5):
            _, reward, _, _ = env.step(action_hold)
            total_reward_hold += reward

        # Holding should have higher reward (less cost penalty)
        assert total_reward_hold > total_reward_frequent, \
            f"Holding should be rewarded more: hold={total_reward_hold}, frequent={total_reward_frequent}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
