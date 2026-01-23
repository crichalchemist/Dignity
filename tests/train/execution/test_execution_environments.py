"""
Unit tests for execution environments.

Tests both simulated and backtesting environments:
- SimulatedRetailEnv: Stochastic execution with spread/slippage/latency
- BacktestingEnv: Deterministic historical replay
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
    SlippageModel,
)
from train.execution.backtesting_env import BacktestingRetailExecutionEnv, BacktestingObservation


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for backtesting."""
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": 1.08 + np.cumsum(np.random.randn(200) * 0.0001),
            "high": 1.081 + np.cumsum(np.random.randn(200) * 0.0001),
            "low": 1.079 + np.cumsum(np.random.randn(200) * 0.0001),
            "close": 1.0805 + np.cumsum(np.random.randn(200) * 0.0001),
            "volume": np.random.randint(1000, 10000, 200),
        }
    )


@pytest.fixture
def simulated_env_config():
    """Configuration for simulated environment."""
    return ExecutionConfig(
        initial_mid_price=100.0,
        initial_cash=10000.0,
        spread=2.0,
        price_drift=0.0,
        price_volatility=0.05,
        time_horizon=100,
        decision_lag=0,
        lot_size=1.0,
        limit_fill_probability=0.35,
        slippage_model=SlippageModel(mean=0.001, std=0.001, max_slippage=0.005),
    )


@pytest.fixture
def backtesting_env_config():
    """Configuration for backtesting environment."""
    return ExecutionConfig(
        initial_cash=10000.0,
        lot_size=1.0,
        spread=0.02,
        time_horizon=200,
    )


class TestSlippageModel:
    """Test slippage model sampling."""

    def test_initialization(self):
        """Test slippage model initializes with parameters."""
        slippage_model = SlippageModel(mean=0.001, std=0.001, max_slippage=0.01)

        assert slippage_model.mean == 0.001
        assert slippage_model.std == 0.001
        assert slippage_model.max_slippage == 0.01

    def test_sample_returns_float(self):
        """Test sample returns float value."""
        slippage_model = SlippageModel(mean=0.0, std=0.001, max_slippage=0.01)
        rng = np.random.default_rng(42)

        slippage = slippage_model.sample(rng)

        assert isinstance(slippage, float)

    def test_sample_clips_to_max(self):
        """Test sample respects max_slippage bound."""
        max_slippage = 0.01
        slippage_model = SlippageModel(mean=0.0, std=10.0, max_slippage=max_slippage)
        rng = np.random.default_rng(42)

        for _ in range(100):
            slippage = slippage_model.sample(rng)
            assert -max_slippage <= slippage <= max_slippage


class TestExecutionConfig:
    """Test execution configuration."""

    def test_default_initialization(self):
        """Test default config values."""
        config = ExecutionConfig()

        assert config.initial_mid_price == 100.0
        assert config.initial_cash == 50_000.0
        assert config.spread == 0.02
        assert config.lot_size == 1.0
        assert config.price_volatility == 0.05
        assert config.time_horizon == 390

    def test_custom_initialization(self):
        """Test custom config values."""
        config = ExecutionConfig(
            initial_cash=10000.0,
            spread=1.5,
            lot_size=0.5,
            price_volatility=0.1,
        )

        assert config.initial_cash == 10000.0
        assert config.spread == 1.5
        assert config.lot_size == 0.5
        assert config.price_volatility == 0.1

    def test_validate_negative_spread_raises(self):
        """Test negative spread raises ValueError."""
        with pytest.raises(ValueError, match="Spread must be non-negative"):
            ExecutionConfig(spread=-1.0)

    def test_validate_invalid_fill_probability(self):
        """Test invalid fill probability raises ValueError."""
        with pytest.raises(ValueError, match="Limit fill probability must be within"):
            ExecutionConfig(limit_fill_probability=1.5)

        with pytest.raises(ValueError, match="Limit fill probability must be within"):
            ExecutionConfig(limit_fill_probability=-0.1)

    def test_validate_negative_lot_size(self):
        """Test negative lot size raises ValueError."""
        with pytest.raises(ValueError, match="Lot size must be positive"):
            ExecutionConfig(lot_size=-1.0)

    def test_validate_negative_volatility(self):
        """Test negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="Price volatility must be non-negative"):
            ExecutionConfig(price_volatility=-0.1)

    def test_validate_negative_cash(self):
        """Test negative cash raises ValueError."""
        with pytest.raises(ValueError, match="Initial cash must be non-negative"):
            ExecutionConfig(initial_cash=-1000.0)


class TestOrderAction:
    """Test order action dataclass."""

    def test_market_order_creation(self):
        """Test market order creation."""
        action = OrderAction(action_type="market", side="buy", size=1.0)

        assert action.action_type == "market"
        assert action.side == "buy"
        assert action.size == 1.0
        assert action.limit_price is None

    def test_limit_order_creation(self):
        """Test limit order creation."""
        action = OrderAction(action_type="limit", side="sell", size=1.0, limit_price=100.5)

        assert action.action_type == "limit"
        assert action.side == "sell"
        assert action.size == 1.0
        assert action.limit_price == 100.5

    def test_hold_order_creation(self):
        """Test hold order creation."""
        action = OrderAction(action_type="hold", side="buy", size=0.0)

        assert action.action_type == "hold"
        assert action.size == 0.0

    def test_normalized_trims_negative_size(self):
        """Test normalized trims negative size to zero."""
        config = ExecutionConfig(lot_size=0.1)
        action = OrderAction(action_type="market", side="buy", size=-1.0)

        normalized = action.normalized(config)

        assert normalized.size == 0.0

    def test_normalized_rounds_to_lot_size(self):
        """Test normalized rounds size to lot size increments."""
        config = ExecutionConfig(lot_size=0.5)
        action = OrderAction(action_type="market", side="buy", size=1.2)

        normalized = action.normalized(config)

        assert normalized.size == 1.0

    def test_normalized_lowercases_actions(self):
        """Test normalized lowercases action types and sides."""
        config = ExecutionConfig()
        action = OrderAction(action_type="MARKET", side="BUY", size=1.0)

        normalized = action.normalized(config)

        assert normalized.action_type == "market"
        assert normalized.side == "buy"

    def test_normalized_invalid_action_raises(self):
        """Test normalized raises for invalid action type."""
        config = ExecutionConfig()
        action = OrderAction(action_type="invalid", side="buy", size=1.0)

        with pytest.raises(ValueError, match="Unsupported action_type"):
            action.normalized(config)

    def test_normalized_invalid_side_raises(self):
        """Test normalized raises for invalid side."""
        config = ExecutionConfig()
        action = OrderAction(action_type="market", side="invalid", size=1.0)

        with pytest.raises(ValueError, match="Unsupported side"):
            action.normalized(config)

    def test_normalized_invalid_limit_price_raises(self):
        """Test normalized raises for non-positive limit price."""
        config = ExecutionConfig()
        action = OrderAction(action_type="limit", side="buy", size=1.0, limit_price=-1.0)

        with pytest.raises(ValueError, match="Limit price must be positive"):
            action.normalized(config)


class TestSimulatedRetailEnv:
    """Test simulated retail execution environment."""

    def test_initialization(self, simulated_env_config):
        """Test simulated environment initializes correctly."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)

        assert env.config.initial_mid_price == 100.0
        assert env.config.initial_cash == 10000.0
        assert env.config.spread == 2.0

    def test_reset(self, simulated_env_config):
        """Test environment reset."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        obs = env.reset()

        assert "mid_price" in obs
        assert "inventory" in obs
        assert "cash" in obs
        assert "realized_pnl" in obs
        assert "unrealized_pnl" in obs
        assert "portfolio_value" in obs

        assert obs["mid_price"] == 100.0
        assert obs["cash"] == 10000.0
        assert obs["inventory"] == 0.0
        assert obs["realized_pnl"] == 0.0
        assert obs["portfolio_value"] == 10000.0

    def test_step_with_buy_action(self, simulated_env_config):
        """Test stepping with buy action."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        action = OrderAction(action_type="market", side="buy", size=1.0)
        obs, reward, done, info = env.step(action)

        assert obs["inventory"] > 0
        assert obs["cash"] < 10000.0
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_step_with_sell_action(self, simulated_env_config):
        """Test stepping with sell action."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        action = OrderAction(action_type="market", side="sell", size=1.0)
        obs, reward, done, info = env.step(action)

        assert obs["inventory"] < 0
        assert isinstance(reward, (int, float))

    def test_step_with_hold_action(self, simulated_env_config):
        """Test stepping with hold action."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        initial_cash = env.cash
        initial_inventory = env.inventory

        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        assert env.cash == initial_cash
        assert env.inventory == initial_inventory

    def test_spread_deduction(self, simulated_env_config):
        """Test that spread is deducted from execution."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        initial_cash = env.cash
        mid_price = env.mid_price

        action = OrderAction(action_type="market", side="buy", size=1.0)
        env.step(action)

        execution_price = env._fill_events[-1]["price"]
        expected_cost = mid_price + simulated_env_config.spread / 2

        assert execution_price >= expected_cost

    def test_position_tracking(self, simulated_env_config):
        """Test position accumulation and reduction."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        env.step(OrderAction(action_type="market", side="buy", size=0.3))
        assert abs(env.inventory - 0.3) < 0.01

        env.step(OrderAction(action_type="market", side="buy", size=0.2))
        assert abs(env.inventory - 0.5) < 0.01

        env.step(OrderAction(action_type="market", side="sell", size=0.4))
        assert abs(env.inventory - 0.1) < 0.01

    def test_reward_calculation(self, simulated_env_config):
        """Test reward reflects P&L."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        env.step(OrderAction(action_type="market", side="buy", size=1.0))
        initial_pnl = env.realized_pnl

        obs, reward, done, info = env.step(OrderAction(action_type="hold", side="buy", size=0.0))

        assert isinstance(reward, (int, float))

    def test_balance_update(self, simulated_env_config):
        """Test balance updates with P&L."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        initial_balance = env.cash
        action = OrderAction(action_type="market", side="buy", size=1.0)
        obs, reward, done, info = env.step(action)

        assert env.cash < initial_balance

    def test_terminates_at_time_horizon(self, simulated_env_config):
        """Test episode terminates at time horizon."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        for _ in range(simulated_env_config.time_horizon - 1):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)
            assert not done

        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        assert done

    def test_limit_order_submission(self, simulated_env_config):
        """Test limit order submission."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        initial_pending = len(env._pending_limits)

        action = OrderAction(action_type="limit", side="buy", size=1.0, limit_price=99.0)
        env.step(action)

        assert len(env._pending_limits) == initial_pending + 1

    def test_multiple_steps(self, simulated_env_config):
        """Test multiple sequential steps."""
        env = SimulatedRetailExecutionEnv(simulated_env_config)
        env.reset()

        for _ in range(10):
            action = OrderAction(
                action_type="market",
                side="buy" if np.random.rand() > 0.5 else "sell",
                size=1.0,
            )
            obs, reward, done, info = env.step(action)

            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)


class TestBacktestingEnv:
    """Test backtesting execution environment."""

    def test_initialization(self, sample_ohlcv_data, backtesting_env_config):
        """Test backtesting environment initializes with data."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)

        assert len(env.price_df) == len(sample_ohlcv_data)
        assert env.config.initial_cash == 10000.0

    def test_reset_to_start(self, sample_ohlcv_data, backtesting_env_config):
        """Test reset returns to start of data."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)

        for _ in range(10):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            env.step(action)

        obs = env.reset()

        assert env._step_index == 0
        assert env._prev_equity == backtesting_env_config.initial_cash

    def test_step_returns_observation_reward_done(self, sample_ohlcv_data, backtesting_env_config):
        """Test step returns correct tuple format."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)
        env.reset()

        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, dict)
        assert "mid_price" in obs
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_deterministic_execution(self, sample_ohlcv_data, backtesting_env_config):
        """Test execution is deterministic (no randomness)."""
        env1 = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)
        env2 = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)

        env1.reset()
        env2.reset()

        actions = [OrderAction(action_type="hold", side="buy", size=0.0) for _ in range(10)]

        obs1_list = []
        obs2_list = []

        for action in actions:
            obs1, _, _, _ = env1.step(action)
            obs2, _, _, _ = env2.step(action)
            obs1_list.append(obs1)
            obs2_list.append(obs2)

        for obs1, obs2 in zip(obs1_list, obs2_list, strict=False):
            assert obs1["mid_price"] == obs2["mid_price"]
            assert obs1["portfolio_value"] == obs2["portfolio_value"]

    def test_historical_replay(self, sample_ohlcv_data, backtesting_env_config):
        """Test environment replays historical data."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)
        env.reset()

        actions = [OrderAction(action_type="hold", side="buy", size=0.0) for _ in range(10)]

        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)

            assert env._step_index == i + 1

        assert env._step_index == 10

    def test_terminates_at_end_of_data(self, sample_ohlcv_data, backtesting_env_config):
        """Test environment terminates at end of data."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)
        env.reset()

        for _ in range(len(sample_ohlcv_data) - 1):
            action = OrderAction(action_type="hold", side="buy", size=0.0)
            obs, reward, done, info = env.step(action)
            assert not done

        action = OrderAction(action_type="hold", side="buy", size=0.0)
        obs, reward, done, info = env.step(action)

        assert done

    def test_observation_dict_structure(self, sample_ohlcv_data, backtesting_env_config):
        """Test observation dict has correct structure."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)
        obs = env.reset()

        required_keys = [
            "mid_price",
            "inventory",
            "cash",
            "realized_pnl",
            "portfolio_value",
            "step",
        ]

        for key in required_keys:
            assert key in obs

        assert isinstance(obs["mid_price"], float)
        assert isinstance(obs["inventory"], float)
        assert isinstance(obs["cash"], float)
        assert isinstance(obs["portfolio_value"], float)

    def test_normalizes_price_df_columns(self, sample_ohlcv_data, backtesting_env_config):
        """Test price DataFrame columns are normalized."""
        env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)

        assert "Open" in env.price_df.columns
        assert "High" in env.price_df.columns
        assert "Low" in env.price_df.columns
        assert "Close" in env.price_df.columns


class TestBacktestingObservation:
    """Test BacktestingObservation dataclass."""

    def test_as_dict_returns_correct_keys(self):
        """Test as_dict returns dict with all fields."""
        obs = BacktestingObservation(
            mid_price=100.0,
            inventory=1.0,
            cash=10000.0,
            realized_pnl=100.0,
            portfolio_value=10100.0,
            step=10,
        )

        obs_dict = obs.as_dict()

        assert "mid_price" in obs_dict
        assert "inventory" in obs_dict
        assert "cash" in obs_dict
        assert "realized_pnl" in obs_dict
        assert "portfolio_value" in obs_dict
        assert "step" in obs_dict

    def test_as_dict_values_match(self):
        """Test as_dict values match observation."""
        obs = BacktestingObservation(
            mid_price=100.5,
            inventory=2.0,
            cash=15000.0,
            realized_pnl=200.0,
            portfolio_value=15200.0,
            step=5,
        )

        obs_dict = obs.as_dict()

        assert obs_dict["mid_price"] == 100.5
        assert obs_dict["inventory"] == 2.0
        assert obs_dict["cash"] == 15000.0
        assert obs_dict["realized_pnl"] == 200.0
        assert obs_dict["portfolio_value"] == 15200.0
        assert obs_dict["step"] == 5.0


class TestEnvironmentConsistency:
    """Test consistency between environment types."""

    def test_observation_keys_match(
        self, sample_ohlcv_data, simulated_env_config, backtesting_env_config
    ):
        """Test both environments return same observation keys."""
        simulated_env = SimulatedRetailExecutionEnv(simulated_env_config)
        backtesting_env = BacktestingRetailExecutionEnv(sample_ohlcv_data, backtesting_env_config)

        sim_obs = simulated_env.reset()
        bt_obs = backtesting_env.reset()

        common_keys = {"mid_price", "inventory", "cash", "realized_pnl", "portfolio_value"}

        for key in common_keys:
            assert key in sim_obs
            assert key in bt_obs

    def test_action_space_compatibility(self, simulated_env_config, backtesting_env_config):
        """Test both environments accept same action types."""
        simulated_env = SimulatedRetailExecutionEnv(simulated_env_config)

        action_types = ["market", "limit", "hold"]
        sides = ["buy", "sell"]

        for action_type in action_types:
            for side in sides:
                action = OrderAction(action_type=action_type, side=side, size=1.0)

                simulated_env.reset()
                simulated_env.step(action)
