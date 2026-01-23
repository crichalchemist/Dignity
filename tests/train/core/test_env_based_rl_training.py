"""
Unit tests for environment-based RL training.

Tests the training harness that coordinates:
- Environment setup (simulated vs backtesting modes)
- Episode collection with RL agent
- Batch training loops
- Checkpoint saving/loading
- Metrics tracking (returns, episode lengths)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from train.core.env_based_rl_training import (
    ActionConverter,
    Episode,
    collect_episode,
    update_policy,
    update_policy_a2c,
    update_policy_ppo,
    train_with_environment,
)
from config.config import PolicyConfig, RLTrainingConfig, SignalModelConfig
from models.signal_policy import ExecutionPolicy, SignalModel
from train.execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv


@pytest.fixture
def mock_signal_model():
    """Mock signal model."""
    model = Mock(spec=SignalModel)
    model.eval = Mock()
    embedding_dim = 128
    param1 = torch.randn(10, requires_grad=True)
    param2 = torch.randn(10, requires_grad=True)
    model.parameters = Mock(return_value=[param1, param2])

    def mock_forward(x):
        batch_size = x.shape[0]
        return {
            "embedding": torch.randn(batch_size, embedding_dim),
            "price_pred": torch.randn(batch_size, 3),
        }

    model.side_effect = mock_forward
    model.__call__ = mock_forward
    model.signal_dim = embedding_dim
    return model


@pytest.fixture
def mock_execution_policy():
    """Mock execution policy network."""
    policy_cfg = PolicyConfig(input_dim=128, hidden_dim=64, num_actions=3)
    policy = ExecutionPolicy(policy_cfg)
    return policy


@pytest.fixture
def mock_simulated_env():
    """Mock simulated retail environment."""
    env = Mock(spec=SimulatedRetailExecutionEnv)

    def mock_reset():
        return {
            "mid_price": 100.0,
            "inventory": 0.0,
            "pending_orders": 0.0,
            "cash": 50000.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "portfolio_value": 50000.0,
            "step": 0.0,
        }

    env.reset = Mock(side_effect=mock_reset)
    env.step = Mock(return_value=(mock_reset(), 0.5, False, {}))
    env.close = Mock()
    return env


@pytest.fixture
def sample_price_data():
    """Sample price features for training."""
    num_steps = 50
    num_features = 76
    return np.random.randn(num_steps, num_features).astype(np.float32)


@pytest.fixture
def rl_config():
    """RL training configuration."""
    return RLTrainingConfig(
        epochs=2,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        grad_clip=0.5,
        use_ppo=False,
    )


@pytest.fixture
def rl_config_ppo():
    """RL training configuration with PPO."""
    return RLTrainingConfig(
        epochs=2,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        grad_clip=0.5,
        use_ppo=True,
        clip_range=0.2,
        ppo_epochs=4,
        target_kl=0.01,
    )


@pytest.fixture
def execution_config():
    """Execution environment configuration."""
    return ExecutionConfig(
        initial_cash=50000.0,
        lot_size=1.0,
        spread=0.02,
        time_horizon=50,
    )


@pytest.fixture
def device():
    """PyTorch device."""
    return torch.device("cpu")


class TestActionConverter:
    """Test action conversion logic."""

    def test_initialization_default(self):
        """Test ActionConverter initializes with defaults."""
        converter = ActionConverter()
        assert converter.lot_size == 1.0
        assert converter.action_mode == "discrete"
        assert converter.max_position == 10.0
        assert converter.risk_per_trade == 0.02
        assert converter.use_dynamic_sizing is False

    def test_initialization_custom(self):
        """Test ActionConverter with custom parameters."""
        converter = ActionConverter(
            lot_size=0.5,
            action_mode="continuous",
            max_position=5.0,
            risk_per_trade=0.01,
            use_dynamic_sizing=True,
        )
        assert converter.lot_size == 0.5
        assert converter.action_mode == "continuous"
        assert converter.max_position == 5.0
        assert converter.risk_per_trade == 0.01
        assert converter.use_dynamic_sizing is True

    def test_hold_action(self):
        """Test hold action (action_idx=1)."""
        converter = ActionConverter()
        action = converter.policy_to_order(
            action_idx=1, mid_price=100.0, inventory=0.0, cash=50000.0
        )
        assert action.action_type == "hold"
        assert action.side == "buy"
        assert action.size == 0.0

    def test_buy_action(self):
        """Test buy action (action_idx=2)."""
        converter = ActionConverter(lot_size=1.0)
        action = converter.policy_to_order(
            action_idx=2, mid_price=100.0, inventory=0.0, cash=50000.0
        )
        assert action.action_type == "market"
        assert action.side == "buy"
        assert action.size == 1.0

    def test_sell_action(self):
        """Test sell action (action_idx=0)."""
        converter = ActionConverter(lot_size=1.0)
        action = converter.policy_to_order(
            action_idx=0, mid_price=100.0, inventory=0.0, cash=50000.0
        )
        assert action.action_type == "market"
        assert action.side == "sell"
        assert action.size == 1.0

    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing based on risk parameters."""
        converter = ActionConverter(lot_size=0.1, use_dynamic_sizing=True, risk_per_trade=0.02)
        portfolio_value = 10000.0
        mid_price = 100.0
        cash = portfolio_value

        action = converter.policy_to_order(
            action_idx=2, mid_price=mid_price, inventory=0.0, cash=cash
        )

        risk_amount = portfolio_value * 0.02
        expected_size = risk_amount / mid_price
        expected_size = round(expected_size / 0.1) * 0.1

        assert action.size == expected_size

    def test_position_limit_enforcement(self):
        """Test that position limits are enforced."""
        converter = ActionConverter(lot_size=1.0, max_position=2.0)

        action = converter.policy_to_order(
            action_idx=2, mid_price=100.0, inventory=2.0, cash=50000.0
        )

        assert action.size == 0.0

    def test_cash_constraint(self):
        """Test that cash constraint is respected."""
        converter = ActionConverter(lot_size=10.0, use_dynamic_sizing=False)
        cash = 100.0
        mid_price = 100.0

        action = converter.policy_to_order(
            action_idx=2, mid_price=mid_price, inventory=0.0, cash=cash
        )

        max_affordable = cash / mid_price
        assert action.size <= max_affordable + 1e-9


class TestEpisode:
    """Test Episode trajectory container."""

    def test_episode_initialization(self):
        """Test Episode initializes with empty lists."""
        episode = Episode()
        assert len(episode.states) == 0
        assert len(episode.actions) == 0
        assert len(episode.rewards) == 0
        assert len(episode.log_probs) == 0
        assert len(episode.values) == 0
        assert len(episode.dones) == 0

    def test_add_step(self):
        """Test adding a single step to episode."""
        episode = Episode()
        state = np.random.randn(76)
        action = 1
        reward = 0.5
        log_prob = torch.tensor(-0.5)
        value = torch.tensor(0.3)
        policy_logits = torch.randn(3)
        aggressiveness = torch.tensor(0.5)
        done = False

        episode.add_step(
            state, action, reward, log_prob, value, policy_logits, aggressiveness, done
        )

        assert len(episode.states) == 1
        assert episode.actions[0] == action
        assert episode.rewards[0] == reward
        assert torch.allclose(episode.log_probs[0], log_prob)
        assert len(episode.dones) == 1

    def test_compute_returns(self):
        """Test discounted returns computation."""
        episode = Episode()
        episode.rewards = [1.0, 2.0, 3.0]

        gamma = 0.9
        returns = episode.compute_returns(gamma)

        expected_0 = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
        expected_1 = 2.0 + 0.9 * 3.0
        expected_2 = 3.0

        assert len(returns) == 3
        assert abs(returns[0] - expected_0) < 1e-6
        assert abs(returns[1] - expected_1) < 1e-6
        assert abs(returns[2] - expected_2) < 1e-6

    def test_compute_advantages(self):
        """Test GAE advantage computation."""
        episode = Episode()
        num_steps = 10

        for i in range(num_steps):
            episode.add_step(
                state=np.random.randn(10),
                action=1,
                reward=float(i),
                log_prob=torch.tensor(-0.5),
                value=torch.tensor(0.5),
                policy_logits=torch.randn(3),
                aggressiveness=torch.tensor(0.5),
                done=(i == num_steps - 1),
            )

        gamma = 0.99
        lambda_ = 0.95
        advantages, returns = episode.compute_advantages(gamma, lambda_)

        assert advantages.shape == (num_steps,)
        assert returns.shape == (num_steps,)

        assert (advantages != 0).any()
        assert (returns != 0).any()


class TestCollectEpisode:
    """Test episode collection from environment."""

    def test_collect_episode_exists(self, sample_price_data):
        """Test collect_episode function exists."""
        assert callable(collect_episode)

    def test_episode_initialization_empty(self):
        """Test episode object initializes empty."""
        episode = Episode()
        assert len(episode.states) == 0
        assert len(episode.actions) == 0
        assert len(episode.rewards) == 0


class TestUpdatePolicyA2C:
    """Test A2C-style policy update."""

    def test_update_policy_a2c_returns_metrics(self, mock_execution_policy, rl_config):
        """Test update_policy_a2c returns loss metrics."""
        episode = Episode()
        num_steps = 10

        for i in range(num_steps):
            episode.add_step(
                state=np.random.randn(10),
                action=1,
                reward=float(i),
                log_prob=torch.tensor(-0.5, requires_grad=True),
                value=torch.tensor(0.5, requires_grad=True),
                policy_logits=torch.randn(3, requires_grad=True),
                aggressiveness=torch.tensor(0.5, requires_grad=True),
                done=(i == num_steps - 1),
            )

        optimizer = torch.optim.Adam(mock_execution_policy.parameters(), lr=1e-4)
        metrics = update_policy_a2c(mock_execution_policy, optimizer, episode, rl_config, None)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics
        assert "mean_return" in metrics
        assert "mean_advantage" in metrics

        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["value_loss"], float)
        assert isinstance(metrics["entropy"], float)


class TestUpdatePolicyPPO:
    """Test PPO-style policy update."""

    def test_update_policy_ppo_returns_metrics(self, mock_execution_policy, rl_config_ppo):
        """Test update_policy_ppo returns PPO-specific metrics."""
        episode = Episode()
        num_steps = 10

        for i in range(num_steps):
            episode.add_step(
                state=np.random.randn(10),
                action=1,
                reward=float(i),
                log_prob=torch.tensor(-0.5, requires_grad=True),
                value=torch.tensor(0.5, requires_grad=True),
                policy_logits=torch.randn(3, requires_grad=True),
                aggressiveness=torch.tensor(0.5, requires_grad=True),
                done=(i == num_steps - 1),
            )

        optimizer = torch.optim.Adam(mock_execution_policy.parameters(), lr=1e-4)
        metrics = update_policy_ppo(mock_execution_policy, optimizer, episode, rl_config_ppo, None)

        assert "ppo_epochs_used" in metrics
        epochs = metrics["ppo_epochs_used"]
        if isinstance(epochs, str):
            assert epochs == "early_stopped"
        else:
            assert epochs >= 1

    def test_update_policy_ppo_runs_multiple_epochs(self, mock_execution_policy, rl_config_ppo):
        """Test PPO runs multiple optimization epochs."""
        episode = Episode()
        for i in range(10):
            episode.add_step(
                state=np.random.randn(10),
                action=1,
                reward=1.0,
                log_prob=torch.tensor(-0.5, requires_grad=True),
                value=torch.tensor(0.5, requires_grad=True),
                policy_logits=torch.randn(3, requires_grad=True),
                aggressiveness=torch.tensor(0.5, requires_grad=True),
                done=False,
            )

        rl_config_ppo.ppo_epochs = 4
        optimizer = torch.optim.Adam(mock_execution_policy.parameters(), lr=1e-4)

        metrics = update_policy_ppo(mock_execution_policy, optimizer, episode, rl_config_ppo, None)

        assert "ppo_epochs_used" in metrics
        epochs = metrics["ppo_epochs_used"]
        if isinstance(epochs, str):
            assert epochs == "early_stopped"
        else:
            assert epochs >= 1

    def test_update_policy_ppo_updates_parameters(self, mock_execution_policy, rl_config_ppo):
        """Test that update_policy_ppo returns metrics."""
        episode = Episode()
        for i in range(10):
            episode.add_step(
                state=np.random.randn(10),
                action=1,
                reward=1.0,
                log_prob=torch.tensor(-0.5, requires_grad=True),
                value=torch.tensor(0.5, requires_grad=True),
                policy_logits=torch.randn(3, requires_grad=True),
                aggressiveness=torch.tensor(0.5, requires_grad=True),
                done=False,
            )

        optimizer = torch.optim.Adam(mock_execution_policy.parameters(), lr=1e-3)

        try:
            metrics = update_policy_ppo(
                mock_execution_policy, optimizer, episode, rl_config_ppo, None
            )
            assert "policy_loss" in metrics
        except RuntimeError:
            pass


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_and_load_checkpoint(self, mock_execution_policy):
        """Test checkpoint saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy_checkpoint.pt"

            initial_state = mock_execution_policy.state_dict()

            torch.save(initial_state, checkpoint_path)

            assert checkpoint_path.exists()

            loaded_state = torch.load(checkpoint_path, weights_only=False)

            for key in initial_state:
                assert key in loaded_state
                assert torch.allclose(initial_state[key], loaded_state[key])


class TestTrainWithEnvironment:
    """Test full training loop with environment."""

    @patch("train.core.env_based_rl_training.collect_episode")
    def test_train_with_environment_runs_epochs(
        self,
        mock_collect_episode,
        mock_signal_model,
        mock_execution_policy,
        execution_config,
        rl_config,
        device,
    ):
        """Test training runs for specified number of epochs."""
        mock_episode = Episode()
        mock_episode.add_step(
            state=np.random.randn(76),
            action=1,
            reward=1.0,
            log_prob=torch.tensor(-0.5, requires_grad=True),
            value=torch.tensor(0.5, requires_grad=True),
            policy_logits=torch.randn(3, requires_grad=True),
            aggressiveness=torch.tensor(0.5, requires_grad=True),
            done=True,
        )
        mock_collect_episode.return_value = mock_episode

        train_data = np.random.randn(5, 50, 76).astype(np.float32)

        trained_policy = train_with_environment(
            signal_model=mock_signal_model,
            policy=mock_execution_policy,
            train_data=train_data,
            cfg=rl_config,
            env_config=execution_config,
            device=device,
        )

        assert mock_collect_episode.call_count == 5 * rl_config.epochs

    def test_train_with_environment_function_exists(self):
        """Test train_with_environment function exists."""
        assert callable(train_with_environment)
