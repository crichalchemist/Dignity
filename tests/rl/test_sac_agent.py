"""
Unit tests for SAC (Soft Actor-Critic) agent implementation.

Tests the SAC components including:
- QNetwork (twin critics for Q-value estimation)
- StochasticPolicy (actor with reparameterization trick)
- SACAgent (main training harness with automatic entropy tuning)
- Soft target network updates
- Save/load functionality
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl.agents.sac_agent import SACAgent
from rl.networks.sac_networks import QNetwork, StochasticPolicy
from rl.replay_buffer import ReplayBuffer


@pytest.fixture
def state_dim():
    """State dimension for testing."""
    return 76


@pytest.fixture
def action_dim():
    """Action dimension for testing."""
    return 1


@pytest.fixture
def hidden_dim():
    """Hidden layer dimension for testing."""
    return 128  # Smaller for faster tests


@pytest.fixture
def sample_state(state_dim):
    """Sample state for testing."""
    return torch.randn(32, state_dim)


@pytest.fixture
def sample_action(action_dim):
    """Sample action for testing."""
    return torch.randn(32, action_dim)


@pytest.fixture
def replay_buffer_with_data(state_dim, action_dim):
    """Pre-filled replay buffer for testing."""
    buffer = ReplayBuffer(capacity=1000)

    # Add 500 random transitions
    for i in range(500):
        state = np.random.randn(state_dim)
        # Generate action with proper shape (don't slice if action_dim > 1)
        action = np.random.uniform(-1, 1, size=action_dim)
        if action_dim == 1:
            action = action[0]  # Scalar for action_dim=1
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        # Vary done: ~10% of transitions are terminal
        done = i % 10 == 0

        buffer.add(state, action, reward, next_state, done)

    return buffer


class TestQNetwork:
    """Test Q-network (critic) for SAC."""

    def test_initialization(self, state_dim, action_dim, hidden_dim):
        """Test Q-network initializes correctly."""
        q_net = QNetwork(state_dim, action_dim, hidden_dim)

        assert isinstance(q_net.net, nn.Sequential)

        # Check input/output dimensions
        first_layer = q_net.net[0]
        assert first_layer.in_features == state_dim + action_dim

    def test_forward_pass(self, state_dim, action_dim, hidden_dim):
        """Test Q-network forward pass."""
        q_net = QNetwork(state_dim, action_dim, hidden_dim)

        batch_size = 32
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)

        q_values = q_net(states, actions)

        # Check output shape
        assert q_values.shape == (batch_size, 1)

        # Q-values should be finite
        assert torch.isfinite(q_values).all()

    def test_gradient_flow(self, state_dim, action_dim, hidden_dim):
        """Test gradients flow through Q-network."""
        q_net = QNetwork(state_dim, action_dim, hidden_dim)

        states = torch.randn(8, state_dim, requires_grad=True)
        actions = torch.randn(8, action_dim, requires_grad=True)

        q_values = q_net(states, actions)
        loss = q_values.mean()
        loss.backward()

        # Check gradients exist
        assert states.grad is not None
        assert actions.grad is not None

        # Check network parameters have gradients
        for param in q_net.parameters():
            assert param.grad is not None


class TestStochasticPolicy:
    """Test stochastic policy (actor) for SAC."""

    def test_initialization(self, state_dim, action_dim, hidden_dim):
        """Test policy network initializes correctly."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        assert isinstance(policy.shared, nn.Sequential)
        assert isinstance(policy.mean_head, nn.Linear)
        assert isinstance(policy.log_std_head, nn.Linear)
        assert policy.action_dim == action_dim

    def test_stochastic_forward(self, state_dim, action_dim, hidden_dim):
        """Test stochastic action sampling."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        batch_size = 32
        states = torch.randn(batch_size, state_dim)

        actions, log_probs = policy(states, deterministic=False)

        # Check shapes
        assert actions.shape == (batch_size, action_dim)
        assert log_probs.shape == (batch_size,)

        # Actions should be bounded to [-1, 1] by tanh
        assert (actions >= -1).all()
        assert (actions <= 1).all()

        # Log probs can have small positive values due to tanh squashing and Jacobian correction
        # Allow small positive tolerance
        assert (log_probs <= 1e-3).all()

        # Should be finite
        assert torch.isfinite(actions).all()
        assert torch.isfinite(log_probs).all()

    def test_deterministic_forward(self, state_dim, action_dim, hidden_dim):
        """Test deterministic action selection."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        batch_size = 32
        states = torch.randn(batch_size, state_dim)

        actions, log_probs = policy(states, deterministic=True)

        # Check shapes
        assert actions.shape == (batch_size, action_dim)
        assert log_probs is None or (log_probs == 0).all()

        # Actions should be bounded
        assert (actions >= -1).all()
        assert (actions <= 1).all()

    def test_stochastic_sampling_varies(self, state_dim, action_dim, hidden_dim):
        """Test that stochastic sampling produces different actions."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        state = torch.randn(1, state_dim)

        # Sample multiple actions
        actions = []
        for _ in range(10):
            action, _ = policy(state, deterministic=False)
            actions.append(action.item())

        # Actions should vary (stochastic policy)
        unique_actions = len(set(actions))
        assert unique_actions > 1, "Stochastic policy should produce varying actions"

    def test_deterministic_sampling_consistent(self, state_dim, action_dim, hidden_dim):
        """Test that deterministic sampling is consistent."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        state = torch.randn(1, state_dim)

        # Sample multiple times
        actions = []
        for _ in range(5):
            action, _ = policy(state, deterministic=True)
            actions.append(action.detach().clone())

        # All actions should be identical
        for i in range(1, len(actions)):
            assert torch.allclose(actions[0], actions[i])

    def test_get_action_method(self, state_dim, action_dim, hidden_dim):
        """Test get_action convenience method."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        states = torch.randn(16, state_dim)

        # Stochastic
        action_stoch = policy.get_action(states, deterministic=False)
        assert action_stoch.shape == (16, action_dim)

        # Deterministic
        action_det = policy.get_action(states, deterministic=True)
        assert action_det.shape == (16, action_dim)

    def test_reparameterization_trick(self, state_dim, action_dim, hidden_dim):
        """Test that reparameterization trick enables gradient flow."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        states = torch.randn(8, state_dim, requires_grad=True)

        actions, log_probs = policy(states, deterministic=False)

        # Backprop through stochastic sampling
        loss = actions.mean()
        loss.backward()

        # Gradients should exist (reparameterization trick)
        assert states.grad is not None
        assert states.grad.norm().item() > 0

    def test_log_prob_correction(self, state_dim, action_dim, hidden_dim):
        """Test that log prob includes tanh correction."""
        policy = StochasticPolicy(state_dim, action_dim, hidden_dim)

        states = torch.randn(100, state_dim)

        # Sample many times
        log_probs = []
        for _ in range(10):
            _, log_prob = policy(states, deterministic=False)
            log_probs.append(log_prob)

        log_probs_tensor = torch.stack(log_probs)

        # Log probs should be finite
        assert torch.isfinite(log_probs_tensor).all()

        # Should have reasonable range
        assert log_probs_tensor.mean() < 0  # Negative (log of probability)


class TestSACAgent:
    """Test SAC agent main harness."""

    def test_initialization_default(self, state_dim, action_dim, hidden_dim):
        """Test SAC agent initializes with default settings."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        assert agent.gamma == 0.99  # Default
        assert agent.tau == 0.005  # Default
        assert agent.auto_entropy_tuning is True

        # Check networks exist
        assert isinstance(agent.policy, StochasticPolicy)
        assert isinstance(agent.q1, QNetwork)
        assert isinstance(agent.q2, QNetwork)
        assert isinstance(agent.q1_target, QNetwork)
        assert isinstance(agent.q2_target, QNetwork)

        # Target networks should have requires_grad=False
        for param in agent.q1_target.parameters():
            assert not param.requires_grad
        for param in agent.q2_target.parameters():
            assert not param.requires_grad

    def test_initialization_manual_alpha(self, state_dim, action_dim, hidden_dim):
        """Test SAC agent with manual alpha (no auto-tuning)."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            alpha=0.5,
            auto_entropy_tuning=False,
        )

        assert agent.auto_entropy_tuning is False
        assert agent.alpha == 0.5
        assert agent.log_alpha is None

    def test_initialization_auto_alpha(self, state_dim, action_dim, hidden_dim):
        """Test SAC agent with automatic entropy tuning."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        assert agent.auto_entropy_tuning is True
        assert agent.target_entropy == -action_dim  # Default heuristic
        assert agent.log_alpha is not None
        assert agent.log_alpha.requires_grad

    def test_select_action_stochastic(self, state_dim, action_dim, hidden_dim):
        """Test stochastic action selection."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        state = np.random.randn(state_dim)

        action = agent.select_action(state, evaluate=False)

        assert action.shape == (action_dim,)
        assert np.all(action >= -1)
        assert np.all(action <= 1)
        assert np.isfinite(action).all()

    def test_select_action_deterministic(self, state_dim, action_dim, hidden_dim):
        """Test deterministic action selection (for evaluation)."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        state = np.random.randn(state_dim)

        # Sample same state multiple times
        actions = []
        for _ in range(3):
            action = agent.select_action(state, evaluate=True)
            actions.append(action)

        # Deterministic actions should be identical
        for i in range(1, len(actions)):
            assert np.allclose(actions[0], actions[i])

    def test_update_returns_metrics(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test that update returns training metrics."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        metrics = agent.update(replay_buffer_with_data, batch_size=64)

        # Check metrics exist
        assert "q1_loss" in metrics
        assert "q2_loss" in metrics
        assert "policy_loss" in metrics
        assert "alpha" in metrics
        assert "q1_mean" in metrics
        assert "q2_mean" in metrics
        assert "log_prob_mean" in metrics

        # With auto-tuning
        if agent.auto_entropy_tuning:
            assert "alpha_loss" in metrics

        # All metrics should be finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_update_modifies_networks(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test that update modifies network parameters."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        # Get initial parameters
        initial_policy_param = next(agent.policy.parameters()).clone().detach()
        initial_q1_param = next(agent.q1.parameters()).clone().detach()

        # Perform update
        agent.update(replay_buffer_with_data, batch_size=64)

        # Check parameters changed
        updated_policy_param = next(agent.policy.parameters())
        updated_q1_param = next(agent.q1.parameters())

        assert not torch.allclose(initial_policy_param, updated_policy_param)
        assert not torch.allclose(initial_q1_param, updated_q1_param)

    def test_update_increments_counter(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test that update counter increments."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        assert agent.update_count == 0

        agent.update(replay_buffer_with_data, batch_size=64)
        assert agent.update_count == 1

        agent.update(replay_buffer_with_data, batch_size=64)
        assert agent.update_count == 2

    def test_soft_update_target_networks(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test soft update of target networks."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            tau=0.1,  # Larger tau for easier testing
        )

        # Get initial target parameters
        initial_target_param = next(agent.q1_target.parameters()).clone().detach()

        # Perform update
        agent.update(replay_buffer_with_data, batch_size=64)

        # Target should have changed (soft update)
        updated_target_param = next(agent.q1_target.parameters())
        assert not torch.allclose(initial_target_param, updated_target_param)

        # But not as much as source network
        source_param = next(agent.q1.parameters())
        # Target should be closer to initial than to source (tau=0.1)
        dist_to_initial = (updated_target_param - initial_target_param).norm()
        dist_to_source = (updated_target_param - source_param).norm()
        # With tau=0.1, target moves 10% toward source
        assert dist_to_source > dist_to_initial * 0.5  # Rough check

    def test_alpha_auto_tuning(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test automatic entropy temperature tuning."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        initial_alpha = agent.alpha

        # Perform multiple updates
        for _ in range(10):
            agent.update(replay_buffer_with_data, batch_size=64)

        # Alpha should have changed (auto-tuning) within tolerance
        # Use absolute/relative tolerance for float comparison
        import math
        assert not math.isclose(agent.alpha, initial_alpha, abs_tol=1e-6, rel_tol=1e-6)

        # Alpha should remain positive
        assert agent.alpha > 0

    def test_save_and_load(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test save and load functionality."""
        agent1 = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        # Train agent a bit
        for _ in range(5):
            agent1.update(replay_buffer_with_data, batch_size=64)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            # Save agent
            agent1.save(checkpoint_path)

            # Create new agent and load
            agent2 = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                auto_entropy_tuning=True,
            )

            agent2.load(checkpoint_path)

            # Check state matches
            assert agent2.update_count == agent1.update_count
            assert abs(agent2.alpha - agent1.alpha) < 1e-6

            # Check actions match (deterministic)
            test_state = np.random.randn(state_dim)
            action1 = agent1.select_action(test_state, evaluate=True)
            action2 = agent2.select_action(test_state, evaluate=True)

            assert np.allclose(action1, action2, atol=1e-5)

        finally:
            os.remove(checkpoint_path)

    def test_manual_soft_update(self, state_dim, action_dim, hidden_dim):
        """Test _soft_update method directly."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            tau=0.5,  # 50% update for testing
        )

        # Modify source network
        source_param = next(agent.q1.parameters())
        source_param.data.fill_(1.0)

        # Initialize target differently
        target_param = next(agent.q1_target.parameters())
        target_param.data.fill_(0.0)

        # Soft update
        agent._soft_update(agent.q1, agent.q1_target)

        # Check target is now 50% of source (tau=0.5)
        expected_value = 0.5 * 1.0 + 0.5 * 0.0
        # Use dtype matching when creating expected tensor
        assert torch.allclose(
            target_param, 
            torch.tensor(expected_value, dtype=target_param.dtype), 
            atol=1e-5
        )


class TestSACIntegration:
    """Integration tests for SAC agent training."""

    def test_multiple_updates_stable(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test that multiple updates are stable (no NaN/Inf)."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        losses = []

        for _ in range(20):
            metrics = agent.update(replay_buffer_with_data, batch_size=64)
            loss = metrics["q1_loss"] + metrics["q2_loss"] + metrics["policy_loss"]
            losses.append(loss)

        # All losses should be finite
        assert all(np.isfinite(loss) for loss in losses)

    def test_action_selection_consistency(self, state_dim, action_dim, hidden_dim, replay_buffer_with_data):
        """Test action selection before and after update."""
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        state = np.random.randn(state_dim)

        # Action before update
        action_before = agent.select_action(state, evaluate=True)

        # Use pre-filled replay buffer fixture for update
        # Update multiple times to ensure policy changes
        for _ in range(3):
            agent.update(replay_buffer_with_data, batch_size=64)

        # Action after update
        action_after = agent.select_action(state, evaluate=True)

        # Actions should be different (policy updated) with relaxed tolerance
        assert not np.allclose(action_before, action_after, atol=1e-4, rtol=1e-4)

        # But both should be valid
        assert np.all(action_before >= -1) and np.all(action_before <= 1)
        assert np.all(action_after >= -1) and np.all(action_after <= 1)

    def test_full_training_loop(self, state_dim, action_dim, hidden_dim):
        """Test a minimal training loop."""
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            auto_entropy_tuning=True,
        )

        buffer = ReplayBuffer(capacity=1000)

        # Collect experience
        state = np.random.randn(state_dim)
        for step in range(200):
            action = agent.select_action(state, evaluate=False)
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = False

            buffer.add(state, action[0], reward, next_state, done)
            state = next_state

            # Start training after some experience
            if step >= 64:
                metrics = agent.update(buffer, batch_size=32)

                # Check metrics are reasonable
                assert np.isfinite(metrics["q1_loss"])
                assert np.isfinite(metrics["policy_loss"])
                assert metrics["alpha"] > 0

        # Agent should have trained
        assert agent.update_count > 0
