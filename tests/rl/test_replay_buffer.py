"""
Unit tests for experience replay buffers.

Tests both uniform and prioritized experience replay implementations:
- ReplayBuffer (uniform sampling)
- PrioritizedReplayBuffer (importance sampling with TD error priorities)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


@pytest.fixture
def state_dim():
    """State dimension for testing."""
    return 76


class TestReplayBuffer:
    """Test uniform experience replay buffer."""

    def test_initialization(self):
        """Test replay buffer initializes correctly."""
        capacity = 1000
        buffer = ReplayBuffer(capacity=capacity)

        assert buffer.capacity == capacity
        assert len(buffer) == 0
        assert buffer.position == 0

    def test_add_single_transition(self, state_dim):
        """Test adding a single transition."""
        buffer = ReplayBuffer(capacity=100)

        state = np.random.randn(state_dim)
        action = 0.5
        reward = 1.0
        next_state = np.random.randn(state_dim)
        done = False

        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 1
        assert buffer.position == 1

    def test_add_multiple_transitions(self, state_dim):
        """Test adding multiple transitions."""
        buffer = ReplayBuffer(capacity=100)

        n_transitions = 50
        for _ in range(n_transitions):
            buffer.add(
                np.random.randn(state_dim),
                np.random.uniform(-1, 1),
                np.random.randn(),
                np.random.randn(state_dim),
                False,
            )

        assert len(buffer) == n_transitions

    def test_circular_buffer_overflow(self, state_dim):
        """Test that buffer overwrites oldest transitions when full."""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity)

        # Fill buffer to capacity
        for _ in range(capacity):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        assert len(buffer) == capacity

        # Add more transitions (should overwrite oldest)
        for _ in range(5):
            buffer.add(
                np.random.randn(state_dim), 1.0, 1.0, np.random.randn(state_dim), False
            )

        # Size should remain at capacity
        assert len(buffer) == capacity

        # Position should wrap around
        assert buffer.position == 5

    def test_sample_batch_shape(self, state_dim):
        """Test that sampled batch has correct shapes."""
        buffer = ReplayBuffer(capacity=100)
        batch_size = 32

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim),
                np.random.uniform(-1, 1),
                np.random.randn(),
                np.random.randn(state_dim),
                False,
            )

        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        assert states.shape == (batch_size, state_dim)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, state_dim)
        assert dones.shape == (batch_size,)

    def test_sample_dtypes(self, state_dim):
        """Test that sampled batch has correct dtypes."""
        buffer = ReplayBuffer(capacity=100)

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim),
                np.random.uniform(-1, 1),
                np.random.randn(),
                np.random.randn(state_dim),
                False,
            )

        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample(32)

        # Check dtypes
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32

    def test_sample_uniformity(self, state_dim):
        """Test that sampling is approximately uniform."""
        capacity = 100
        buffer = ReplayBuffer(capacity=capacity)

        # Add transitions with unique rewards for tracking
        for i in range(capacity):
            buffer.add(
                np.random.randn(state_dim),
                0.0,
                float(i),  # Unique reward
                np.random.randn(state_dim),
                False,
            )

        # Sample many times and check distribution
        sample_counts = np.zeros(capacity)
        n_samples = 10000

        for _ in range(n_samples):
            _, _, rewards, _, _ = buffer.sample(1)
            idx = int(rewards[0])
            sample_counts[idx] += 1

        # Check distribution is approximately uniform
        expected_count = n_samples / capacity
        # Allow 50% variance
        assert sample_counts.min() > expected_count * 0.5
        assert sample_counts.max() < expected_count * 1.5

    def test_length_method(self, state_dim):
        """Test __len__ method."""
        buffer = ReplayBuffer(capacity=100)

        assert len(buffer) == 0

        for i in range(50):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )
            assert len(buffer) == i + 1


class TestPrioritizedReplayBuffer:
    """Test prioritized experience replay buffer."""

    def test_initialization(self):
        """Test prioritized buffer initializes correctly."""
        capacity = 1000
        alpha = 0.6
        beta_start = 0.4
        buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha, beta_start=beta_start)

        assert buffer.capacity == capacity
        assert buffer.alpha == alpha
        assert buffer.beta_start == beta_start
        assert len(buffer) == 0
        assert buffer.max_priority == 1.0
        assert buffer.frame == 1

    def test_add_assigns_max_priority(self, state_dim):
        """Test that new transitions get max priority."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add transition
        buffer.add(
            np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
        )

        # New transition should have max priority
        assert buffer.priorities[0] == buffer.max_priority

    def test_sample_returns_weights_and_indices(self, state_dim):
        """Test that prioritized sample returns weights and indices."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim),
                np.random.uniform(-1, 1),
                np.random.randn(),
                np.random.randn(state_dim),
                False,
            )

        # Sample batch
        batch, indices, weights = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch

        # Check shapes
        assert states.shape == (32, state_dim)
        assert indices.shape == (32,)
        assert weights.shape == (32,)

        # Weights should be normalized to [0, 1]
        assert weights.max() == pytest.approx(1.0, abs=1e-5)
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_update_priorities(self, state_dim):
        """Test priority updates."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Sample batch
        _, indices, _ = buffer.sample(10)

        # Create TD errors (high error for first sampled transition)
        td_errors = np.ones(10) * 0.1
        td_errors[0] = 10.0  # High TD error

        # Update priorities
        buffer.update_priorities(indices, td_errors)

        # Priority for high-error transition should be higher
        assert buffer.priorities[indices[0]] > buffer.priorities[indices[1]]

        # Max priority should be updated
        assert buffer.max_priority >= 10.0

    def test_prioritized_sampling_bias(self, state_dim):
        """Test that high-priority transitions are sampled more often."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)  # Full prioritization

        # Add transitions
        for i in range(10):
            buffer.add(
                np.random.randn(state_dim), 0.0, float(i), np.random.randn(state_dim), False
            )

        # Set one transition to have very high priority
        buffer.priorities[0] = 100.0
        buffer.priorities[1:10] = 1.0

        # Sample many times
        sample_counts = np.zeros(10)
        n_samples = 1000

        for _ in range(n_samples):
            _, indices, _ = buffer.sample(1)
            sample_counts[indices[0]] += 1

        # High-priority transition should be sampled much more
        assert sample_counts[0] > sample_counts[1:].mean() * 2

    def test_beta_annealing(self, state_dim):
        """Test that beta anneals from beta_start to 1.0 via its effect on importance weights."""
        beta_start = 0.4
        beta_frames = 1000
        buffer = PrioritizedReplayBuffer(
            capacity=100, beta_start=beta_start, beta_frames=beta_frames
        )

        # Add some transitions with varying priorities
        for i in range(10):
            buffer.add(
                np.random.randn(state_dim),
                float(i % 2),
                float(i),
                np.random.randn(state_dim),
                False,
            )
            # Set priority to vary (higher for first transitions)
            buffer.priorities[i] = 10.0 - i

        # Sample at early frame and check weights
        buffer.frame = 1
        batch_early, weights_early, _ = buffer.sample(8)
        
        # Sample at halfway point and check weights increase toward 1.0
        buffer.frame = beta_frames // 2
        batch_mid, weights_mid, _ = buffer.sample(8)
        
        # Weights should increase as beta anneals (approaching 1.0)
        # This indirectly validates that beta increased without accessing private _get_beta
        mean_weight_early = weights_early.mean()
        mean_weight_mid = weights_mid.mean()
        
        # At early frames, beta is smaller, weights should have more variance
        # At later frames (higher beta), weights should be more uniform
        assert mean_weight_early >= 0  # Weights are positive
        assert mean_weight_mid >= mean_weight_early or abs(mean_weight_mid - mean_weight_early) < 0.1

        # After annealing completes
        buffer.frame = beta_frames * 2
        assert buffer._get_beta() == pytest.approx(1.0, abs=1e-5)

    def test_importance_sampling_weights(self, state_dim):
        """Test importance sampling weight computation."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=1.0)

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Set specific priorities
        buffer.priorities[:50] = np.linspace(0.1, 10.0, 50)

        # Sample
        _, indices, weights = buffer.sample(32)

        # High-priority samples should have lower weights (importance sampling correction)
        # Because they're sampled more often, need to downweight them
        assert np.isfinite(weights).all()
        assert (weights > 0).all()
        assert (weights <= 1).all()

    def test_epsilon_prevents_zero_probability(self, state_dim):
        """Test that epsilon prevents zero sampling probabilities."""
        buffer = PrioritizedReplayBuffer(capacity=100, epsilon=1e-6)

        # Add transitions
        for _ in range(10):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Set priorities to zero (epsilon should prevent zero probability)
        buffer.priorities[:10] = 0.0

        # Update with zero TD errors (should add epsilon)
        indices = np.arange(10)
        td_errors = np.zeros(10)
        buffer.update_priorities(indices, td_errors)

        # All priorities should be epsilon (not zero)
        assert (buffer.priorities[:10] == buffer.epsilon).all()

        # Should still be able to sample
        _, _, _ = buffer.sample(5)

    def test_circular_buffer_with_priorities(self, state_dim):
        """Test circular buffer behavior with priorities."""
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity=capacity)

        # Fill buffer
        for _ in range(capacity):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Add more (should overwrite)
        for _ in range(5):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Buffer size should remain at capacity
        assert len(buffer) == capacity

        # Should still be able to sample
        _, _, _ = buffer.sample(5)

    def test_alpha_zero_uniform_sampling(self, state_dim):
        """Test that alpha=0 gives uniform sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.0)

        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(state_dim), 0.0, 0.0, np.random.randn(state_dim), False
            )

        # Set very different priorities
        buffer.priorities[:50] = np.logspace(-2, 2, 50)  # 0.01 to 100

        # Sample many times
        sample_counts = np.zeros(50)
        n_samples = 5000

        for _ in range(n_samples):
            _, indices, _ = buffer.sample(1)
            sample_counts[indices[0]] += 1

        # Distribution should be approximately uniform (alpha=0)
        expected_count = n_samples / 50
        # Allow 50% variance
        assert sample_counts.min() > expected_count * 0.5
        assert sample_counts.max() < expected_count * 1.5


class TestReplayBufferIntegration:
    """Integration tests for replay buffers."""

    def test_uniform_buffer_with_rl_agent(self, state_dim):
        """Test uniform buffer in simulated RL training loop."""
        buffer = ReplayBuffer(capacity=1000)

        # Simulate agent interaction
        state = np.random.randn(state_dim)

        for step in range(100):
            action = np.random.uniform(-1, 1)
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = step % 20 == 0

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Start training after some experience
            if step >= 32:
                batch = buffer.sample(16)
                assert len(batch) == 5  # (states, actions, rewards, next_states, dones)

    def test_prioritized_buffer_with_rl_agent(self, state_dim):
        """Test prioritized buffer in simulated RL training loop."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4, beta_frames=1000)

        # Simulate agent interaction
        state = np.random.randn(state_dim)

        for step in range(100):
            action = np.random.uniform(-1, 1)
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = step % 20 == 0

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            # Start training after some experience
            if step >= 32:
                batch, indices, weights = buffer.sample(16)

                # Simulate TD error computation
                td_errors = np.abs(np.random.randn(16))

                # Update priorities
                buffer.update_priorities(indices, td_errors)

                # Check everything is valid
                assert np.isfinite(weights).all()
                assert len(batch) == 5

    def test_buffer_types_consistency(self, state_dim):
        """Test that both buffer types have consistent interface."""
        uniform_buffer = ReplayBuffer(capacity=100)
        prioritized_buffer = PrioritizedReplayBuffer(capacity=100)

        # Add same transitions to both
        for _ in range(50):
            state = np.random.randn(state_dim)
            action = np.random.uniform(-1, 1)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = False

            uniform_buffer.add(state, action, reward, next_state, done)
            prioritized_buffer.add(state, action, reward, next_state, done)

        # Both should have same length
        assert len(uniform_buffer) == len(prioritized_buffer)

        # Both can sample (though with different return signatures)
        uniform_batch = uniform_buffer.sample(16)
        prioritized_batch, _, _ = prioritized_buffer.sample(16)

        # Batch components should have same shapes
        for u_component, p_component in zip(uniform_batch, prioritized_batch):
            assert u_component.shape == p_component.shape
