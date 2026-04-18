"""Pytest configuration and fixtures for Dignity tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_sequence():
    """Sample transaction sequence for testing."""
    return torch.randn(4, 100, 9)  # [B, T, F]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return torch.randint(0, 2, (4,))  # Binary labels


@pytest.fixture
def sample_sequence_32():
    """32-feature quant input matching the cascade model's input_size."""
    return torch.randn(4, 100, 32)  # [B=4, T=100, F=32]


@pytest.fixture
def cascade_labels():
    """Guided Learning label dict — one entry per cascade head.

    Matches the label schema expected by Dignity.cascade_loss():
        regime  [B]    int64  — volatility quantile 0-3
        var     [B, 1] float  — realized drawdown fraction in [0, 1]
        alpha   [B, 1] float  — n-step future return in [-1, 1]
        action  [B]    int64  — RL action index (0=HOLD, 1=BUY, 2=SELL)
    """
    return {
        "regime": torch.randint(0, 4, (4,)),
        "var": torch.rand(4, 1),
        "alpha": torch.randn(4, 1).clamp(-1, 1),
        "action": torch.randint(0, 3, (4,)),
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
