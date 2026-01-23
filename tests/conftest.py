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


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)
