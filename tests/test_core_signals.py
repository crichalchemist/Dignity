# tests/test_core_signals.py
import pytest
import numpy as np
from dignity.core.signals import compute_entropy, compute_volatility

def test_entropy_calculation():
    """Entropy should be 0 for uniform distribution"""
    values = np.ones(100)
    entropy = compute_entropy(values, window=10)
    assert len(entropy) == 100
    assert np.allclose(entropy[-10:], 0.0)

def test_volatility_calculation():
    """Volatility should detect changes"""
    values = np.concatenate([np.ones(50), np.ones(50) * 2])
    volatility = compute_volatility(values, window=10)
    assert len(volatility) == 100
    assert volatility[55] > volatility[5]
