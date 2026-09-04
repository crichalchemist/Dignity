"""Test core utilities."""

import numpy as np
import pandas as pd
import pytest

from core.config import DignityConfig
from core.signals import SignalProcessor


class TestSignalProcessor:
    """Test signal processing functions."""

    def test_volatility(self):
        """Test volatility calculation."""
        values = np.array([100, 102, 98, 101, 99, 103])
        vol = SignalProcessor.volatility(values, window=3)

        assert len(vol) == len(values)
        assert vol[0] == vol[1] == vol[2]  # First values filled
        assert vol[-1] > 0  # Should have volatility

    def test_entropy(self):
        """Test entropy calculation."""
        # Uniform distribution should have high entropy
        uniform = np.random.uniform(0, 100, 1000)
        uniform_entropy = SignalProcessor.entropy(uniform)

        # Constant values should have zero entropy
        constant = np.ones(1000)
        constant_entropy = SignalProcessor.entropy(constant)

        assert uniform_entropy > constant_entropy
        assert constant_entropy == 0.0

    def test_price_momentum(self):
        """Test price momentum calculation."""
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        momentum = SignalProcessor.price_momentum(prices, window=2)

        assert len(momentum) == len(prices)
        # Check that momentum is computed (non-zero after window)
        assert np.any(momentum != 0)

    def test_directional_change(self):
        """Test directional change detection."""
        # Upward trend
        prices = np.array([100, 102, 105, 103, 106])
        dc = SignalProcessor.directional_change(prices, threshold=0.015)

        assert len(dc) == len(prices)
        assert np.sum(dc == 1) > 0  # Should have upward changes

    def test_regime_detection(self):
        """Test regime detection."""
        # Create volatility with different regimes
        vol = np.concatenate(
            [
                np.ones(100) * 0.5,  # Low vol
                np.ones(100) * 2.0,  # High vol
                np.ones(100) * 1.0,  # Normal vol
            ]
        )

        regimes = SignalProcessor.regime_detection(vol)

        assert len(regimes) == len(vol)
        assert np.any(regimes == 0)  # Low vol regime
        assert np.any(regimes == 2)  # High vol regime


class TestDignityConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = DignityConfig()

        assert config.model.task == "risk"
        assert config.model.hidden_size == 256
        assert config.data.seq_len == 100
        assert config.train.epochs == 50

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test save/load config."""
        config1 = DignityConfig()
        config1.model.hidden_size = 512

        yaml_path = tmp_path / "test_config.yaml"
        config1.to_yaml(str(yaml_path))

        config2 = DignityConfig.from_yaml(str(yaml_path))

        assert config2.model.hidden_size == 512
        assert config2.data.seq_len == config1.data.seq_len
