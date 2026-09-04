"""Test core utilities."""

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config import DignityConfig, PrivacyConfig
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


def _valid_privacy_block():
    return {
        "epsilon_total": 1.0,
        "k": 5,
        "key_env": None,
        "features": {
            "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 1000]},
            "price": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 500]},
            "fee_rate": {"mechanism": "generalize", "bins": 10},
        },
    }


class TestPrivacyConfig:
    """Misconfiguration fails at load time, and absence means no privacy stage."""

    def test_absent_block_yields_none(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"model": {"task": "risk"}}))
        assert DignityConfig.from_yaml(str(path)).privacy is None

    def test_valid_block_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"privacy": _valid_privacy_block()}))
        cfg = DignityConfig.from_yaml(str(path))
        assert cfg.privacy is not None
        assert cfg.privacy.features["volume"].bounds == (0.0, 1000.0)

        out = tmp_path / "out.yaml"
        cfg.to_yaml(str(out))
        again = DignityConfig.from_yaml(str(out))
        assert again.privacy.to_dict() == cfg.privacy.to_dict()

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (
                lambda b: b["features"]["volume"].update(mechanism="gaussian"),
                "unknown mechanism",
            ),
            (lambda b: b["features"]["volume"].update(epsilon=0.0), "epsilon > 0"),
            (lambda b: b["features"]["volume"].pop("bounds"), "requires bounds"),
            (lambda b: b["features"]["volume"].update(bounds=[10, 10]), "lo < hi"),
            (lambda b: b["features"]["fee_rate"].update(epsilon=0.1), "bins only"),
            (lambda b: b["features"]["fee_rate"].update(bounds=[0, 1]), "bins only"),
            (lambda b: b.update(k=1), "k must be"),
            (lambda b: b["features"]["fee_rate"].update(bins=1), "bins must be"),
            (
                lambda b: b["features"]["volume"].update(epsilon=0.9),
                "exceeds epsilon_total",
            ),
        ],
    )
    def test_rejects_bad_privacy_block(self, mutate, match):
        block = _valid_privacy_block()
        mutate(block)
        with pytest.raises(ValueError, match=match):
            PrivacyConfig(**block)
