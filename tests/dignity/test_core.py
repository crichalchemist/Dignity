"""Test core utilities."""

import pytest
import numpy as np
import pandas as pd
from dignity.core.signals import SignalProcessor
from dignity.core.privacy import PrivacyManager
from dignity.core.config import DignityConfig


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
        vol = np.concatenate([
            np.ones(100) * 0.5,   # Low vol
            np.ones(100) * 2.0,   # High vol
            np.ones(100) * 1.0    # Normal vol
        ])
        
        regimes = SignalProcessor.regime_detection(vol)
        
        assert len(regimes) == len(vol)
        assert np.any(regimes == 0)  # Low vol regime
        assert np.any(regimes == 2)  # High vol regime


class TestPrivacyManager:
    """Test privacy-preserving operations."""
    
    def test_hash_identifier(self):
        """Test identifier hashing."""
        addr1 = "0x1234567890abcdef"
        addr2 = "0x1234567890abcdef"
        addr3 = "0xfedcba0987654321"
        
        hash1 = PrivacyManager.hash_identifier(addr1)
        hash2 = PrivacyManager.hash_identifier(addr2)
        hash3 = PrivacyManager.hash_identifier(addr3)
        
        # Same input = same hash
        assert hash1 == hash2
        # Different input = different hash
        assert hash1 != hash3
        # Hash is hex string
        assert len(hash1) == 64
    
    def test_anonymize_addresses(self):
        """Test batch address anonymization."""
        addresses = ["addr1", "addr2", "addr3"]
        hashed = PrivacyManager.anonymize_addresses(addresses)
        
        assert len(hashed) == len(addresses)
        assert all(len(h) == 64 for h in hashed)
        assert len(set(hashed)) == len(addresses)  # All unique
    
    def test_quantize_amounts(self):
        """Test amount quantization."""
        amounts = np.random.uniform(10, 100, 1000)
        quantized = PrivacyManager.quantize_amounts(amounts, bins=10)
        
        # Should have fewer unique values
        assert len(np.unique(quantized)) <= 10
        # Values should be within original range
        assert np.min(quantized) >= np.min(amounts)
        assert np.max(quantized) <= np.max(amounts)
    
    def test_add_noise(self):
        """Test differential privacy noise."""
        values = np.array([100.0, 200.0, 300.0])
        noisy = PrivacyManager.add_noise(values, epsilon=1.0)
        
        # Should be different but similar
        assert not np.array_equal(values, noisy)
        assert np.allclose(values, noisy, atol=50)  # Reasonable noise


class TestDignityConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DignityConfig()
        
        assert config.model.task == 'risk'
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
