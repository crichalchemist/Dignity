"""Test model components."""

import pytest
import torch
from models.backbone.cnn1d import CNN1D
from models.backbone.lstm import StackedLSTM
from models.backbone.attention import AdditiveAttention
from models.backbone.hybrid import DignityBackbone
from models.head.risk import RiskHead
from models.head.forecast import ForecastHead
from models.head.policy import PolicyHead
from models.dignity import Dignity


class TestBackboneComponents:
    """Test individual backbone components."""
    
    def test_cnn1d(self):
        """Test 1D-CNN component."""
        model = CNN1D(input_size=9, hidden_size=64, kernel_size=3)
        x = torch.randn(4, 100, 9)  # [B, T, F]
        
        out = model(x)
        
        assert out.shape == (4, 100, 64)  # [B, T, H]
    
    def test_stacked_lstm(self):
        """Test LSTM component."""
        model = StackedLSTM(input_size=64, hidden_size=128, num_layers=2)
        x = torch.randn(4, 100, 64)
        
        out, (h_n, c_n) = model(x)
        
        assert out.shape == (4, 100, 128)  # [B, T, H]
        assert h_n.shape == (2, 4, 128)  # [L, B, H]
        assert c_n.shape == (2, 4, 128)
    
    def test_additive_attention(self):
        """Test attention mechanism."""
        model = AdditiveAttention(hidden_size=128)
        x = torch.randn(4, 100, 128)
        
        context, weights = model(x)
        
        assert context.shape == (4, 128)  # [B, H]
        assert weights.shape == (4, 100)  # [B, T]
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_dignity_backbone(self):
        """Test full backbone."""
        model = DignityBackbone(
            input_size=9,
            hidden_size=256,
            n_layers=2
        )
        x = torch.randn(4, 100, 9)
        
        context, attn_weights = model(x)
        
        assert context.shape == (4, 256)
        assert attn_weights.shape == (4, 100)


class TestTaskHeads:
    """Test task-specific heads."""
    
    def test_risk_head(self):
        """Test risk scoring head."""
        head = RiskHead(input_size=256, hidden_size=128)
        context = torch.randn(4, 256)
        
        risk = head(context)
        
        assert risk.shape == (4, 1)
        assert torch.all(risk >= 0) and torch.all(risk <= 1)  # Sigmoid output
    
    def test_forecast_head(self):
        """Test forecasting head."""
        head = ForecastHead(
            input_size=256,
            pred_len=5,
            num_features=3
        )
        context = torch.randn(4, 256)
        
        predictions = head(context)
        
        assert predictions.shape == (4, 5, 3)  # [B, pred_len, features]
    
    def test_policy_head(self):
        """Test policy head."""
        head = PolicyHead(input_size=256, n_actions=3)
        context = torch.randn(4, 256)
        
        logits, value = head(context)
        
        assert logits.shape == (4, 3)  # [B, actions]
        assert value.shape == (4, 1)  # [B, 1]
        
        # Test action sampling
        action, log_prob, value = head.sample_action(context)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)


class TestDignityModel:
    """Test complete Dignity model."""
    
    def test_risk_model(self):
        """Test risk scoring model."""
        model = Dignity(
            task='risk',
            input_size=9,
            hidden_size=128,
            n_layers=2
        )
        x = torch.randn(4, 100, 9)
        
        predictions, attn = model(x)
        
        assert predictions.shape == (4, 1)
        assert attn.shape == (4, 100)
    
    def test_forecast_model(self):
        """Test forecasting model."""
        model = Dignity(
            task='forecast',
            input_size=9,
            hidden_size=128,
            n_layers=2,
            pred_len=5,
            num_features=3
        )
        x = torch.randn(4, 100, 9)
        
        predictions, attn = model(x)
        
        assert predictions.shape == (4, 5, 3)
    
    def test_model_summary(self):
        """Test model summary."""
        model = Dignity(task='risk', input_size=9, hidden_size=128)
        summary = model.summary()
        
        assert 'Dignity' in summary
        assert '128' in summary  # hidden size
        # Parameter count is formatted with commas, so just check it exists
        assert 'Total parameters' in summary
    
    def test_predict(self):
        """Test inference mode."""
        model = Dignity(task='risk', input_size=9, hidden_size=64)
        x = torch.randn(2, 50, 9)
        
        predictions = model.predict(x)
        
        assert predictions.shape == (2, 1)
        assert not model.training  # Should be in eval mode
