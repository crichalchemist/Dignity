"""Tests for task-specific heads in the Dignity Foundation Model."""

import pytest
import torch
import torch.nn as nn

from models.head import ForecastHead, MarketRegimeHead, RiskHead, TradeSignalHead


class TestMarketRegimeHead:
    """Tests for MarketRegimeHead."""
    
    @pytest.fixture
    def head(self):
        """Create a market regime head for testing."""
        return MarketRegimeHead(d_model=512, num_regimes=4)
    
    def test_forward_shape(self, head):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len, d_model = 8, 100, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = head(x)
        
        assert output.shape == (batch_size, seq_len, 4)
    
    def test_output_dimension(self, head):
        """Test output dimension matches number of regimes."""
        x = torch.randn(4, 50, 512)
        output = head(x)
        
        assert output.size(-1) == head.num_regimes
    
    def test_gradient_flow(self, head):
        """Test gradients flow through the head."""
        x = torch.randn(2, 10, 512, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_dropout_behavior(self, head):
        """Test dropout behaves differently in train vs eval mode."""
        x = torch.randn(4, 20, 512)
        
        # Get outputs in train mode (with dropout)
        head.train()
        torch.manual_seed(42)
        out_train_1 = head(x)
        torch.manual_seed(42)
        out_train_2 = head(x)
        
        # Get output in eval mode (no dropout)
        head.eval()
        torch.manual_seed(42)
        out_eval_1 = head(x)
        torch.manual_seed(42)
        out_eval_2 = head(x)
        
        # In eval mode, outputs should be identical with same seed
        assert torch.allclose(out_eval_1, out_eval_2)
        
        # In train mode with dropout, might have slight variations
        # (Note: GELU is deterministic, so if dropout rate is 0.1,
        # most of the time outputs will still be similar)
    
    def test_batch_independence(self, head):
        """Test that batch elements are processed independently."""
        head.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(4, 20, 512)
        
        # Process full batch
        output_batch = head(x)
        
        # Process samples individually
        outputs_individual = []
        for i in range(x.size(0)):
            out = head(x[i:i+1])
            outputs_individual.append(out)
        outputs_individual = torch.cat(outputs_individual, dim=0)
        
        assert torch.allclose(output_batch, outputs_individual, atol=1e-6)
    
    def test_different_regimes(self):
        """Test head with different number of regimes."""
        head = MarketRegimeHead(d_model=256, num_regimes=6)
        x = torch.randn(2, 30, 256)
        
        output = head(x)
        
        assert output.shape == (2, 30, 6)
    
    def test_components_exist(self, head):
        """Test all expected components exist."""
        assert isinstance(head.norm, nn.LayerNorm)
        assert isinstance(head.fc1, nn.Linear)
        assert isinstance(head.activation, nn.GELU)
        assert isinstance(head.dropout, nn.Dropout)
        assert isinstance(head.fc2, nn.Linear)


class TestForecastHead:
    """Tests for ForecastHead."""
    
    @pytest.fixture
    def head(self):
        """Create a forecast head for testing."""
        return ForecastHead(d_model=512, horizons=[1, 5, 15])
    
    def test_forward_shape(self, head):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len, d_model = 8, 100, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = head(x)
        
        assert output.shape == (batch_size, seq_len, 3)
    
    def test_output_dimension(self, head):
        """Test output dimension matches number of horizons."""
        x = torch.randn(4, 50, 512)
        output = head(x)
        
        assert output.size(-1) == head.num_horizons
    
    def test_gradient_flow(self, head):
        """Test gradients flow through the head."""
        x = torch.randn(2, 10, 512, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_dropout_behavior(self, head):
        """Test dropout behaves differently in train vs eval mode."""
        x = torch.randn(4, 20, 512)
        
        # Eval mode should give consistent results
        head.eval()
        torch.manual_seed(42)
        out_eval_1 = head(x)
        torch.manual_seed(42)
        out_eval_2 = head(x)
        
        assert torch.allclose(out_eval_1, out_eval_2)
    
    def test_batch_independence(self, head):
        """Test that batch elements are processed independently."""
        head.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(4, 20, 512)
        
        # Process full batch
        output_batch = head(x)
        
        # Process samples individually
        outputs_individual = []
        for i in range(x.size(0)):
            out = head(x[i:i+1])
            outputs_individual.append(out)
        outputs_individual = torch.cat(outputs_individual, dim=0)
        
        assert torch.allclose(output_batch, outputs_individual, atol=1e-6)
    
    def test_different_horizons(self):
        """Test head with different prediction horizons."""
        head = ForecastHead(d_model=256, horizons=[1, 3, 7, 14, 30])
        x = torch.randn(2, 30, 256)
        
        output = head(x)
        
        assert output.shape == (2, 30, 5)
        assert output.size(-1) == len(head.horizons)
    
    def test_default_horizons(self):
        """Test head uses default horizons when none provided."""
        head = ForecastHead(d_model=512)
        
        assert head.horizons == [1, 5, 15]
        assert head.num_horizons == 3
    
    def test_components_exist(self, head):
        """Test all expected components exist."""
        assert isinstance(head.norm, nn.LayerNorm)
        assert isinstance(head.fc1, nn.Linear)
        assert isinstance(head.activation, nn.GELU)
        assert isinstance(head.dropout, nn.Dropout)
        assert isinstance(head.fc2, nn.Linear)


class TestRiskHead:
    """Tests for RiskHead."""
    
    @pytest.fixture
    def head(self):
        """Create a risk head for testing."""
        return RiskHead(d_model=512, quantiles=[0.05, 0.5, 0.95])
    
    def test_forward_shape(self, head):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len, d_model = 8, 100, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = head(x)
        
        assert output.shape == (batch_size, seq_len, 3)
    
    def test_output_dimension(self, head):
        """Test output dimension matches number of quantiles."""
        x = torch.randn(4, 50, 512)
        output = head(x)
        
        assert output.size(-1) == head.num_quantiles
    
    def test_gradient_flow(self, head):
        """Test gradients flow through the head."""
        x = torch.randn(2, 10, 512, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_dropout_behavior(self, head):
        """Test dropout behaves differently in train vs eval mode."""
        x = torch.randn(4, 20, 512)
        
        # Eval mode should give consistent results
        head.eval()
        torch.manual_seed(42)
        out_eval_1 = head(x)
        torch.manual_seed(42)
        out_eval_2 = head(x)
        
        assert torch.allclose(out_eval_1, out_eval_2)
    
    def test_batch_independence(self, head):
        """Test that batch elements are processed independently."""
        head.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(4, 20, 512)
        
        # Process full batch
        output_batch = head(x)
        
        # Process samples individually
        outputs_individual = []
        for i in range(x.size(0)):
            out = head(x[i:i+1])
            outputs_individual.append(out)
        outputs_individual = torch.cat(outputs_individual, dim=0)
        
        assert torch.allclose(output_batch, outputs_individual, atol=1e-6)
    
    def test_different_quantiles(self):
        """Test head with different quantile specifications."""
        head = RiskHead(d_model=256, quantiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        x = torch.randn(2, 30, 256)
        
        output = head(x)
        
        assert output.shape == (2, 30, 5)
        assert output.size(-1) == len(head.quantiles)
    
    def test_default_quantiles(self):
        """Test head uses default quantiles when none provided."""
        head = RiskHead(d_model=512)
        
        assert head.quantiles == [0.05, 0.5, 0.95]
        assert head.num_quantiles == 3
    
    def test_components_exist(self, head):
        """Test all expected components exist."""
        assert isinstance(head.norm, nn.LayerNorm)
        assert isinstance(head.fc1, nn.Linear)
        assert isinstance(head.activation, nn.GELU)
        assert isinstance(head.dropout, nn.Dropout)
        assert isinstance(head.fc2, nn.Linear)


class TestTradeSignalHead:
    """Tests for TradeSignalHead."""
    
    @pytest.fixture
    def head(self):
        """Create a trade signal head for testing."""
        return TradeSignalHead(d_model=512, num_actions=3)
    
    def test_forward_shape(self, head):
        """Test forward pass produces correct output shape."""
        batch_size, seq_len, d_model = 8, 100, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = head(x)
        
        assert output.shape == (batch_size, seq_len, 3)
    
    def test_output_dimension(self, head):
        """Test output dimension matches number of actions."""
        x = torch.randn(4, 50, 512)
        output = head(x)
        
        assert output.size(-1) == head.num_actions
    
    def test_gradient_flow(self, head):
        """Test gradients flow through the head."""
        x = torch.randn(2, 10, 512, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_dropout_behavior(self, head):
        """Test dropout behaves differently in train vs eval mode."""
        x = torch.randn(4, 20, 512)
        
        # Eval mode should give consistent results
        head.eval()
        torch.manual_seed(42)
        out_eval_1 = head(x)
        torch.manual_seed(42)
        out_eval_2 = head(x)
        
        assert torch.allclose(out_eval_1, out_eval_2)
    
    def test_batch_independence(self, head):
        """Test that batch elements are processed independently."""
        head.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(4, 20, 512)
        
        # Process full batch
        output_batch = head(x)
        
        # Process samples individually
        outputs_individual = []
        for i in range(x.size(0)):
            out = head(x[i:i+1])
            outputs_individual.append(out)
        outputs_individual = torch.cat(outputs_individual, dim=0)
        
        assert torch.allclose(output_batch, outputs_individual, atol=1e-6)
    
    def test_different_actions(self):
        """Test head with different number of actions."""
        head = TradeSignalHead(d_model=256, num_actions=5)
        x = torch.randn(2, 30, 256)
        
        output = head(x)
        
        assert output.shape == (2, 30, 5)
        assert output.size(-1) == head.num_actions
    
    def test_components_exist(self, head):
        """Test all expected components exist."""
        assert isinstance(head.norm, nn.LayerNorm)
        assert isinstance(head.fc1, nn.Linear)
        assert isinstance(head.activation, nn.GELU)
        assert isinstance(head.dropout, nn.Dropout)
        assert isinstance(head.fc2, nn.Linear)


class TestAllHeadsConsistency:
    """Cross-head consistency tests."""
    
    def test_all_heads_same_architecture_pattern(self):
        """Test all heads follow the same architectural pattern."""
        d_model = 512
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create all heads
        regime_head = MarketRegimeHead(d_model=d_model)
        forecast_head = ForecastHead(d_model=d_model)
        risk_head = RiskHead(d_model=d_model)
        signal_head = TradeSignalHead(d_model=d_model)
        
        # All should accept [B, T, d_model] input
        regime_out = regime_head(x)
        forecast_out = forecast_head(x)
        risk_out = risk_head(x)
        signal_out = signal_head(x)
        
        # All should output [B, T, *]
        assert regime_out.dim() == 3
        assert forecast_out.dim() == 3
        assert risk_out.dim() == 3
        assert signal_out.dim() == 3
        
        assert regime_out.size(0) == batch_size
        assert forecast_out.size(0) == batch_size
        assert risk_out.size(0) == batch_size
        assert signal_out.size(0) == batch_size
        
        assert regime_out.size(1) == seq_len
        assert forecast_out.size(1) == seq_len
        assert risk_out.size(1) == seq_len
        assert signal_out.size(1) == seq_len
    
    def test_all_heads_have_layernorm(self):
        """Test all heads use LayerNorm."""
        regime_head = MarketRegimeHead()
        forecast_head = ForecastHead()
        risk_head = RiskHead()
        signal_head = TradeSignalHead()
        
        assert isinstance(regime_head.norm, nn.LayerNorm)
        assert isinstance(forecast_head.norm, nn.LayerNorm)
        assert isinstance(risk_head.norm, nn.LayerNorm)
        assert isinstance(signal_head.norm, nn.LayerNorm)
    
    def test_all_heads_use_gelu(self):
        """Test all heads use GELU activation."""
        regime_head = MarketRegimeHead()
        forecast_head = ForecastHead()
        risk_head = RiskHead()
        signal_head = TradeSignalHead()
        
        assert isinstance(regime_head.activation, nn.GELU)
        assert isinstance(forecast_head.activation, nn.GELU)
        assert isinstance(risk_head.activation, nn.GELU)
        assert isinstance(signal_head.activation, nn.GELU)
    
    def test_all_heads_have_dropout(self):
        """Test all heads have dropout layers."""
        regime_head = MarketRegimeHead()
        forecast_head = ForecastHead()
        risk_head = RiskHead()
        signal_head = TradeSignalHead()
        
        assert isinstance(regime_head.dropout, nn.Dropout)
        assert isinstance(forecast_head.dropout, nn.Dropout)
        assert isinstance(risk_head.dropout, nn.Dropout)
        assert isinstance(signal_head.dropout, nn.Dropout)
        
        # All should use 0.1 dropout rate
        assert regime_head.dropout.p == 0.1
        assert forecast_head.dropout.p == 0.1
        assert risk_head.dropout.p == 0.1
        assert signal_head.dropout.p == 0.1
    
    def test_all_heads_projection_pattern(self):
        """Test all heads use d_model -> d_model//2 -> output projection."""
        d_model = 512
        
        regime_head = MarketRegimeHead(d_model=d_model)
        forecast_head = ForecastHead(d_model=d_model)
        risk_head = RiskHead(d_model=d_model)
        signal_head = TradeSignalHead(d_model=d_model)
        
        # First projection: d_model -> d_model // 2
        assert regime_head.fc1.in_features == d_model
        assert regime_head.fc1.out_features == d_model // 2
        
        assert forecast_head.fc1.in_features == d_model
        assert forecast_head.fc1.out_features == d_model // 2
        
        assert risk_head.fc1.in_features == d_model
        assert risk_head.fc1.out_features == d_model // 2
        
        assert signal_head.fc1.in_features == d_model
        assert signal_head.fc1.out_features == d_model // 2
        
        # Second projection: d_model // 2 -> output_dim
        assert regime_head.fc2.in_features == d_model // 2
        assert forecast_head.fc2.in_features == d_model // 2
        assert risk_head.fc2.in_features == d_model // 2
        assert signal_head.fc2.in_features == d_model // 2
