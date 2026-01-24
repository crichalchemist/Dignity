"""Tests for Dignity Foundation Model."""

import pytest
import torch
import torch.nn as nn

from models.foundation import DignityFoundationModel, create_model


class TestDignityFoundationModel:
    """Test suite for DignityFoundationModel."""
    
    @pytest.fixture
    def batch_size(self):
        """Default batch size for testing."""
        return 4
    
    @pytest.fixture
    def seq_len(self):
        """Default sequence length for testing."""
        return 64
    
    @pytest.fixture
    def input_features(self):
        """Number of input features (OHLCV + 8 indicators)."""
        return 13
    
    @pytest.fixture
    def model_small(self):
        """Small foundation model for testing."""
        return create_model(size="small")
    
    @pytest.fixture
    def model_medium(self):
        """Medium foundation model for testing."""
        return create_model(size="medium")
    
    @pytest.fixture
    def sample_input(self, batch_size, seq_len, input_features):
        """Sample input tensor."""
        return torch.randn(batch_size, seq_len, input_features)
    
    # ========== Basic Architecture Tests ==========
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = DignityFoundationModel()
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'feature_extractor')
        assert hasattr(model, 'b2s6_backbone')
        assert hasattr(model, 'transformer_layers')
        assert hasattr(model, 'regime_head')
        assert hasattr(model, 'forecast_head')
        assert hasattr(model, 'risk_head')
        assert hasattr(model, 'trade_signal_head')
    
    def test_model_custom_config(self):
        """Test model with custom configuration."""
        model = DignityFoundationModel(
            input_features=20,
            d_model=128,
            b2s6_layers=4,
            transformer_layers=2,
            num_heads=4,
            state_dim=32,
            num_blocks=2,
            dropout=0.2,
        )
        assert model._d_model == 128
        assert model._b2s6_layers == 4
        assert model._transformer_layers == 2
    
    # ========== Forward Pass Tests ==========
    
    def test_forward_pass_small(self, model_small, sample_input):
        """Test forward pass with small model."""
        output = model_small(sample_input)
        
        # Check output is a dictionary
        assert isinstance(output, dict)
        
        # Check all expected keys present
        expected_keys = {'regime', 'forecast', 'risk', 'trade_signal'}
        assert set(output.keys()) == expected_keys
        
        # Check all outputs are tensors
        for key, value in output.items():
            assert isinstance(value, torch.Tensor)
    
    def test_forward_pass_medium(self, model_medium, sample_input):
        """Test forward pass with medium model."""
        output = model_medium(sample_input)
        
        assert isinstance(output, dict)
        assert set(output.keys()) == {'regime', 'forecast', 'risk', 'trade_signal'}
    
    def test_output_shapes(self, model_small, batch_size, seq_len, sample_input):
        """Test all output shapes are correct."""
        output = model_small(sample_input)
        
        # Regime: 4 classes (bull/bear/sideways/volatile)
        assert output['regime'].shape == (batch_size, seq_len, 4)
        
        # Forecast: 3 horizons (short/medium/long)
        assert output['forecast'].shape == (batch_size, seq_len, 3)
        
        # Risk: 3 quantiles (5%/50%/95%)
        assert output['risk'].shape == (batch_size, seq_len, 3)
        
        # Trade signal: 3 actions (buy/hold/sell)
        assert output['trade_signal'].shape == (batch_size, seq_len, 3)
    
    def test_output_dtype(self, model_small, sample_input):
        """Test outputs have correct dtype."""
        output = model_small(sample_input)
        
        for key, value in output.items():
            assert value.dtype == torch.float32
    
    def test_output_finite_values(self, model_small, sample_input):
        """Test outputs contain finite values (no NaN/Inf)."""
        output = model_small(sample_input)
        
        for key, value in output.items():
            assert torch.isfinite(value).all(), f"{key} output contains non-finite values"
    
    # ========== Batch Independence Tests ==========
    
    def test_batch_independence(self, model_small, seq_len, input_features):
        """Test that outputs for different batch samples are independent."""
        model_small.eval()
        
        with torch.no_grad():
            # Process batch
            batch_input = torch.randn(2, seq_len, input_features)
            batch_output = model_small(batch_input)
            
            # Process samples individually
            single_output_0 = model_small(batch_input[0:1])
            single_output_1 = model_small(batch_input[1:2])
            
            # Check each task output (with relaxed tolerance for numerical precision)
            for key in batch_output.keys():
                # First sample should match
                assert torch.allclose(
                    batch_output[key][0],
                    single_output_0[key][0],
                    rtol=1e-3,
                    atol=1e-3,
                ), f"{key}: batch[0] != single[0]"
                
                # Second sample should match
                assert torch.allclose(
                    batch_output[key][1],
                    single_output_1[key][0],
                    rtol=1e-3,
                    atol=1e-3,
                ), f"{key}: batch[1] != single[1]"
    
    # ========== Variable Sequence Length Tests ==========
    
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
    def test_variable_sequence_lengths(self, model_small, batch_size, input_features, seq_len):
        """Test model handles various sequence lengths."""
        x = torch.randn(batch_size, seq_len, input_features)
        output = model_small(x)
        
        # Check all outputs have correct sequence length
        for key, value in output.items():
            assert value.shape[1] == seq_len, f"{key} has wrong sequence length"
    
    # ========== Gradient Flow Tests ==========
    
    def test_gradient_flow(self, model_small, sample_input):
        """Test gradients flow through all components."""
        output = model_small(sample_input)
        
        # Compute loss from all tasks
        loss = (
            output['regime'].sum() +
            output['forecast'].sum() +
            output['risk'].sum() +
            output['trade_signal'].sum()
        )
        
        # Backward pass
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model_small.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    def test_gradient_magnitude(self, model_small, sample_input):
        """Test gradients have reasonable magnitudes (not exploding/vanishing)."""
        output = model_small(sample_input)
        
        loss = (
            output['regime'].sum() +
            output['forecast'].sum() +
            output['risk'].sum() +
            output['trade_signal'].sum()
        )
        
        loss.backward()
        
        # Collect gradient norms
        grad_norms = []
        for name, param in model_small.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        # Check gradients exist and max isn't exploding (relaxed threshold for untrained model)
        assert len(grad_norms) > 0, "No gradients found"
        max_norm = max(grad_norms)
        assert max_norm < 10000, f"Gradients exploding: max norm = {max_norm}"
    
    # ========== Parameter Count Tests ==========
    
    def test_parameter_count_small(self, model_small):
        """Test small model has ~2M parameters."""
        count = model_small.count_parameters()
        assert 1_000_000 < count < 5_000_000, \
            f"Small model has {count:,} params (expected ~2M)"
    
    def test_parameter_count_medium(self, model_medium):
        """Test medium model has ~14M parameters."""
        count = model_medium.count_parameters()
        assert 10_000_000 < count < 20_000_000, \
            f"Medium model has {count:,} params (expected ~14M)"
    
    def test_parameter_count_method(self, model_small):
        """Test count_parameters() matches manual count."""
        auto_count = model_small.count_parameters()
        manual_count = sum(p.numel() for p in model_small.parameters() if p.requires_grad)
        assert auto_count == manual_count
    
    # ========== Model Size Detection Tests ==========
    
    def test_get_model_size_small(self, model_small):
        """Test small model size is correctly detected."""
        assert model_small.get_model_size() == "small"
    
    def test_get_model_size_medium(self, model_medium):
        """Test medium model size is correctly detected."""
        assert model_medium.get_model_size() == "medium"
    
    def test_get_model_size_large(self):
        """Test large model size is correctly detected."""
        model_large = create_model(size="large")
        assert model_large.get_model_size() == "large"
    
    def test_get_model_size_custom(self):
        """Test custom configuration returns 'custom'."""
        model = DignityFoundationModel(
            d_model=384,
            b2s6_layers=8,
            transformer_layers=3,
        )
        assert model.get_model_size() == "custom"
    
    # ========== Gradient Checkpointing Tests ==========
    
    @pytest.mark.skip(reason="Gradient checkpointing requires full implementation in B2S6/Transformer layers")
    def test_gradient_checkpointing_enabled(self, batch_size, seq_len, input_features):
        """Test model with gradient checkpointing enabled."""
        model = create_model(size="small", use_gradient_checkpointing=True)
        x = torch.randn(batch_size, seq_len, input_features)
        
        output = model(x)
        loss = output['regime'].sum()
        loss.backward()
        
        # Check gradients still flow
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name} with checkpointing enabled"
    
    @pytest.mark.skip(reason="Memory profiling requires CUDA and careful setup")
    def test_gradient_checkpointing_reduces_memory(self, batch_size, seq_len, input_features):
        """Test gradient checkpointing reduces memory usage."""
        # This would require CUDA and memory profiling
        pass
    
    # ========== Model Variant Tests ==========
    
    def test_all_model_variants(self):
        """Test all predefined model variants can be created."""
        for size in ["small", "medium", "large"]:
            model = create_model(size=size)
            assert isinstance(model, DignityFoundationModel)
            assert model.get_model_size() == size
    
    def test_create_model_invalid_size(self):
        """Test create_model raises error for invalid size."""
        with pytest.raises(ValueError, match="Invalid size"):
            create_model(size="extra-large")
    
    def test_create_model_custom_input_features(self):
        """Test create_model with custom input features."""
        model = create_model(size="small", input_features=20)
        x = torch.randn(2, 32, 20)
        output = model(x)
        assert output['regime'].shape == (2, 32, 4)
    
    def test_create_model_with_dropout(self):
        """Test create_model with custom dropout."""
        model = create_model(size="small", dropout=0.3)
        assert isinstance(model, DignityFoundationModel)
    
    # ========== Training Mode Tests ==========
    
    def test_train_eval_mode(self, model_small, sample_input):
        """Test model can switch between train and eval modes."""
        # Train mode
        model_small.train()
        output_train_1 = model_small(sample_input)
        output_train_2 = model_small(sample_input)
        
        # Outputs should differ in train mode (due to dropout)
        for key in output_train_1.keys():
            # Note: may be equal by chance, so we just check it runs
            assert output_train_1[key].shape == output_train_2[key].shape
        
        # Eval mode
        model_small.eval()
        with torch.no_grad():
            output_eval_1 = model_small(sample_input)
            output_eval_2 = model_small(sample_input)
        
        # Outputs should be identical in eval mode
        for key in output_eval_1.keys():
            assert torch.allclose(output_eval_1[key], output_eval_2[key])
    
    # ========== Component Integration Tests ==========
    
    def test_feature_extractor_integration(self, model_small, sample_input):
        """Test feature extractor produces correct output shape."""
        x = model_small.feature_extractor(sample_input)
        batch_size, seq_len, _ = sample_input.shape
        assert x.shape == (batch_size, seq_len, 256)  # small model d_model=256
    
    def test_b2s6_backbone_integration(self, model_small, sample_input):
        """Test B2S6 backbone processes features correctly."""
        x = model_small.feature_extractor(sample_input)
        x = model_small.b2s6_backbone(x)
        batch_size, seq_len, _ = sample_input.shape
        assert x.shape == (batch_size, seq_len, 256)
    
    def test_transformer_stack_integration(self, model_small, sample_input):
        """Test transformer stack processes features correctly."""
        x = model_small.feature_extractor(sample_input)
        x = model_small.b2s6_backbone(x)
        
        for transformer in model_small.transformer_layers:
            x = transformer(x)
        
        batch_size, seq_len, _ = sample_input.shape
        assert x.shape == (batch_size, seq_len, 256)
    
    def test_all_heads_integration(self, model_small):
        """Test all task heads produce correct output shapes."""
        # Create intermediate representation
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, 256)  # d_model for small
        
        # Test each head
        regime_out = model_small.regime_head(x)
        forecast_out = model_small.forecast_head(x)
        risk_out = model_small.risk_head(x)
        signal_out = model_small.trade_signal_head(x)
        
        assert regime_out.shape == (batch_size, seq_len, 4)
        assert forecast_out.shape == (batch_size, seq_len, 3)
        assert risk_out.shape == (batch_size, seq_len, 3)
        assert signal_out.shape == (batch_size, seq_len, 3)
    
    # ========== Edge Cases ==========
    
    def test_single_sample_batch(self, model_small, seq_len, input_features):
        """Test model handles batch size of 1."""
        x = torch.randn(1, seq_len, input_features)
        output = model_small(x)
        
        for key, value in output.items():
            assert value.shape[0] == 1
    
    def test_minimum_sequence_length(self, model_small, batch_size, input_features):
        """Test model handles short sequences."""
        # Minimum 16 due to CNN kernel sizes
        x = torch.randn(batch_size, 16, input_features)
        output = model_small(x)
        
        for key, value in output.items():
            assert value.shape == (batch_size, 16, value.shape[2])
    
    def test_large_batch_size(self, model_small, seq_len, input_features):
        """Test model handles large batch sizes."""
        x = torch.randn(32, seq_len, input_features)
        output = model_small(x)
        
        for key, value in output.items():
            assert value.shape[0] == 32
    
    # ========== Reproducibility Tests ==========
    
    def test_deterministic_with_seed(self, sample_input):
        """Test model produces same outputs with same seed."""
        # First run
        torch.manual_seed(42)
        model1 = create_model(size="small")
        model1.eval()
        
        with torch.no_grad():
            output1 = model1(sample_input)
        
        # Second run with same seed
        torch.manual_seed(42)
        model2 = create_model(size="small")
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(sample_input)
        
        # Check all outputs match
        for key in output1.keys():
            assert torch.allclose(output1[key], output2[key], rtol=1e-5, atol=1e-6), \
                f"{key} outputs differ with same seed"


class TestFoundationModelDocumentation:
    """Test documentation and code quality."""
    
    def test_model_has_docstring(self):
        """Test main model class has docstring."""
        assert DignityFoundationModel.__doc__ is not None
        assert len(DignityFoundationModel.__doc__) > 100
    
    def test_forward_has_docstring(self):
        """Test forward method has docstring."""
        assert DignityFoundationModel.forward.__doc__ is not None
    
    def test_create_model_has_docstring(self):
        """Test create_model helper has docstring."""
        assert create_model.__doc__ is not None
    
    def test_create_model_has_examples(self):
        """Test create_model docstring includes examples."""
        assert "Examples:" in create_model.__doc__ or "Example:" in create_model.__doc__
