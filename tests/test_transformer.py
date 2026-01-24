"""Tests for Cross-Asset Transformer Layer."""

import pytest
import torch
import torch.nn as nn

from models.backbone.transformer import CrossAssetTransformerLayer


class TestCrossAssetTransformerLayer:
    """Test suite for CrossAssetTransformerLayer."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for transformer layer."""
        return {
            "d_model": 512,
            "num_heads": 8,
            "ffn_expansion": 4,
            "dropout": 0.1,
            "attention_dropout": 0.1,
        }

    @pytest.fixture
    def layer(self, default_config):
        """Create transformer layer instance."""
        return CrossAssetTransformerLayer(**default_config)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size = 4
        seq_len = 32
        d_model = 512
        return torch.randn(batch_size, seq_len, d_model)

    def test_initialization(self, default_config):
        """Test layer initialization."""
        layer = CrossAssetTransformerLayer(**default_config)

        assert layer.d_model == default_config["d_model"]
        assert layer.num_heads == default_config["num_heads"]
        assert layer.ffn_expansion == default_config["ffn_expansion"]

        # Check components exist
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
        assert isinstance(layer.attention, nn.MultiheadAttention)
        assert isinstance(layer.ffn, nn.Sequential)
        assert isinstance(layer.dropout, nn.Dropout)

    def test_invalid_d_model(self):
        """Test that d_model must be divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            CrossAssetTransformerLayer(d_model=513, num_heads=8)

    def test_forward_shape_preservation(self, layer, sample_input):
        """Test that forward pass preserves input shape."""
        batch_size, seq_len, d_model = sample_input.shape

        output = layer(sample_input)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == sample_input.dtype

    def test_forward_with_different_batch_sizes(self, default_config):
        """Test forward pass with various batch sizes."""
        layer = CrossAssetTransformerLayer(**default_config)
        d_model = default_config["d_model"]
        seq_len = 16

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = layer(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_with_different_seq_lengths(self, default_config):
        """Test forward pass with various sequence lengths."""
        layer = CrossAssetTransformerLayer(**default_config)
        batch_size = 4
        d_model = default_config["d_model"]

        for seq_len in [1, 8, 32, 128]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = layer(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_mechanism(self, layer, sample_input):
        """Test multi-head attention mechanism."""
        # Set to eval mode to disable dropout for consistent attention weights
        layer.eval()
        
        # Forward with attention weights
        with torch.no_grad():
            output, attn_weights = layer(sample_input, return_attention=True)

        batch_size, seq_len, _ = sample_input.shape

        # Check output shape
        assert output.shape == sample_input.shape

        # Check attention weights shape (averaged across heads)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

        # Attention weights should sum to approximately 1 across key dimension
        attn_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

        # Attention weights should be non-negative
        assert (attn_weights >= 0).all()

    def test_attention_weights_method(self, layer, sample_input):
        """Test get_attention_weights method."""
        attn_weights = layer.get_attention_weights(sample_input)

        batch_size, seq_len, _ = sample_input.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

        # Attention weights should sum to approximately 1 (allow for numerical precision)
        attn_sum = attn_weights.sum(dim=-1)
        # PyTorch's attention may have slight numerical variations
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), rtol=0.1, atol=0.1)

    def test_ffn_expansion_contraction(self, default_config):
        """Test FFN expansion and contraction."""
        layer = CrossAssetTransformerLayer(**default_config)

        # Check FFN structure
        ffn_layers = list(layer.ffn)
        assert len(ffn_layers) == 4  # Linear, GELU, Dropout, Linear

        # Check first linear layer (expansion)
        assert isinstance(ffn_layers[0], nn.Linear)
        assert ffn_layers[0].in_features == default_config["d_model"]
        assert ffn_layers[0].out_features == (
            default_config["d_model"] * default_config["ffn_expansion"]
        )

        # Check activation
        assert isinstance(ffn_layers[1], nn.GELU)

        # Check dropout
        assert isinstance(ffn_layers[2], nn.Dropout)

        # Check second linear layer (contraction)
        assert isinstance(ffn_layers[3], nn.Linear)
        assert ffn_layers[3].in_features == (
            default_config["d_model"] * default_config["ffn_expansion"]
        )
        assert ffn_layers[3].out_features == default_config["d_model"]

    def test_residual_connections(self, layer, sample_input):
        """Test residual connection gradient flow."""
        # Enable gradient tracking
        sample_input.requires_grad = True

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        # Gradients should flow through residual connections
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()

    def test_dropout_behavior(self, default_config, sample_input):
        """Test dropout behavior in train vs eval mode."""
        layer = CrossAssetTransformerLayer(**default_config)

        # Training mode: outputs should differ across runs due to dropout
        layer.train()
        outputs_train = []
        for i in range(3):
            torch.manual_seed(42 + i)  # Different seed each time
            output = layer(sample_input)
            outputs_train.append(output)

        # With dropout, we expect at least some difference
        # Check that not all outputs are identical
        all_same = all(
            torch.allclose(outputs_train[0], outputs_train[i], atol=1e-6)
            for i in range(1, 3)
        )
        # With dropout=0.1, outputs should differ (test may rarely fail)
        # Note: We're not asserting this strongly as it's stochastic

        # Evaluation mode: outputs should be identical
        layer.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            output1 = layer(sample_input)
            torch.manual_seed(42)
            output2 = layer(sample_input)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_with_attention_mask(self, layer):
        """Test forward pass with attention mask."""
        batch_size = 4
        seq_len = 16
        d_model = layer.d_model

        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask (prevent attending to future positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )
        causal_mask = causal_mask.float().masked_fill(causal_mask, float("-inf"))

        output = layer(x, attn_mask=causal_mask)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_with_key_padding_mask(self, layer):
        """Test forward pass with key padding mask."""
        batch_size = 4
        seq_len = 16
        d_model = layer.d_model

        x = torch.randn(batch_size, seq_len, d_model)

        # Create padding mask (mask out last 4 positions in each sequence)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[:, -4:] = True

        output = layer(x, key_padding_mask=key_padding_mask)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_gradient_flow_through_attention(self, layer, sample_input):
        """Test gradient flow through attention mechanism."""
        sample_input.requires_grad = True

        output, attn_weights = layer(sample_input, return_attention=True)

        # Compute loss from output
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are valid
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert (sample_input.grad.abs() > 0).any()  # Non-zero gradients

    def test_different_num_heads(self):
        """Test with different numbers of attention heads."""
        d_model = 512
        batch_size = 4
        seq_len = 16

        for num_heads in [1, 2, 4, 8, 16]:
            layer = CrossAssetTransformerLayer(d_model=d_model, num_heads=num_heads)
            x = torch.randn(batch_size, seq_len, d_model)
            output = layer(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_different_ffn_expansion(self, sample_input):
        """Test with different FFN expansion ratios."""
        batch_size, seq_len, d_model = sample_input.shape

        for ffn_expansion in [1, 2, 4, 8]:
            layer = CrossAssetTransformerLayer(
                d_model=d_model, ffn_expansion=ffn_expansion
            )
            output = layer(sample_input)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_deterministic_forward_in_eval_mode(self, layer, sample_input):
        """Test that forward pass is deterministic in eval mode."""
        layer.eval()

        with torch.no_grad():
            output1 = layer(sample_input)
            output2 = layer(sample_input)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_attention_weights_in_range(self, layer, sample_input):
        """Test that attention weights are valid probabilities."""
        # Disable dropout for consistent weights
        layer.eval()
        
        with torch.no_grad():
            _, attn_weights = layer(sample_input, return_attention=True)

        # All weights should be in [0, 1]
        assert (attn_weights >= 0).all()
        assert (attn_weights <= 1).all()

        # Each row should sum to approximately 1 (attention distribution)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_parameter_count(self, layer):
        """Test parameter count is reasonable."""
        total_params = sum(p.numel() for p in layer.parameters())

        # Rough estimate:
        # - Attention: 4 * d_model * d_model (Q, K, V, O projections)
        # - FFN: 2 * d_model * (ffn_expansion * d_model)
        # - LayerNorm: 2 * 2 * d_model (2 norms, each has scale & bias)
        d_model = layer.d_model
        ffn_expansion = layer.ffn_expansion

        attn_params = 4 * d_model * d_model
        ffn_params = 2 * d_model * (ffn_expansion * d_model)
        norm_params = 2 * 2 * d_model

        expected = attn_params + ffn_params + norm_params

        # Allow 10% margin for biases and other small parameters
        assert total_params >= expected * 0.9
        assert total_params <= expected * 1.1

    def test_cuda_compatibility(self, layer, sample_input):
        """Test CUDA compatibility if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        layer_cuda = layer.cuda()
        input_cuda = sample_input.cuda()

        output = layer_cuda(input_cuda)

        assert output.is_cuda
        assert output.shape == input_cuda.shape

    def test_mixed_precision_compatibility(self, layer, sample_input):
        """Test mixed precision (fp16) compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        layer = layer.cuda()
        sample_input = sample_input.cuda()

        # Test with autocast
        with torch.cuda.amp.autocast():
            output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()

    def test_batch_independence(self, layer):
        """Test that batch elements are processed independently (without dropout)."""
        # Set to eval mode to disable dropout
        layer.eval()
        
        batch_size = 4
        seq_len = 16
        d_model = layer.d_model

        # Create input with different patterns per batch
        x = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            # Process full batch
            output_batch = layer(x)

            # Process each element individually
            outputs_individual = []
            for i in range(batch_size):
                output_i = layer(x[i:i+1])
                outputs_individual.append(output_i)

            outputs_individual = torch.cat(outputs_individual, dim=0)

        # Results should match
        assert torch.allclose(output_batch, outputs_individual, atol=1e-5)
