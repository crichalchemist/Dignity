"""Tests for CNN1D Feature Extractor."""

import pytest
import torch
import torch.nn as nn

from models.backbone.cnn1d import CNN1DFeatureExtractor


class TestCNN1DFeatureExtractor:
    """Test suite for CNN1DFeatureExtractor."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for feature extractor."""
        return {
            "input_features": 13,
            "d_model": 512,
            "kernel_sizes": [3, 5, 7],
            "hidden_channels": 128,
            "dropout": 0.1,
        }

    @pytest.fixture
    def extractor(self, default_config):
        """Create feature extractor instance."""
        return CNN1DFeatureExtractor(**default_config)

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size = 4
        seq_len = 32
        input_features = 13  # OHLCV + 8 technical indicators
        return torch.randn(batch_size, seq_len, input_features)

    def test_initialization(self, default_config):
        """Test feature extractor initialization."""
        extractor = CNN1DFeatureExtractor(**default_config)

        assert extractor.input_features == default_config["input_features"]
        assert extractor.d_model == default_config["d_model"]
        assert extractor.kernel_sizes == default_config["kernel_sizes"]
        assert extractor.hidden_channels == default_config["hidden_channels"]

        # Check components exist
        assert len(extractor.conv_pathways) == len(default_config["kernel_sizes"])
        assert isinstance(extractor.projection, nn.Linear)

        # Should not use residual when input_features != d_model
        assert not extractor.use_residual

    def test_initialization_with_residual(self):
        """Test initialization when residual connection is enabled."""
        extractor = CNN1DFeatureExtractor(
            input_features=512, d_model=512, hidden_channels=128
        )

        assert extractor.use_residual
        assert isinstance(extractor.residual_norm, nn.LayerNorm)

    def test_default_kernel_sizes(self):
        """Test that default kernel sizes are [3, 5, 7]."""
        extractor = CNN1DFeatureExtractor(input_features=13, d_model=512)

        assert extractor.kernel_sizes == [3, 5, 7]
        assert len(extractor.conv_pathways) == 3

    def test_forward_shape_preservation(self, extractor, sample_input):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, _ = sample_input.shape

        output = extractor(sample_input)

        # Should output [batch_size, seq_len, d_model]
        assert output.shape == (batch_size, seq_len, extractor.d_model)
        assert output.dtype == sample_input.dtype

    def test_forward_with_different_batch_sizes(self, default_config):
        """Test forward pass with various batch sizes."""
        extractor = CNN1DFeatureExtractor(**default_config)
        seq_len = 16
        input_features = default_config["input_features"]
        d_model = default_config["d_model"]

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, seq_len, input_features)
            output = extractor(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_with_different_seq_lengths(self, default_config):
        """Test forward pass with various sequence lengths."""
        extractor = CNN1DFeatureExtractor(**default_config)
        batch_size = 4
        input_features = default_config["input_features"]
        d_model = default_config["d_model"]

        for seq_len in [1, 8, 32, 64, 128]:
            x = torch.randn(batch_size, seq_len, input_features)
            output = extractor(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_multi_scale_pathways(self, default_config):
        """Test that multi-scale pathways are created correctly."""
        extractor = CNN1DFeatureExtractor(**default_config)
        kernel_sizes = default_config["kernel_sizes"]

        # Check number of pathways
        assert len(extractor.conv_pathways) == len(kernel_sizes)

        # Check each pathway structure
        for i, pathway in enumerate(extractor.conv_pathways):
            layers = list(pathway)

            # Should have Conv1D, BatchNorm, GELU, Dropout
            assert len(layers) == 4
            assert isinstance(layers[0], nn.Conv1d)
            assert isinstance(layers[1], nn.BatchNorm1d)
            assert isinstance(layers[2], nn.GELU)
            assert isinstance(layers[3], nn.Dropout)

            # Check conv layer properties
            conv = layers[0]
            assert conv.in_channels == default_config["input_features"]
            assert conv.out_channels == default_config["hidden_channels"]
            assert conv.kernel_size == (kernel_sizes[i],)
            # Causal padding: kernel_size - 1
            assert conv.padding == (kernel_sizes[i] - 1,)

    def test_causal_convolution(self, default_config):
        """Test that convolutions are causal (no future information leakage)."""
        extractor = CNN1DFeatureExtractor(**default_config)
        extractor.eval()

        batch_size = 1
        seq_len = 10
        input_features = default_config["input_features"]

        # Create input with zeros except at specific timestep
        x = torch.zeros(batch_size, seq_len, input_features)
        timestep = 5
        x[:, timestep, :] = 1.0

        with torch.no_grad():
            output = extractor(x)

        # Output at timesteps before the spike should be very small
        # (only affected by padding zeros, not the spike)
        pre_spike = output[:, :timestep, :]
        assert torch.allclose(pre_spike, torch.zeros_like(pre_spike), atol=1e-1)

        # Output at and after the spike can be non-zero
        # (this is expected due to the convolution processing the spike)

    def test_residual_connection(self):
        """Test residual connection when input_features == d_model."""
        d_model = 512
        extractor = CNN1DFeatureExtractor(
            input_features=d_model, d_model=d_model, hidden_channels=128, dropout=0.0
        )

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)

        # Set to eval mode
        extractor.eval()

        with torch.no_grad():
            output = extractor(x)

        # Output should not be identical to input (due to transformation)
        assert not torch.allclose(output, x)

        # But output magnitude should be reasonable (residual helps)
        assert output.abs().mean() > 0

    def test_no_residual_connection(self, default_config):
        """Test that residual connection is not used when dimensions don't match."""
        extractor = CNN1DFeatureExtractor(**default_config)

        # input_features (13) != d_model (512)
        assert not extractor.use_residual
        assert not hasattr(extractor, "residual_norm") or extractor.residual_norm is None

    def test_dropout_behavior(self, default_config):
        """Test dropout behavior in train vs eval mode."""
        extractor = CNN1DFeatureExtractor(**default_config)

        batch_size = 4
        seq_len = 32
        input_features = default_config["input_features"]
        x = torch.randn(batch_size, seq_len, input_features)

        # Train mode: dropout should cause variation
        extractor.train()
        outputs_train = []
        for _ in range(5):
            with torch.no_grad():
                outputs_train.append(extractor(x))

        # Check that outputs vary (dropout is active)
        variation = torch.stack([torch.abs(outputs_train[i] - outputs_train[0]).mean() 
                                for i in range(1, 5)])
        assert variation.mean() > 1e-6

        # Eval mode: should be deterministic
        extractor.eval()
        with torch.no_grad():
            output1 = extractor(x)
            output2 = extractor(x)

        assert torch.allclose(output1, output2)

    def test_batch_independence(self, default_config):
        """Test that samples in batch are processed independently."""
        extractor = CNN1DFeatureExtractor(**default_config)
        extractor.eval()

        seq_len = 16
        input_features = default_config["input_features"]

        # Create two different inputs
        x1 = torch.randn(1, seq_len, input_features)
        x2 = torch.randn(1, seq_len, input_features)

        # Process separately
        with torch.no_grad():
            out1 = extractor(x1)
            out2 = extractor(x2)

        # Process together
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            out_batch = extractor(x_batch)

        # Results should match
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)

    def test_projection_layer(self, default_config):
        """Test that projection layer has correct dimensions."""
        extractor = CNN1DFeatureExtractor(**default_config)

        kernel_sizes = default_config["kernel_sizes"]
        hidden_channels = default_config["hidden_channels"]
        d_model = default_config["d_model"]

        # Total channels after concatenation
        total_channels = len(kernel_sizes) * hidden_channels

        # Check projection layer
        assert extractor.projection.in_features == total_channels
        assert extractor.projection.out_features == d_model

    def test_custom_kernel_sizes(self):
        """Test with custom kernel sizes."""
        custom_kernels = [2, 4, 6, 8]
        extractor = CNN1DFeatureExtractor(
            input_features=13,
            d_model=512,
            kernel_sizes=custom_kernels,
            hidden_channels=64,
        )

        assert len(extractor.conv_pathways) == len(custom_kernels)
        assert extractor.kernel_sizes == custom_kernels

        # Test forward pass
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 13)

        output = extractor(x)
        assert output.shape == (batch_size, seq_len, 512)

    def test_single_kernel_size(self):
        """Test with a single kernel size."""
        extractor = CNN1DFeatureExtractor(
            input_features=13,
            d_model=256,
            kernel_sizes=[5],
            hidden_channels=256,
        )

        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 13)

        output = extractor(x)
        assert output.shape == (batch_size, seq_len, 256)

    def test_output_range_reasonable(self, extractor, sample_input):
        """Test that output values are in a reasonable range."""
        extractor.eval()

        with torch.no_grad():
            output = extractor(sample_input)

        # Output should not have extreme values (basic sanity check)
        assert output.abs().max() < 100
        assert output.isfinite().all()

    def test_gradient_flow(self, extractor, sample_input):
        """Test that gradients flow through the network."""
        extractor.train()

        # Forward pass
        output = extractor(sample_input)

        # Compute loss (dummy loss for testing)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not param.grad.isnan().any(), f"NaN gradient for {name}"

    def test_batchnorm_in_pathways(self, default_config):
        """Test that BatchNorm is correctly placed in pathways."""
        extractor = CNN1DFeatureExtractor(**default_config)

        for pathway in extractor.conv_pathways:
            layers = list(pathway)
            
            # BatchNorm should be after Conv1d
            assert isinstance(layers[0], nn.Conv1d)
            assert isinstance(layers[1], nn.BatchNorm1d)

            # Conv1d should not have bias (BatchNorm handles it)
            assert not layers[0].bias

    def test_sequence_length_one(self, default_config):
        """Test edge case with sequence length of 1."""
        extractor = CNN1DFeatureExtractor(**default_config)

        batch_size = 2
        seq_len = 1
        input_features = default_config["input_features"]
        d_model = default_config["d_model"]

        x = torch.randn(batch_size, seq_len, input_features)
        output = extractor(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.isfinite().all()
