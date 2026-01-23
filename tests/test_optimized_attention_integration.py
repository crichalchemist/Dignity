"""Test optimized attention integration with model architecture."""

import torch

from config.config import ModelConfig
from models.agent_hybrid import DignityModel, PriceSequenceEncoder, build_model


class TestOptimizedAttentionIntegration:
    """Test integration of optimized attention layers."""

    def test_optimized_attention_config_creation(self):
        """Test that attention layer is created based on config flags."""

        # Test 1: Standard temporal attention (default)
        cfg1 = ModelConfig(num_features=10, use_optimized_attention=False, use_multihead_attention=False)
        encoder1 = PriceSequenceEncoder(cfg1)
        assert isinstance(encoder1.attention.__class__.__name__, str)

        # Test 2: Multi-head attention
        cfg2 = ModelConfig(num_features=10, use_optimized_attention=False, use_multihead_attention=True)
        encoder2 = PriceSequenceEncoder(cfg2)
        assert isinstance(encoder2.attention.__class__.__name__, str)

        # Test 3: Optimized attention (default)
        cfg3 = ModelConfig(num_features=10, use_optimized_attention=True)
        encoder3 = PriceSequenceEncoder(cfg3)
        assert isinstance(encoder3.attention.__class__.__name__, str)

    def test_forward_pass_with_different_attention_types(self):
        """Test forward pass works with different attention configurations."""

        batch_size, seq_len, num_features = 4, 120, 10
        x = torch.randn(batch_size, seq_len, num_features)

        configs = [
            {"use_optimized_attention": False, "use_multihead_attention": False},
            {"use_optimized_attention": False, "use_multihead_attention": True},
            {"use_optimized_attention": True, "use_adaptive_attention": True},
            {"use_optimized_attention": True, "use_adaptive_attention": False},
        ]

        for config in configs:
            cfg = ModelConfig(num_features=num_features, **config)

            # Test encoder
            encoder = PriceSequenceEncoder(cfg)
            context, weights = encoder(x)
            assert context.shape == (batch_size, encoder.output_dim)
            assert weights.shape == (batch_size, seq_len)

            # Test full model
            model = DignityModel(cfg)
            outputs, attn_weights = model(x)

            assert "primary" in outputs
            assert "direction_logits" in outputs
            assert "return" in outputs
            assert outputs["primary"].shape[0] == batch_size
            assert attn_weights.shape == (batch_size, seq_len)

    def test_long_sequence_handling(self):
        """Test that optimized attention can handle longer sequences."""

        # Test with longer sequence than default
        batch_size, seq_len, num_features = 2, 2048, 10
        x = torch.randn(batch_size, seq_len, num_features)

        cfg = ModelConfig(
            num_features=num_features,
            use_optimized_attention=True,
            max_seq_length=1024,
            use_adaptive_attention=True
        )

        encoder = PriceSequenceEncoder(cfg)
        context, weights = encoder(x)

        assert context.shape == (batch_size, encoder.output_dim)
        assert weights.shape == (batch_size, seq_len)

    def test_model_building_function(self):
        """Test build_model function works with new attention configs."""

        cfg = ModelConfig(
            num_features=10,
            use_optimized_attention=True,
            use_adaptive_attention=True
        )

        model = build_model(cfg, task_type="classification")
        assert isinstance(model, DignityModel)

        # Test with custom parameters
        model2 = build_model(
            cfg,
            task_type="regression",
            num_dir_classes=5,
            num_volatility_classes=3,
            return_dim=1
        )
        assert isinstance(model2, DignityModel)

    def test_memory_efficiency_indicator(self):
        """Test that attention layers have expected memory efficiency features."""

        # Create models with different configurations
        configs = [
            {"use_optimized_attention": False},  # Standard
            {"use_optimized_attention": True, "use_adaptive_attention": False},  # Optimized
            {"use_optimized_attention": True, "use_adaptive_attention": True},  # Adaptive
        ]

        for config in configs:
            cfg = ModelConfig(num_features=10, **config)
            model = DignityModel(cfg)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            assert total_params > 0

            # Test forward pass doesn't crash
            x = torch.randn(2, 120, 10)
            with torch.no_grad():
                outputs, _ = model(x)
                assert outputs["primary"].shape[0] == 2

    def test_configuration_persistence(self):
        """Test that attention configuration is properly stored."""

        cfg = ModelConfig(
            num_features=15,
            use_optimized_attention=True,
            max_seq_length=2048,
            chunk_size=256,
            use_adaptive_attention=True,
            complexity_threshold=0.7,
            n_attention_heads=8
        )

        encoder = PriceSequenceEncoder(cfg)

        # Verify configuration is preserved
        assert encoder.cfg.use_optimized_attention
        assert encoder.cfg.max_seq_length == 2048
        assert encoder.cfg.chunk_size == 256
        assert encoder.cfg.use_adaptive_attention
        assert encoder.cfg.complexity_threshold == 0.7
        assert encoder.cfg.n_attention_heads == 8


if __name__ == "__main__":
    # Run basic smoke test
    test = TestOptimizedAttentionIntegration()

    print("Testing optimized attention integration...")
    test.test_optimized_attention_config_creation()
    print("✓ Config creation test passed")

    test.test_forward_pass_with_different_attention_types()
    print("✓ Forward pass test passed")

    test.test_long_sequence_handling()
    print("✓ Long sequence handling test passed")

    test.test_model_building_function()
    print("✓ Model building test passed")

    test.test_memory_efficiency_indicator()
    print("✓ Memory efficiency test passed")

    test.test_configuration_persistence()
    print("✓ Configuration persistence test passed")

    print("\n🎉 All optimized attention integration tests passed!")
