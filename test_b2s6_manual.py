"""Manual test for B2S6Backbone implementation."""

import sys

import torch

sys.path.insert(0, '/Volumes/Containers/Dignity/.worktree/Dignity-Model')

from models.backbone.b2s6 import B2S6Backbone


def test_basic_functionality():
    """Test basic B2S6Backbone functionality."""
    print("Testing B2S6Backbone basic functionality...")
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass (12 layers)...")
    model = B2S6Backbone(d_model=256, num_layers=12, state_dim=64, num_blocks=4)
    x = torch.randn(2, 50, 256)
    y = model(x)
    assert y.shape == (2, 50, 256), f"Expected shape (2, 50, 256), got {y.shape}"
    print("   ✓ Basic forward pass works")
    
    # Test 2: Input projection
    print("\n2. Testing input projection (input_features != d_model)...")
    model_proj = B2S6Backbone(
        d_model=256, 
        num_layers=6, 
        input_features=100
    )
    x_proj = torch.randn(2, 30, 100)
    y_proj = model_proj(x_proj)
    assert y_proj.shape == (2, 30, 256), f"Expected shape (2, 30, 256), got {y_proj.shape}"
    assert model_proj.input_projection is not None, "Input projection should exist"
    print("   ✓ Input projection works correctly")
    
    # Test 3: No input projection when equal
    print("\n3. Testing no input projection when input_features == d_model...")
    model_no_proj = B2S6Backbone(
        d_model=256, 
        num_layers=6, 
        input_features=256
    )
    assert model_no_proj.input_projection is None, "Input projection should be None"
    print("   ✓ No unnecessary input projection")
    
    # Test 4: Deep network (24 layers)
    print("\n4. Testing deep network (24 layers) stability...")
    model_deep = B2S6Backbone(
        d_model=128, 
        num_layers=24, 
        state_dim=32, 
        num_blocks=4
    )
    x_deep = torch.randn(2, 50, 128)
    y_deep = model_deep(x_deep)
    
    assert not torch.isnan(y_deep).any(), "Output contains NaN values"
    assert not torch.isinf(y_deep).any(), "Output contains Inf values"
    assert y_deep.abs().max() < 100.0, f"Output exploded: max={y_deep.abs().max()}"
    assert y_deep.abs().mean() > 1e-6, f"Output vanished: mean={y_deep.abs().mean()}"
    print("   ✓ Deep network (24 layers) is stable")
    
    # Test 5: Gradient checkpointing
    print("\n5. Testing gradient checkpointing...")
    model_checkpoint = B2S6Backbone(
        d_model=128,
        num_layers=6,
        use_gradient_checkpointing=True
    )
    model_checkpoint.train()
    x_checkpoint = torch.randn(2, 50, 128, requires_grad=True)
    y_checkpoint = model_checkpoint(x_checkpoint)
    loss = y_checkpoint.sum()
    loss.backward()
    
    assert x_checkpoint.grad is not None, "Gradients should exist"
    assert x_checkpoint.grad.abs().sum() > 0, "Gradients should be non-zero"
    print("   ✓ Gradient checkpointing works")
    
    # Test 6: Gradient flow in deep network
    print("\n6. Testing gradient flow in deep network (24 layers)...")
    model_grad = B2S6Backbone(d_model=128, num_layers=24, state_dim=32, num_blocks=4)
    x_grad = torch.randn(2, 50, 128, requires_grad=True)
    y_grad = model_grad(x_grad)
    loss_grad = y_grad.sum()
    loss_grad.backward()
    
    grad_norm = x_grad.grad.norm().item()
    assert grad_norm > 1e-4, f"Gradient vanished: norm={grad_norm}"
    assert grad_norm < 1e4, f"Gradient exploded: norm={grad_norm}"
    print(f"   ✓ Gradients flow correctly (grad_norm={grad_norm:.4f})")
    
    # Test 7: Different layer counts
    print("\n7. Testing different layer counts (6, 12, 24)...")
    for num_layers in [6, 12, 24]:
        model_test = B2S6Backbone(d_model=128, num_layers=num_layers)
        assert len(model_test.layers) == num_layers
        x_test = torch.randn(2, 20, 128)
        y_test = model_test(x_test)
        assert y_test.shape == x_test.shape
    print("   ✓ All layer counts work correctly")
    
    # Test 8: Final layer norm
    print("\n8. Testing final layer normalization...")
    model_norm = B2S6Backbone(d_model=128, num_layers=4)
    x_norm = torch.randn(2, 20, 128) * 5.0  # High variance
    y_norm = model_norm(x_norm)
    
    mean = y_norm.mean().item()
    std = y_norm.std().item()
    assert abs(mean) < 1.0, f"Mean not controlled: {mean}"
    assert 0.5 < std < 2.0, f"Std not controlled: {std}"
    print(f"   ✓ Final layer norm applied (mean={mean:.4f}, std={std:.4f})")
    
    print("\n" + "="*60)
    print("✓ All tests passed successfully!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
