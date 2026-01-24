"""Quick functional test for Dignity Foundation Model."""

import torch

from models.foundation import create_model


def test_functional():
    """Test that the foundation model works end-to-end."""
    print('=== Dignity Foundation Model - Quick Functional Test ===\n')

    # Create small model
    print('Creating small model...')
    model = create_model(size='small')
    param_count = sum(p.numel() for p in model.parameters())
    print(f'✓ Model created with {param_count:,} parameters\n')

    # Test forward pass with raw input features (OHLCV + 8 indicators = 13)
    print('Testing forward pass...')
    batch_size, seq_len, input_features = 2, 128, 13
    x = torch.randn(batch_size, seq_len, input_features)

    outputs = model(x)
    print(f'✓ Forward pass successful\n')

    # Check all outputs
    print('Output shapes:')
    print(f'  regime:   {outputs["regime"].shape} (expected: [{batch_size}, {seq_len}, 4])')
    print(f'  forecast: {outputs["forecast"].shape} (expected: [{batch_size}, {seq_len}, 3])')
    print(f'  risk:     {outputs["risk"].shape} (expected: [{batch_size}, {seq_len}, 3])')
    print(f'  trade_signal: {outputs["trade_signal"].shape} (expected: [{batch_size}, {seq_len}, 3])')
    print()

    # Verify shapes
    assert outputs["regime"].shape == (batch_size, seq_len, 4)
    assert outputs["forecast"].shape == (batch_size, seq_len, 3)
    assert outputs["risk"].shape == (batch_size, seq_len, 3)
    assert outputs["trade_signal"].shape == (batch_size, seq_len, 3)

    # Test gradient flow
    print('Testing gradient flow...')
    loss = (outputs['regime'].sum() + outputs['forecast'].sum() + 
            outputs['risk'].sum() + outputs['trade_signal'].sum())
    loss.backward()

    gradients_exist = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f'✓ All parameters have gradients: {gradients_exist}\n')

    # Test medium model creation
    print('Testing medium model creation...')
    model_medium = create_model(size='medium')
    param_count_medium = sum(p.numel() for p in model_medium.parameters())
    print(f'✓ Medium model created with {param_count_medium:,} parameters\n')

    # Test eval mode
    print('Testing eval mode...')
    model.eval()
    with torch.no_grad():
        outputs_eval = model(x)
    print(f'✓ Eval mode works\n')

    print('=== All Functional Tests Passed! ===')
    return True


if __name__ == '__main__':
    test_functional()
