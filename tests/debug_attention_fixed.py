"""Direct test of FIXED attention mechanisms to verify shape fixes."""

import torch

from utils.attention_optimization import (
    AdaptiveAttention,
    OptimizedMultiHeadAttention,
    TemporalAttention,
)

# Create test input
batch_size, seq_len, input_dim = 4, 120, 160
x = torch.randn(batch_size, seq_len, input_dim)

print(f"Input shape: {x.shape}")

# Test OptimizedMultiHeadAttention
print("\n--- Testing FIXED OptimizedMultiHeadAttention ---")
attn1 = OptimizedMultiHeadAttention(
    input_dim=input_dim,
    attention_dim=128,
    n_heads=4,
    max_seq_length=2048,
    use_chunking=False,
    chunk_size=512
)

try:
    context, weights = attn1(x)
    print(f"Output shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Expected: context=[{batch_size}, {input_dim}], weights=[{batch_size}, {seq_len}]")

    if context.shape == (batch_size, input_dim):
        print("✓ FIXED OptimizedMultiHeadAttention shape is correct!")
    else:
        print(
            f"✗ FIXED OptimizedMultiHeadAttention shape is wrong: expected ({batch_size}, {input_dim}), got {context.shape}")

except Exception as e:
    print(f"✗ Error in FIXED OptimizedMultiHeadAttention: {e}")
    import traceback

    traceback.print_exc()

# Test TemporalAttention for comparison
print("\n--- Testing TemporalAttention (for comparison) ---")
attn2 = TemporalAttention(input_dim=input_dim, attention_dim=128)

try:
    context, weights = attn2(x)
    print(f"Output shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Expected: context=[{batch_size}, {input_dim}], weights=[{batch_size}, {seq_len}]")

    if context.shape == (batch_size, input_dim):
        print("✓ TemporalAttention shape is correct!")
    else:
        print(f"✗ TemporalAttention shape is wrong: expected ({batch_size}, {input_dim}), got {context.shape}")

except Exception as e:
    print(f"✗ Error in TemporalAttention: {e}")

# Test AdaptiveAttention
print("\n--- Testing FIXED AdaptiveAttention ---")
attn3 = AdaptiveAttention(input_dim=input_dim, attention_dim=128, complexity_threshold=0.5)

try:
    context, weights = attn3(x)
    print(f"Output shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Expected: context=[{batch_size}, {input_dim}], weights=[{batch_size}, {seq_len}]")

    if context.shape == (batch_size, input_dim):
        print("✓ FIXED AdaptiveAttention shape is correct!")
    else:
        print(f"✗ FIXED AdaptiveAttention shape is wrong: expected ({batch_size}, {input_dim}), got {context.shape}")

except Exception as e:
    print(f"✗ Error in FIXED AdaptiveAttention: {e}")
    import traceback

    traceback.print_exc()
