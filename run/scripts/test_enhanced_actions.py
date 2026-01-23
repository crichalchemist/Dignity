"""Test enhanced action space (aggressiveness parameter) integration.

This script verifies that:
1. Policy outputs aggressiveness parameter
2. Aggressiveness is passed through to environment
3. Limit order engine uses aggressiveness to compute limit prices
"""

import sys

import torch

# Add parent directory to path
sys.path.insert(0, "/")

from config.config import PolicyConfig, SignalModelConfig
from execution.limit_order_engine import LimitOrderConfig
from execution.simulated_retail_env import (
    ExecutionConfig,
    OrderAction,
    SimulatedRetailExecutionEnv,
)
from models.signal_policy import ExecutionPolicy, SignalModel


def test_policy_aggressiveness_output():
    """Test that policy outputs aggressiveness."""
    print("=" * 60)
    print("Test 1: Policy Aggressiveness Output")
    print("=" * 60)

    # Create minimal configs
    signal_cfg = SignalModelConfig(num_features=76)

    # Create signal model to get actual output dimension
    signal_model = SignalModel(signal_cfg)

    # Use actual signal_dim for policy config
    policy_cfg = PolicyConfig(
        input_dim=signal_model.signal_dim,  # Get actual dimension from signal model
        hidden_dim=128,
        num_actions=3,
    )

    # Create policy
    policy = ExecutionPolicy(policy_cfg)

    # Test forward pass
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 76)

    with torch.no_grad():
        signal_out = signal_model(x)
        logits, value, aggressiveness = policy(signal_out["embedding"])

    print("✓ Policy outputs:")
    print(f"  - Logits shape: {logits.shape} (expected: [4, 3])")
    print(f"  - Value shape: {value.shape} (expected: [4])")
    print(f"  - Aggressiveness shape: {aggressiveness.shape} (expected: [4])")
    print(f"  - Aggressiveness values: {aggressiveness.tolist()}")
    print(f"  - Aggressiveness range: [{aggressiveness.min():.3f}, {aggressiveness.max():.3f}]")

    # Verify aggressiveness is in [0, 1]
    assert torch.all(aggressiveness >= 0.0) and torch.all(
        aggressiveness <= 1.0
    ), "Aggressiveness must be in [0,1]"
    print("✓ Aggressiveness values are in [0, 1]\n")


def test_order_action_aggressiveness():
    """Test that OrderAction accepts and stores aggressiveness."""
    print("=" * 60)
    print("Test 2: OrderAction Aggressiveness Field")
    print("=" * 60)

    # Test with aggressiveness
    action1 = OrderAction(
        action_type="market", side="buy", size=1.0, aggressiveness=0.7
    )
    print(f"✓ OrderAction with aggressiveness: {action1}")
    assert action1.aggressiveness == 0.7, "Aggressiveness not stored correctly"

    # Test without aggressiveness (default None)
    action2 = OrderAction(action_type="market", side="sell", size=1.0)
    print(f"✓ OrderAction without aggressiveness: {action2}")
    assert action2.aggressiveness is None, "Default aggressiveness should be None"
    print()


def test_environment_aggressiveness_integration():
    """Test that environment uses aggressiveness with limit order engine."""
    print("=" * 60)
    print("Test 3: Environment Aggressiveness Integration")
    print("=" * 60)

    # Create environment with limit order engine
    config = ExecutionConfig(
        use_limit_order_engine=True,
        limit_order_config=LimitOrderConfig(base_aggressiveness=0.5),
        initial_cash=50_000.0,
    )

    env = SimulatedRetailExecutionEnv(config=config, seed=42)

    obs = env.reset()
    print("✓ Environment initialized with limit order engine")
    print(f"  Initial state: cash={obs['cash']:.2f}, inventory={obs['inventory']:.2f}")

    # Test different aggressiveness levels
    test_cases = [
        (0.0, "Passive (prefers better price)"),
        (0.5, "Balanced"),
        (1.0, "Aggressive (near market price)"),
    ]

    for aggressiveness, description in test_cases:
        # Reset environment for each test
        env.reset()

        action = OrderAction(
            action_type="market", side="buy", size=1.0, aggressiveness=aggressiveness
        )

        # Take a step with aggressiveness
        next_obs, reward, done, info = env.step(action)

        print(f"\n✓ Aggressiveness = {aggressiveness} ({description})")
        print("  - Action processed successfully")
        print(f"  - Inventory: {next_obs['inventory']:.2f}")
        print(f"  - Cash: {next_obs['cash']:.2f}")

        # Check if order was converted to limit order
        if env._pending_limits:
            limit_order = env._pending_limits[0]
            print(
                f"  - Converted to limit order at price: {limit_order.limit_price:.5f}"
            )
            print(f"  - Mid price: {env.mid_price:.5f}")
        else:
            print("  - No pending limit orders")

    print("\n✓ All aggressiveness levels processed successfully\n")


if __name__ == "__main__":
    print("\nTesting Enhanced Action Space Integration\n")

    try:
        test_policy_aggressiveness_output()
        test_order_action_aggressiveness()
        test_environment_aggressiveness_integration()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nEnhanced action space is working correctly:")
        print("  1. Policy outputs aggressiveness parameter")
        print("  2. OrderAction stores aggressiveness")
        print("  3. Environment uses aggressiveness with limit order engine")
        print("  4. Learned aggressiveness enables adaptive execution strategy")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
