# Phase 3 Quick Start Guide

**Last Updated**: 2025-12-29
**Est. Time**: 5-10 minutes

Get started with Phase 3 features (transaction costs, position sizing, and risk management) in under 10 minutes.

---

## What's in Phase 3?

Phase 3 adds production-ready features to the RL trading system:

✅ **Transaction Costs**: Commission, spreads, slippage tracking
✅ **Position Sizing**: Dynamic Kelly-criterion-inspired sizing
✅ **Risk Management**: Stop-loss, take-profit, drawdown limits

---

## Quick Setup (3 Steps)

### Step 1: Configure Execution Environment

```python
from execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv

config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,         # 1 basis point commission
    variable_spread=True,          # Enable volatility-based spread widening
    enable_drawdown_limit=True,    # Portfolio safety net
    max_drawdown_pct=0.20,         # 20% max drawdown
)

env = SimulatedRetailExecutionEnv(config, seed=42)
```

### Step 2: Configure Position Sizing

```python
from train.core.env_based_rl_training import ActionConverter

converter = ActionConverter(
    lot_size=1.0,
    max_position=10.0,             # Max 10 lots per pair
    risk_per_trade=0.02,           # Risk 2% per trade
    use_dynamic_sizing=True,       # Enable Kelly-inspired sizing
)
```

### Step 3: Run a Training Episode

```python
# Simple training episode
obs = env.reset()
done = False

while not done:
    # Get action from policy (simplified example)
    action_idx = 2  # BUY (0=SELL, 1=HOLD, 2=BUY)

    # Convert to order with position sizing
    order = converter.policy_to_order(
        action_idx,
        obs['mid_price'],
        obs['inventory'],
        obs['cash']
    )

    # Execute in environment
    obs, reward, done, info = env.step(order)

# Check results
print(f"Final Portfolio: ${obs['portfolio_value']:,.2f}")
print(f"Total Costs: ${env._commission_paid + env._spread_paid:.2f}")
```

**Done!** You're now using Phase 3 features.

---

## Validation Checklist

Verify Phase 3 is working correctly:

- [ ] Transaction costs are tracked: `env._commission_paid > 0`
- [ ] Position sizes scale with portfolio: Larger portfolio → larger positions
- [ ] Drawdown limit terminates episodes: Episode ends if drawdown exceeds limit
- [ ] Variable spreads widen during volatility: Check `env._spread_paid` variation

---

## Common Configurations

### Minimal (No Risk Controls)

```python
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,
    # All risk controls disabled - pure RL learning
)
```

### Balanced (Recommended)

```python
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,
    variable_spread=True,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.20,
)

converter = ActionConverter(
    max_position=10.0,
    risk_per_trade=0.02,
    use_dynamic_sizing=True,
)
```

### Conservative (High Safety)

```python
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,
    variable_spread=True,
    enable_stop_loss=True,
    stop_loss_pct=0.02,            # 2% stop
    enable_take_profit=True,
    take_profit_pct=0.04,          # 4% target
    enable_drawdown_limit=True,
    max_drawdown_pct=0.10,         # 10% max drawdown
)

converter = ActionConverter(
    max_position=5.0,              # Lower limit
    risk_per_trade=0.01,           # 1% risk
    use_dynamic_sizing=True,
)
```

---

## Key Metrics to Monitor

After running episodes, check these metrics:

```python
# Transaction costs
print(f"Commission: ${env._commission_paid:.2f}")
print(f"Spread: ${env._spread_paid:.2f}")
print(f"Slippage: ${env._slippage_paid:.2f}")
print(f"Total: ${env._commission_paid + env._spread_paid + env._slippage_paid:.2f}")

# Position sizing
print(f"Final Inventory: {env.inventory} lots")
print(f"Fill Events: {len(env._fill_events)}")

# Risk management
print(f"Stop-Loss Triggered: {env._stop_loss_triggered} times")
print(f"Take-Profit Triggered: {env._take_profit_triggered} times")
print(f"Peak Portfolio: ${env._peak_portfolio_value:,.2f}")
```

---

## Troubleshooting

### Issue: "Position sizes don't scale with portfolio"

**Solution**: Ensure `use_dynamic_sizing=True` in ActionConverter

```python
converter = ActionConverter(use_dynamic_sizing=True)  # ✅ Correct
```

### Issue: "Episodes terminate early"

**Cause**: Drawdown limit exceeded
**Solution**: Check drawdown setting or disable limit

```python
config = ExecutionConfig(
    enable_drawdown_limit=True,
    max_drawdown_pct=0.30,  # Increase tolerance
)
```

### Issue: "Transaction costs are zero"

**Cause**: Commission and commission_pct both set to 0.0
**Solution**: Set one of the commission fields

```python
config = ExecutionConfig(
    commission_pct=0.0001,  # ✅ Set commission
)
```

### Issue: "Spread doesn't widen during volatility"

**Cause**: `variable_spread=False` (default)
**Solution**: Enable variable spreads

```python
config = ExecutionConfig(
    variable_spread=True,  # ✅ Enable
)
```

---

## Next Steps

### Beginner

1. Run example above to validate Phase 3 works
2. Experiment with different `risk_per_trade` values (0.01, 0.02, 0.05)
3. Compare episodes with/without risk controls

### Intermediate

1. Review full configuration options: [Configuration Reference](../CONFIGURATION_REFERENCE.md)
2. Read implementation details: [Phase 3 Implementation Summary](../implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)
3. Explore multi-pair training patterns

### Advanced

1. Design custom reward functions incorporating transaction costs
2. Tune hyperparameters (risk_per_trade, max_position, drawdown limits)
3. Implement portfolio-level position sizing across multiple pairs

---

## Testing Your Setup

Quick validation test:

```python
import numpy as np

# Test 1: Transaction costs accumulate
env.reset()
for _ in range(10):
    order = OrderAction(action_type="market", side="buy", size=1.0)
    env.step(order)
assert env._commission_paid > 0, "Commission should accumulate"

# Test 2: Position sizing scales with portfolio
converter = ActionConverter(use_dynamic_sizing=True, risk_per_trade=0.02)

size_10k = converter._calculate_position_size("buy", 0, 100, 10_000)
size_50k = converter._calculate_position_size("buy", 0, 100, 50_000)
assert size_50k > size_10k, "Position size should scale with portfolio"

# Test 3: Drawdown limit terminates episode
config = ExecutionConfig(enable_drawdown_limit=True, max_drawdown_pct=0.05)
env = SimulatedRetailExecutionEnv(config)
# ... run losing trades ...
# Episode should terminate when drawdown exceeds 5%

print("✅ All tests passed!")
```

---

## Configuration Templates

Copy-paste ready configurations:

### FX Day Trading

```python
config = ExecutionConfig(
    initial_cash=50_000.0,
    time_horizon=390,              # Full trading day
    spread=0.00015,                # 1.5 pips (EUR/USD)
    commission_pct=0.00007,        # 0.7 pips
    variable_spread=True,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.20,
)

converter = ActionConverter(
    max_position=10.0,
    risk_per_trade=0.02,
    use_dynamic_sizing=True,
)
```

### Crypto Trading (Higher Costs)

```python
config = ExecutionConfig(
    initial_cash=10_000.0,
    spread=0.10,                   # 0.1% spread
    commission_pct=0.001,          # 0.1% taker fee
    variable_spread=True,
    spread_volatility_multiplier=3.0,  # Crypto = high volatility
    enable_drawdown_limit=True,
    max_drawdown_pct=0.30,         # Higher volatility tolerance
)

converter = ActionConverter(
    max_position=5.0,              # Lower due to volatility
    risk_per_trade=0.03,           # Higher risk for higher returns
    use_dynamic_sizing=True,
)
```

---

## Resources

- **Full Configuration Guide**: [Configuration Reference](../CONFIGURATION_REFERENCE.md)
- **Implementation Details**: [Phase 3 Implementation Summary](../implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)
- **Testing & Validation**: [Testing & Validation Report](../TESTING_VALIDATION_REPORT.md)
- **Architecture Overview**: [Architecture & API Reference](../ARCHITECTURE_API_REFERENCE.md)

---

## Quick Reference

| Feature          | Config Parameter        | Default | Recommended     |
|------------------|-------------------------|---------|-----------------|
| Commission (%)   | `commission_pct`        | 0.0     | 0.0001 (FX)     |
| Variable Spreads | `variable_spread`       | False   | True            |
| Drawdown Limit   | `enable_drawdown_limit` | False   | True (training) |
| Max Drawdown     | `max_drawdown_pct`      | 0.20    | 0.15-0.25       |
| Risk per Trade   | `risk_per_trade`        | 0.02    | 0.015-0.02      |
| Dynamic Sizing   | `use_dynamic_sizing`    | False   | True            |
| Max Position     | `max_position`          | 10.0    | 5-15 (by pair)  |

---

**Ready to train?** Run your first Phase 3 episode and monitor the metrics above!

**Questions?** See [troubleshooting](#troubleshooting) or review
the [Configuration Reference](../CONFIGURATION_REFERENCE.md).

---

**Last Updated**: 2025-12-29
**Maintainer**: Sequence FX Team
**Version**: Phase 3 Complete
