# Hyperparameter Tuning Guide

**Last Updated**: 2025-12-29
**Phase**: 4 Complete (Reward Engineering + Hyperparameter Tuning)

This guide provides systematic approaches to finding optimal hyperparameters for the RL trading system.

---

## Overview

Hyperparameter tuning optimizes:

1. **Position Sizing**: `risk_per_trade`, `max_position`, `use_dynamic_sizing`
2. **Reward Engineering**: `reward_type`, cost/drawdown penalties
3. **Risk Management**: `max_drawdown_pct`, stop-loss/take-profit
4. **RL Training**: `learning_rate`, `gamma`, `entropy_coef`

---

## Quick Start

### Run Tuning with Default Settings

```bash
# Random search (recommended for exploration)
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 50 \
  --episodes 10 \
  --grid-type conservative

# Grid search (exhaustive, slower)
python scripts/run_hyperparameter_tuning.py \
  --mode grid \
  --max-configs 30 \
  --grid-type position_sizing
```

Results are saved to `tuning_results/tuning_results.json` with detailed metrics.

---

## Tuning Strategies

### 1. Random Search (Recommended)

**When to use**: Exploring large search spaces quickly

**Advantages**:

- Efficient for high-dimensional spaces
- Can run in parallel
- Good coverage with fewer evaluations

**Example**:

```bash
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 100 \
  --episodes 10 \
  --episode-length 390 \
  --grid-type aggressive
```

### 2. Grid Search

**When to use**: Exhaustive search over small, focused parameter spaces

**Advantages**:

- Guarantees coverage of all combinations
- Easy to visualize results
- Good for 2-3 parameter searches

**Example**:

```bash
python scripts/run_hyperparameter_tuning.py \
  --mode grid \
  --max-configs 50 \
  --grid-type position_sizing
```

### 3. Focused Search (Two-Stage)

**Stage 1**: Find optimal position sizing

```bash
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 40 \
  --grid-type position_sizing
```

**Stage 2**: Optimize reward engineering with best position sizing

```bash
# Modify grid in script to use best risk_per_trade/max_position
# Then tune reward_type, penalties
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 30 \
  --grid-type conservative
```

---

## Predefined Grids

### Conservative Grid

**Target**: Production deployment with safety-first approach

**Parameters**:

- `risk_per_trade`: 1.0%, 1.5%, 2.0%
- `max_position`: 5, 10 lots
- `reward_type`: incremental_pnl, cost_aware
- `drawdown_penalty_weight`: 500, 1000
- `max_drawdown_pct`: 15%, 20%

**Grid size**: ~96 configurations

### Aggressive Grid

**Target**: Research and exploration with higher risk tolerance

**Parameters**:

- `risk_per_trade`: 1.5%, 2.0%, 2.5%, 3.0%
- `max_position`: 10, 15, 20 lots
- `reward_type`: incremental_pnl, cost_aware, sharpe
- `drawdown_penalty_weight`: 0, 1000, 2000
- `max_drawdown_pct`: 20%, 25%, 30%

**Grid size**: ~972 configurations (use random search!)

### Position Sizing Grid

**Target**: Focused optimization of risk parameters only

**Parameters**:

- `risk_per_trade`: 1.0%, 1.2%, 1.5%, 1.7%, 2.0%, 2.2%, 2.5%, 3.0%
- `max_position`: 5, 7.5, 10, 12.5, 15 lots
- All other params fixed at recommended defaults

**Grid size**: ~40 configurations (small enough for grid search)

---

## Metrics

Tuning framework tracks multiple performance metrics:

### Primary Metrics

1. **Total Return** (`total_return`)
    - Average return across all episodes
    - Formula: `(final_value - initial_cash) / initial_cash`
    - Higher is better

2. **Sharpe Ratio** (`sharpe_ratio`)
    - Risk-adjusted return
    - Formula: `mean(returns) / std(returns)`
    - Higher is better

3. **Max Drawdown** (`max_drawdown`)
    - Maximum portfolio decline from peak
    - Formula: `(peak - trough) / peak`
    - Lower is better

4. **Risk-Adjusted Return** (`risk_adjusted_return`)
    - Return penalized by drawdown
    - Formula: `total_return - (max_drawdown * 0.5)`
    - Balances return and risk
    - **Default optimization target**

### Secondary Metrics

- `total_trades`: Number of trades executed
- `avg_trade_size`: Average position size
- `total_commission`: Commission costs paid
- `total_spread`: Spread costs paid
- `final_portfolio_value`: Final portfolio value
- `episode_length`: Steps per episode

---

## Analyzing Results

### View Top Configurations

Results are automatically sorted by risk-adjusted return. The tuning script prints:

1. **Top 5 by Risk-Adjusted Return** (default optimization target)
2. **Top 5 by Total Return** (maximize profit)
3. **Top 5 by Sharpe Ratio** (maximize risk-adjusted performance)

### Manual Analysis

```python
import json

# Load results
with open('tuning_results/tuning_results.json') as f:
    data = json.load(f)

# Extract results
results = data['results']

# Sort by your preferred metric
sorted_by_return = sorted(results, key=lambda r: r['metrics']['total_return'], reverse=True)
sorted_by_sharpe = sorted(results, key=lambda r: r['metrics']['sharpe_ratio'], reverse=True)

# Print top config
best = sorted_by_return[0]
print(f"Best config: {best['config']}")
print(f"Return: {best['metrics']['total_return']:.2%}")
print(f"Sharpe: {best['metrics']['sharpe_ratio']:.3f}")
```

---

## Common Patterns

### High Returns with High Risk

**Characteristics**:

- High `risk_per_trade` (2.5-3%)
- Large `max_position` (15-20 lots)
- Low or zero `drawdown_penalty_weight`
- High `max_drawdown_pct` (25-30%)

**Use Case**: Research, backtesting, understanding limits

### Balanced Return/Risk

**Characteristics**:

- Moderate `risk_per_trade` (1.5-2%)
- Moderate `max_position` (10 lots)
- Moderate `drawdown_penalty_weight` (1000)
- Moderate `max_drawdown_pct` (20%)
- `reward_type`: "incremental_pnl" or "cost_aware"

**Use Case**: Production deployment (recommended)

### High Sharpe Ratio

**Characteristics**:

- Conservative `risk_per_trade` (1-1.5%)
- Smaller `max_position` (5-10 lots)
- High `drawdown_penalty_weight` (1500-2000)
- Low `max_drawdown_pct` (15%)
- `reward_type`: "sharpe" or "incremental_pnl"

**Use Case**: Risk-averse trading, live capital

---

## Tuning Recommendations

### 1. Start with Position Sizing

Position sizing has the highest impact on performance. Focus here first.

**Recommended approach**:

```bash
python scripts/run_hyperparameter_tuning.py \
  --mode grid \
  --grid-type position_sizing \
  --episodes 20
```

### 2. Then Optimize Rewards

Once you have optimal position sizing, tune reward engineering.

**Key parameters**:

- `reward_type`: Try incremental_pnl first, then cost_aware if overtrading
- `cost_penalty_weight`: 0-5 range
- `drawdown_penalty_weight`: 500-2000 range

### 3. Finally Tune RL Parameters

RL hyperparameters usually have lower impact than position sizing.

**Key parameters**:

- `learning_rate`: 1e-4 to 1e-3 (start with 3e-4)
- `gamma`: 0.95 to 0.99 (higher = more long-term focus)
- `entropy_coef`: 0.001 to 0.05 (higher = more exploration)

### 4. Validate on Held-Out Data

After tuning, validate top configs on different time periods or market conditions.

---

## Example: Complete Tuning Workflow

```bash
# Step 1: Find optimal position sizing (40 configs, 20 episodes each)
python scripts/run_hyperparameter_tuning.py \
  --mode grid \
  --grid-type position_sizing \
  --episodes 20 \
  --episode-length 390

# Step 2: Analyze results, identify best risk_per_trade and max_position
# (Manually check tuning_results/tuning_results.json)

# Step 3: Tune reward engineering with optimal position sizing
# (Edit script to fix position sizing params, vary reward params)
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 30 \
  --episodes 20

# Step 4: Validate top 3 configs on longer episodes
python scripts/run_hyperparameter_tuning.py \
  --mode random \
  --samples 3 \
  --episodes 50 \
  --episode-length 1000

# Step 5: Deploy best config to production training
```

---

## Interpreting Results

### Good Configuration Signs

✅ **Positive total return** across multiple episodes
✅ **Sharpe ratio > 1.0** (return exceeds volatility)
✅ **Max drawdown < 20%** (manageable risk)
✅ **Consistent performance** (low variance across episodes)
✅ **Reasonable trade frequency** (not overtrading)

### Red Flags

❌ **Negative Sharpe ratio** (returns don't justify risk)
❌ **Max drawdown > 30%** (excessive risk)
❌ **Very high trade frequency** (likely overtrading, high costs)
❌ **High variance** (unstable performance)
❌ **Zero trades** (agent not learning to trade)

---

## Advanced: Custom Grids

Create custom grids for specific tuning needs:

```python
from train.hyperparameter_tuning import HyperparameterGrid

# Example: Focus on reward engineering only
custom_grid = HyperparameterGrid(
    risk_per_trade=[0.02],  # Fixed
    max_position=[10.0],  # Fixed
    reward_type=["incremental_pnl", "cost_aware", "sharpe"],
    cost_penalty_weight=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    drawdown_penalty_weight=[0.0, 500.0, 1000.0, 1500.0, 2000.0],
    max_drawdown_pct=[0.20],  # Fixed
    enable_stop_loss=[False],  # Fixed
    learning_rate=[3e-4],  # Fixed
    gamma=[0.99],  # Fixed
    entropy_coef=[0.01],  # Fixed
)

# Grid size: 3 × 6 × 5 = 90 configurations
```

---

## Performance Considerations

### Tuning Speed

- **Single config**: ~1-2 seconds (10 episodes × 100 steps)
- **50 configs**: ~2-3 minutes
- **100 configs**: ~5-7 minutes
- **1000 configs**: ~50-70 minutes

### Parallelization

Future enhancement: Run multiple configs in parallel

```python
# Not yet implemented - future work
tuner.parallel_grid_search(grid, n_workers=8)
```

---

## Troubleshooting

### Issue: All configurations perform poorly

**Cause**: Base environment config may be too harsh
**Solution**: Check commission_pct, spread, max_drawdown_pct in base config

### Issue: Results show high variance

**Cause**: Not enough episodes per configuration
**Solution**: Increase `--episodes` to 20-50

### Issue: Grid search takes too long

**Cause**: Grid too large
**Solution**: Use random search with `--mode random --samples N`

---

## Next Steps

After finding optimal hyperparameters:

1. **Train full RL policy** with optimal config (Phase 5)
2. **Backtest on historical data** to validate performance
3. **Deploy to paper trading** before live capital
4. **Monitor performance** and retune quarterly

---

## Resources

- **Tuning Framework**: `train/hyperparameter_tuning.py`
- **Tuning Script**: `scripts/run_hyperparameter_tuning.py`
- **Results Format**: JSON with config + metrics
- **Related Docs**:
    - [Configuration Reference](../CONFIGURATION_REFERENCE.md)
    - [Phase 3 Quick Start](PHASE_3_QUICK_START.md)
    - [Architecture & API Reference](../api/ARCHITECTURE_API_REFERENCE.md)

---

**Last Updated**: 2025-12-29
**Maintainer**: Sequence FX Team
**Version**: Phase 4 Complete
