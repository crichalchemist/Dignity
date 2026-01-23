# Configuration Reference Guide

**Last Updated**: 2025-12-29
**Phase**: 3 Complete

This guide consolidates all configuration options for the Sequence FX trading system, with production-ready examples and
parameter tuning guidelines.

---

## Table of Contents

1. [Execution Environment Configuration](#execution-environment-configuration)
2. [Position Sizing Configuration](#position-sizing-configuration)
3. [RL Training Configuration](#rl-training-configuration)
4. [Production-Ready Configurations](#production-ready-configurations)
5. [Parameter Tuning Guidelines](#parameter-tuning-guidelines)
6. [Common Configuration Patterns](#common-configuration-patterns)

---

## Execution Environment Configuration

### ExecutionConfig

The `ExecutionConfig` dataclass controls the simulated retail execution environment with realistic market friction.

```python
from execution.simulated_retail_env import ExecutionConfig

config = ExecutionConfig(
    # Basic Setup
    initial_mid_price=100.0,      # Starting price for simulation
    initial_cash=50_000.0,         # Starting capital
    time_horizon=390,              # Steps per episode (390 = full trading day)
    lot_size=1.0,                  # Trading lot size

    # Market Microstructure
    spread=0.02,                   # Base bid-ask spread (dollars)
    price_drift=0.0,               # Trend component (0 = random walk)
    price_volatility=0.05,         # Price volatility (5%)

    # Execution Mechanics
    decision_lag=1,                # Steps between decision and execution
    limit_fill_probability=0.35,   # Probability of limit order fill
    limit_price_improvement=0.01,  # Price improvement on limit fills

    # Phase 3.1: Transaction Costs
    commission_per_lot=0.0,        # Fixed commission per lot (e.g., $7)
    commission_pct=0.0001,         # Commission as % of notional (0.01%)
    variable_spread=True,          # Enable volatility-based spread widening
    spread_volatility_multiplier=2.0,  # Spread multiplier during high volatility

    # Phase 3.3: Risk Management
    enable_stop_loss=False,        # Enable automatic stop-loss
    stop_loss_pct=0.02,            # Stop loss at 2% loss
    enable_take_profit=False,      # Enable automatic take-profit
    take_profit_pct=0.04,          # Take profit at 4% gain
    enable_drawdown_limit=True,    # Enable episode termination on drawdown
    max_drawdown_pct=0.20,         # Maximum 20% portfolio drawdown
)
```

### Key Parameters Explained

#### Transaction Costs

**`commission_per_lot`** (float, default: 0.0)

- Fixed commission per trading lot
- Example: $7 per 100k lot in FX retail trading
- Use when broker charges per-lot fees

**`commission_pct`** (float, default: 0.0)

- Commission as percentage of notional value
- Example: 0.0001 = 1 basis point = $10 on $100,000 trade
- Use when broker charges percentage-based fees

**`variable_spread`** (bool, default: False)

- Enable spread widening during high volatility
- Spread multiplies by `spread_volatility_multiplier` when volatility_ratio > 1.5
- Realistic behavior: liquidity providers widen quotes during turbulence

**`spread_volatility_multiplier`** (float, default: 2.0)

- Multiplier applied to base spread during high volatility
- Example: 0.02 base spread → 0.04 during volatility spike

#### Risk Management

**`enable_stop_loss`** (bool, default: False)

- Automatically close position when loss exceeds threshold
- Per-position risk control
- **Recommendation**: Disable for training (let agent learn), enable for safety during exploration

**`stop_loss_pct`** (float, default: 0.02)

- Stop loss threshold as % of entry price
- Example: 0.02 = close position if 2% loss from entry
- Calculated per-position, not portfolio-wide

**`enable_take_profit`** (bool, default: False)

- Automatically close position when profit exceeds threshold
- Teaches agent to book profits vs. letting winners run
- **Recommendation**: Disable for training (reward function handles this)

**`take_profit_pct`** (float, default: 0.04)

- Take profit threshold as % of entry price
- Example: 0.04 = close position if 4% gain from entry

**`enable_drawdown_limit`** (bool, default: False)

- Terminate episode if portfolio drawdown exceeds limit
- Portfolio-level risk control (across all positions)
- **Recommendation**: Enable for training (prevents catastrophic episodes)

**`max_drawdown_pct`** (float, default: 0.20)

- Maximum allowable drawdown from peak portfolio value
- Example: 0.20 = terminate if portfolio drops 20% from peak
- Acts as circuit breaker for training stability

#### Reward Engineering

**`reward_type`** (str, default: "incremental_pnl")

- Type of reward function to use
- Options: "portfolio_value", "incremental_pnl", "sharpe", "cost_aware"
- **portfolio_value**: Raw portfolio value (original, can cause training instability)
- **incremental_pnl**: Change in portfolio value (recommended for most use cases)
- **sharpe**: Risk-adjusted returns using rolling Sharpe-like metric
- **cost_aware**: Incremental PnL with explicit transaction cost penalties

**`cost_penalty_weight`** (float, default: 0.0)

- Weight for transaction cost penalty in reward calculation
- Only used with "cost_aware" reward type
- Example: 1.0 = each dollar of cost reduces reward by 1.0
- **Recommendation**: Start with 1.0-5.0 to discourage excessive trading

**`drawdown_penalty_weight`** (float, default: 0.0)

- Weight for drawdown penalty in reward calculation
- Used with "incremental_pnl" and "cost_aware" reward types
- Penalizes portfolio decline from peak
- Example: 1000.0 = 10% drawdown reduces reward by 100
- **Recommendation**: 500-2000 for moderate drawdown sensitivity

**`sharpe_window`** (int, default: 50)

- Window size for Sharpe ratio calculation
- Only used with "sharpe" reward type
- Number of recent steps to include in risk-adjusted return calculation
- **Recommendation**: 30-100 steps (balance responsiveness vs. stability)

**`reward_scaling`** (float, default: 1e-4)

- Scaling factor applied to all rewards
- Helps normalize reward magnitudes for training stability
- Portfolio values are large ($50k+), scaling brings them to reasonable range
- **Recommendation**: 1e-4 for portfolio_value, 1.0 for incremental rewards

---

## Position Sizing Configuration

### ActionConverter

The `ActionConverter` class controls how policy network outputs translate to position sizes.

```python
from train.core.env_based_rl_training import ActionConverter

converter = ActionConverter(
    lot_size=1.0,               # Base lot size for rounding
    max_position=10.0,          # Maximum inventory (long or short)
    risk_per_trade=0.02,        # Risk 2% of portfolio per trade
    use_dynamic_sizing=True,    # Enable portfolio-based position sizing
)
```

### Key Parameters Explained

**`lot_size`** (float, default: 1.0)

- Base lot size for position rounding
- All position sizes are multiples of this value
- Example: lot_size=0.5 → positions of 0.5, 1.0, 1.5, 2.0, ...

**`max_position`** (float, default: 10.0)

- Maximum absolute inventory (long or short)
- Hard constraint prevents catastrophic concentration
- **Per-pair limit** in multi-pair trading
- Example: max_position=10.0 → can hold max 10 lots long or 10 lots short

**`risk_per_trade`** (float, default: 0.02)

- Fraction of portfolio to risk per trade
- Used when `use_dynamic_sizing=True`
- Formula: `size = (portfolio_value * risk_per_trade) / mid_price`
- Example: 0.02 = risk 2% of portfolio, Kelly-criterion-inspired

**`use_dynamic_sizing`** (bool, default: False)

- Enable position sizing based on portfolio value
- When True: position size scales with portfolio growth/shrinkage
- When False: fixed `lot_size` positions
- **Recommendation**: Enable for realistic training (prevents compounding losses)

---

## RL Training Configuration

### RLTrainingConfig

Controls the reinforcement learning training process (from `config/config.py`).

```python
from config.config import RLTrainingConfig

rl_config = RLTrainingConfig(
    epochs=100,                # Number of training epochs
    learning_rate=1e-4,        # Policy optimizer learning rate
    discount_factor=0.99,      # Gamma for future reward discounting
    entropy_coef=0.01,         # Entropy bonus for exploration
    value_loss_coef=0.5,       # Weight for value function loss
    max_grad_norm=0.5,         # Gradient clipping threshold
)
```

### Key Parameters Explained

**`learning_rate`** (float, default: 1e-4)

- Step size for policy gradient updates
- Smaller = more stable, slower learning
- Larger = faster learning, risk of instability
- **Recommendation**: Start with 1e-4, reduce if unstable

**`discount_factor`** (float, default: 0.99)

- Gamma parameter for future reward discounting
- Higher = agent values long-term rewards more
- Example: 0.99 = rewards 100 steps away are worth 36.6% of immediate reward
- **Recommendation**: 0.95-0.99 for trading (balance short/long-term)

**`entropy_coef`** (float, default: 0.01)

- Exploration bonus coefficient
- Higher = more exploration, more random actions
- Lower = more exploitation, greedier policy
- **Recommendation**: Start with 0.01, increase if agent gets stuck in local optimum

---

## Production-Ready Configurations

### Conservative Training (High Safety)

For initial training or high-risk capital:

```python
# Maximum safety, minimal risk exposure
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,         # 1 basis point
    variable_spread=True,
    enable_stop_loss=True,         # Hard stops enabled
    stop_loss_pct=0.01,            # Tight 1% stop
    enable_take_profit=True,       # Book profits early
    take_profit_pct=0.02,          # 2% profit target
    enable_drawdown_limit=True,
    max_drawdown_pct=0.10,         # Conservative 10% max drawdown
)

converter = ActionConverter(
    max_position=5.0,              # Lower position limit
    risk_per_trade=0.01,           # Conservative 1% risk
    use_dynamic_sizing=True,
)
```

**Use Case**: First-time training, high-value accounts, low risk tolerance

### Aggressive Training (Low Constraints)

For research and exploration:

```python
# Let agent learn organically, minimal constraints
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,
    variable_spread=True,
    enable_stop_loss=False,        # No hard stops - agent learns risk
    enable_take_profit=False,      # No forced profit-taking
    enable_drawdown_limit=True,    # Only portfolio-level safety
    max_drawdown_pct=0.30,         # Loose 30% max drawdown
)

converter = ActionConverter(
    max_position=20.0,             # Higher position limit
    risk_per_trade=0.03,           # Aggressive 3% risk
    use_dynamic_sizing=True,
)
```

**Use Case**: Research, experimentation, high-risk tolerance, exploration phase

### Production-Like (Recommended)

Balanced approach for production deployment:

```python
# Realistic market conditions, moderate safety
config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,         # 1 basis point (typical FX)
    variable_spread=True,
    spread_volatility_multiplier=2.5,  # Realistic spread widening
    enable_stop_loss=False,        # Let agent learn optimal stops
    enable_take_profit=False,      # Let agent learn profit-taking
    enable_drawdown_limit=True,    # Portfolio-level safety net
    max_drawdown_pct=0.20,         # 20% max drawdown

    # Phase 4: Reward engineering
    reward_type="incremental_pnl",  # Stable training with incremental rewards
    drawdown_penalty_weight=1000.0,  # Discourage large drawdowns
    reward_scaling=1.0,  # No scaling needed for incremental rewards
)

converter = ActionConverter(
    max_position=10.0,             # Moderate position limit
    risk_per_trade=0.015,          # Conservative 1.5% risk
    use_dynamic_sizing=True,       # Portfolio-adaptive sizing
)

rl_config = RLTrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    discount_factor=0.98,          # Balanced short/long-term focus
    entropy_coef=0.01,
)
```

**Use Case**: Production deployment, live capital, balanced risk/reward

### Multi-Pair Portfolio Training

For training across multiple FX pairs:

```python
# Per-pair configuration (apply to each pair independently)
config = ExecutionConfig(
    initial_cash=50_000.0,         # Shared cash pool in production
    commission_pct=0.0001,
    variable_spread=True,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.25,         # Portfolio-level limit
)

converter = ActionConverter(
    max_position=8.0,              # Per-pair position limit
    risk_per_trade=0.012,          # 1.2% per pair (8 pairs × 1.2% = ~10% total)
    use_dynamic_sizing=True,
)

# Training loop
for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", ...]:
    env = SimulatedRetailExecutionEnv(config)
    # Train on pair-specific data
```

**Use Case**: Multi-pair trading, portfolio diversification, cross-pair learning

---

## Parameter Tuning Guidelines

### Transaction Cost Calibration

**Commission**:

- **Retail FX**: 0.5-1.5 pips (commission_pct = 0.00005 to 0.00015)
- **ECN brokers**: ~$5-7 per 100k lot (commission_per_lot = 5.0 to 7.0)
- **Crypto**: 0.1-0.5% (commission_pct = 0.001 to 0.005)

**Spreads**:

- **EUR/USD**: 0.5-2 pips (spread = 0.00005 to 0.0002)
- **GBP/JPY**: 2-5 pips (spread = 0.002 to 0.005)
- **Exotics**: 5-20 pips (spread = 0.005 to 0.020)

### Position Sizing Tuning

**Risk per trade** (`risk_per_trade`):

- **Conservative**: 0.5-1% (0.005 to 0.01)
- **Moderate**: 1-2% (0.01 to 0.02)
- **Aggressive**: 2-5% (0.02 to 0.05)
- **Rule of thumb**: Total exposure across all pairs should not exceed 10-20% of portfolio

**Maximum position** (`max_position`):

- **High volatility pairs** (GBP/JPY): Lower limits (5-8 lots)
- **Low volatility pairs** (EUR/USD): Higher limits (10-15 lots)
- **Total exposure**: Sum across all pairs should align with portfolio risk tolerance

### Risk Management Tuning

**Drawdown limit** (`max_drawdown_pct`):

- **Conservative**: 10-15% (0.10 to 0.15)
- **Moderate**: 15-25% (0.15 to 0.25)
- **Aggressive**: 25-40% (0.25 to 0.40)
- **Align with**: Account size, risk tolerance, recovery time

**Stop-loss** (`stop_loss_pct`):

- **Tight stops**: 0.5-1% (0.005 to 0.01) - high-frequency strategies
- **Moderate stops**: 1-2% (0.01 to 0.02) - swing trading
- **Wide stops**: 2-5% (0.02 to 0.05) - position trading
- **Pair volatility**: Use tighter stops for low volatility, wider for high volatility

---

## Reward Engineering Configurations

### Incremental PnL with Drawdown Penalty (Recommended)

```python
config = ExecutionConfig(
    reward_type="incremental_pnl",
    drawdown_penalty_weight=1000.0,  # Discourage drawdowns
    reward_scaling=1.0,
)
```

**When to use**: Default for most training scenarios. Provides stable training with explicit drawdown discouragement.

### Cost-Aware Reward (Discourage Overtrading)

```python
config = ExecutionConfig(
    reward_type="cost_aware",
    cost_penalty_weight=3.0,  # Penalize transaction costs
    drawdown_penalty_weight=500.0,  # Also penalize drawdowns
    reward_scaling=1.0,
)
```

**When to use**: When agent tends to overtrade. Explicitly penalizes commissions, spreads, and slippage.

### Sharpe Ratio (Risk-Adjusted Returns)

```python
config = ExecutionConfig(
    reward_type="sharpe",
    sharpe_window=50,  # 50-step rolling window
    reward_scaling=1.0,
)
```

**When to use**: When optimizing for risk-adjusted performance. Best for longer training runs where volatility can be
measured.

### Portfolio Value (Baseline)

```python
config = ExecutionConfig(
    reward_type="portfolio_value",
    reward_scaling=1e-4,  # Important: scale down large values
)
```

**When to use**: Testing/comparison only. Not recommended for training due to potential instability.

---

## Common Configuration Patterns

### Pattern 1: Disable All Risk Controls (Pure RL Learning)

```python
config = ExecutionConfig(
    enable_stop_loss=False,
    enable_take_profit=False,
    enable_drawdown_limit=False,  # Warning: No safety net!
)
```

**When to use**: Research, controlled environment, understanding agent behavior
**Risk**: Agent can suffer catastrophic losses during exploration

### Pattern 2: Portfolio-Level Safety Only

```python
config = ExecutionConfig(
    enable_stop_loss=False,
    enable_take_profit=False,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.20,
)
```

**When to use**: Production training, balanced risk/learning
**Benefit**: Agent learns position-level risk management, portfolio protected

### Pattern 3: Hard Stops for Exploration

```python
config = ExecutionConfig(
    enable_stop_loss=True,
    stop_loss_pct=0.02,
    enable_take_profit=False,
    enable_drawdown_limit=True,
)
```

**When to use**: Early training, preventing runaway losses
**Risk**: Agent may learn to "game" stops instead of proper risk management

### Pattern 4: Graduated Risk Reduction

```python
# Phase 1: Exploration (high risk tolerance)
config_phase1 = ExecutionConfig(max_drawdown_pct=0.30, enable_stop_loss=False)

# Phase 2: Refinement (moderate risk)
config_phase2 = ExecutionConfig(max_drawdown_pct=0.20, enable_stop_loss=False)

# Phase 3: Production (conservative risk)
config_phase3 = ExecutionConfig(max_drawdown_pct=0.15, enable_stop_loss=True, stop_loss_pct=0.02)
```

**When to use**: Curriculum learning, staged deployment
**Benefit**: Agent learns progressively with increasing constraints

---

## Configuration Validation Checklist

Before training:

- [ ] Commission settings match broker fees
- [ ] Spread settings match typical pair spreads
- [ ] Position limits align with account size and risk tolerance
- [ ] Drawdown limit set (recommended: enable for all training)
- [ ] Risk per trade ≤ 2% for moderate risk, ≤ 5% for aggressive
- [ ] Total exposure across pairs ≤ 20% of portfolio
- [ ] Variable spread enabled for realistic market conditions
- [ ] Lot size matches broker requirements
- [ ] Time horizon appropriate for strategy (390 steps = full day)

---

## Example: Complete Training Setup

```python
from execution.simulated_retail_env import ExecutionConfig, SimulatedRetailExecutionEnv
from train.core.env_based_rl_training import ActionConverter
from config.config import RLTrainingConfig

# 1. Execution environment (realistic FX conditions)
env_config = ExecutionConfig(
    initial_cash=50_000.0,
    lot_size=1.0,
    spread=0.00015,                # 1.5 pips for EUR/USD
    commission_pct=0.00007,        # 0.7 pips (typical retail FX)
    variable_spread=True,
    spread_volatility_multiplier=2.0,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.20,
    time_horizon=390,              # Full trading day
)

# 2. Position sizing (2% risk, Kelly-inspired)
action_converter = ActionConverter(
    lot_size=1.0,
    max_position=10.0,
    risk_per_trade=0.02,
    use_dynamic_sizing=True,
)

# 3. RL training (conservative learning rate)
rl_config = RLTrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    discount_factor=0.98,
    entropy_coef=0.01,
)

# 4. Initialize environment
env = SimulatedRetailExecutionEnv(env_config, seed=42)

# 5. Train!
# (See ARCHITECTURE_API_REFERENCE.md for full training loop)
```

---

## Next Steps

- **Quick Start**: See [Phase 3 Quick Start Guide](guides/PHASE_3_QUICK_START.md)
- **Implementation Details**: See [Phase 3 Implementation Summary](implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)
- **Architecture Overview**: See [Architecture & API Reference](ARCHITECTURE_API_REFERENCE.md)
- **Testing & Validation**: See [Testing & Validation Report](TESTING_VALIDATION_REPORT.md)

---

**Last Updated**: 2025-12-29
**Maintainer**: Sequence FX Team
**Version**: Phase 3 Complete
