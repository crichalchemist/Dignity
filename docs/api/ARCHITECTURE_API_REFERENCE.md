# Architecture & API Reference

**Date:** 2025-12-29 (Updated for Phase 3)
**Version:** 2.0
**Status:** Complete implementation (Phases 1-3, includes transaction costs, position sizing, risk management)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module Reference](#module-reference)
3. [Training Workflow](#training-workflow)
4. [RL Pipeline](#rl-pipeline)
5. [Configuration Guide](#configuration-guide)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                           │
│  [Raw OHLCV] → [Validate] → [Features] → [Normalize]       │
│                                                              │
│  • download_gdelt.py (HTTPS + checksums)                   │
│  • download_all_fx_data.py (bounded downloads)             │
│  • prepare_dataset.py (validation + features + splits)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  Model Architecture                          │
│  [Shared Encoder] → [Task Heads]                            │
│                                                              │
│  • SharedEncoder: CNN + LSTM + Attention (unified base)     │
│  • PriceSequenceEncoder: Price-specific variant             │
│  • SignalBackbone: Signal-specific variant                  │
│  • DignityModel: Multi-head supervised learning             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   [Supervised]   [RL Policy]   [Execution]
   Training       Training      Environment
```

### Data Flow: Supervised Training

```
[DataLoaders]
    ↓
[train_model()] or [train_multitask()]
    ├─ Forward pass: x → model(x) → outputs
    ├─ Compute losses: direction, return, volatility, etc.
    ├─ Backward pass: loss.backward()
    ├─ Risk gating: optional logit clamping
    └─ Checkpoint: EarlyStopping, CheckpointManager
    ↓
[eval/run_evaluation.py]
    ├─ Classification metrics: accuracy, precision, recall, F1
    ├─ Regression metrics: RMSE, MAE
    └─ Confusion matrix + per-regime performance
```

### Data Flow: RL Training

```
[Environment Factory]
    ├─ SimulatedRetailExecutionEnv
    ├─ BacktestingEnv (optional)
    └─ HistData or synthetic
    ↓
[A3CAgent.train()]
    ├─ Worker pool (n_workers processes)
    ├─ Each worker: collect rollouts → compute GAE → update policy
    ├─ Global model synchronized across workers
    └─ Async checkpoint saving
    ↓
[eval_a3c_agent()]
    ├─ Greedy rollout on test environment
    ├─ Metrics: reward, Sharpe ratio, win rate
    └─ Policy comparison
```

---

## Module Reference

### Core Modules

| Module                             | Purpose                   | Key Classes/Functions                                       |
|------------------------------------|---------------------------|-------------------------------------------------------------|
| `config/config.py`                 | Unified dataclass configs | `ModelConfig`, `TrainingConfig`, `DataConfig`               |
| `data/agents/base_agent.py`        | Base data pipeline        | `BaseDataAgent`, `SequenceDataset`, `ensure_utc_timezone()` |
| `data/agents/single_task_agent.py` | Single-task data flow     | `SingleTaskDataAgent`                                       |
| `data/agents/multitask_agent.py`   | Multi-task data flow      | `MultiTaskDataAgent`                                        |
| `data/prepare_dataset.py`          | Entry point for data prep | `process_pair()`, `validate_dataframe()`                    |
| `models/agent_hybrid.py`           | Hybrid architectures      | `SharedEncoder`, `PriceSequenceEncoder`, `DignityModel`     |
| `models/signal_policy.py`          | Signal + policy models    | `SignalBackbone`, `SignalModel`, `ExecutionPolicy`          |
| `train/core/agent_train.py`        | Training loops            | `train_model()`, `train_multitask()`, `_evaluate()`         |
| `train/run_training.py`            | Supervised training CLI   | `main()`                                                    |
| `rl/agents/a3c_agent.py`           | A3C implementation        | `A3CAgent`, `ActorCriticNetwork`                            |
| `rl/run_a3c_training.py`           | RL training CLI           | `main()`                                                    |
| `eval/run_evaluation.py`           | Evaluation CLI            | `main()`                                                    |
| `eval/agent_eval.py`               | Evaluation functions      | `evaluate_a3c_agent()`                                      |
| `risk/risk_manager.py`             | Risk gating               | `RiskManager`                                               |

### Utility Modules

| Module                            | Purpose               | Key Classes/Functions                             |
|-----------------------------------|-----------------------|---------------------------------------------------|
| `utils/seed.py`                   | Reproducibility       | `set_seed()` (returns seed value)                 |
| `utils/logger.py`                 | Logging               | `get_logger()`                                    |
| `utils/training_checkpoint.py`    | Checkpoint management | `EarlyStopping`, `CheckpointManager`              |
| `utils/amp.py`                    | Mixed precision       | `AMPTrainer`                                      |
| `utils/async_checkpoint_saver.py` | Async I/O             | `AsyncCheckpointSaver`                            |
| `utils/attention_optimization.py` | Optimized attention   | `TemporalAttention`, `MultiHeadTemporalAttention` |
| `utils/datetime_utils.py`         | Timezone handling     | `convert_to_utc_and_dedup()`                      |

---

## Training Workflow

### Step 1: Data Preparation

```bash
# Single pair, supervised
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 --t-out 10 \
  --task-type classification

# Multi-pair, with sentiment features
python data/prepare_dataset.py \
  --pairs gbpusd,eurusd,usdjpy \
  --include-sentiment \
  --intrinsic-time \
  --dc-threshold-up 0.001
```

**Outputs:**

- `data/cache/gbpusd_features_cache.csv` (features)
- `data/cache/gbpusd_splits.npz` (train/val/test indices)

### Step 2: Supervised Training

```bash
# Basic training (CPU)
python train/run_training.py \
  --pairs gbpusd \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 1e-3

# GPU with AMP and DataLoader tuning
python train/run_training.py \
  --pairs gbpusd \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --device cuda \
  --use-amp \
  --num-workers 4 \
  --pin-memory \
  --prefetch-factor 4 \
  --checkpoint-path models/gbpusd_best.pt
```

**Outputs:**

- `models/gbpusd_best.pt` (best checkpoint)

### Step 3: Evaluation

```bash
python eval/run_evaluation.py \
  --pairs gbpusd \
  --checkpoint-path models/gbpusd_best.pt
```

**Outputs:**

- Accuracy, F1, confusion matrix (classification)
- RMSE, MAE (regression)
- Per-regime performance

---

## RL Pipeline

### Step 1: RL Policy Training

**Option A: Stochastic Simulation (Default)**

```bash
# Train A3C agent with stochastic retail execution simulation
python rl/run_a3c_training.py \
  --pair gbpusd \
  --num-workers 4 \
  --total-steps 100000 \
  --learning-rate 1e-4 \
  --entropy-coef 0.01 \
  --device cuda \
  --checkpoint-path models/a3c_gbpusd.pt
```

**Option B: Deterministic Backtesting (Historical Replay)**

```bash
# Train A3C agent with deterministic backtesting.py historical replay
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 4 \
  --total-steps 100000 \
  --learning-rate 1e-4 \
  --entropy-coef 0.01 \
  --device cuda \
  --checkpoint-path models/a3c_gbpusd_bt.pt
```

**Key Differences:**

- **Simulated mode** (`--env-mode simulated`): Stochastic execution with spread, slippage, realistic fills (
  non-deterministic, better for exploration)
- **Backtesting mode** (`--env-mode backtesting`): Deterministic bar-by-bar replay using backtesting.py (reproducible,
  good for validation)

**Outputs:**

- `models/a3c_gbpusd.pt` (trained policy)

### Step 2: RL Policy Evaluation

```python
from rl.agents.a3c_agent import A3CAgent, A3CConfig
from config.config import ModelConfig
from eval.agent_eval import evaluate_a3c_agent
from execution.simulated_retail_env import SimulatedRetailExecutionEnv

# Load trained model
model_cfg = ModelConfig(num_features=20)
a3c_cfg = A3CConfig(checkpoint_path="models/a3c_gbpusd.pt")
agent = A3CAgent(model_cfg, a3c_cfg, action_dim=3, env_factory=lambda: SimulatedRetailExecutionEnv())

def env_factory():
    return SimulatedRetailExecutionEnv(pair="gbpusd", initial_balance=10000)

metrics = evaluate_a3c_agent(agent, env_factory, num_episodes=100)
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

---

## Configuration Guide

### DataConfig

```python
@dataclass
class DataConfig:
    csv_path: str                      # Path to OHLCV CSV
    t_in: int = 120                   # Lookback window (bars)
    t_out: int = 10                   # Forecast horizon (bars)
    target_type: str = "classification"  # "classification" or "regression"
    flat_threshold: float = 0.0001    # Threshold for "flat" class
    train_ratio: float = 0.7          # Train fraction (time-ordered)
    val_ratio: float = 0.15           # Validation fraction
    intrinsic_time: bool = False      # Convert to directional-change bars
    include_sentiment: bool = False   # Attach GDELT sentiment features
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    num_features: int                 # Input feature dimension
    hidden_size_lstm: int = 64        # LSTM hidden dim
    num_layers_lstm: int = 1          # Number of LSTM layers
    cnn_num_filters: int = 32         # CNN filter count
    cnn_kernel_size: int = 3          # CNN kernel size
    attention_dim: int = 64           # Attention hidden dim
    dropout: float = 0.1              # Dropout rate
    bidirectional: bool = True        # Bidirectional LSTM
    use_optimized_attention: bool = False  # Optimized attention
    use_multihead_attention: bool = False  # Multi-head attention
    n_attention_heads: int = 4        # Number of heads (if multihead)
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    epochs: int = 10                  # Training epochs
    batch_size: int = 64              # Batch size
    learning_rate: float = 1e-3       # Adam LR
    weight_decay: float = 0.0         # L2 regularization
    device: str = "cpu"               # Device
    grad_clip: float = 1.0            # Gradient clipping norm
    early_stop_patience: int = 3      # Patience for early stopping
    top_n_checkpoints: int = 3        # Keep top N checkpoints
```

### A3CConfig

```python
@dataclass
class A3CConfig:
    n_workers: int = 4                # Number of worker processes
    total_steps: int = 100_000        # Total environment steps
    rollout_length: int = 5           # Steps per rollout
    learning_rate: float = 1e-4       # Adam LR
    gamma: float = 0.99               # Discount factor
    entropy_coef: float = 0.01        # Entropy bonus
    value_loss_coef: float = 0.5      # Value loss weight
    max_grad_norm: float = 0.5        # Gradient clipping
```

---

## API Reference

### Key Functions

#### `set_seed(seed: int | None) -> int`

Set reproducible seeds across all libraries.

```python
from utils.seed import set_seed
seed_value = set_seed(42)  # Returns 42
```

#### `train_model(...) -> Dict[str, List[float]]`

Train a DignityModel on classification/regression task.

```python
from train.core.agent_train import train_model
from models.agent_hybrid import build_model

model = build_model(model_cfg, task_type="classification")
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=training_cfg,
    task_type="classification",
)
# history["train_loss"], history["val_loss"], history["val_metric"]
```

#### `evaluate_a3c_agent(...) -> Dict[str, float]`

Evaluate trained A3C agent on environment.

```python
from eval.agent_eval import evaluate_a3c_agent

metrics = evaluate_a3c_agent(
    agent=trained_agent,
    env_factory=lambda: SimulatedRetailExecutionEnv(),
    num_episodes=100,
    device="cuda",
)
# Returns: mean_reward, std_reward, win_rate, sharpe_ratio, ...
```

#### `EarlyStopping(patience: int) -> bool`

Monitor validation metric and halt training on plateau.

```python
from utils.training_checkpoint import EarlyStopping

early_stop = EarlyStopping(patience=3, min_delta=1e-3, mode="min")
for epoch in range(100):
    val_loss = train_epoch(...)
    if early_stop(val_loss):
        print("Training stopped early")
        break
```

#### `CheckpointManager(save_dir: Path, top_n: int)`

Manage checkpoint retention and best model tracking.

```python
from utils.training_checkpoint import CheckpointManager

manager = CheckpointManager("models/checkpoints", top_n=3)
for epoch in range(100):
    loss = train_epoch(...)
    manager.save(model.state_dict(), score=loss, epoch=epoch)

best_state = manager.load_best()
model.load_state_dict(best_state)
```

#### `validate_dataframe(df: pd.DataFrame) -> pd.DataFrame`

Validate and sanitize raw price data.

```python
from data.prepare_dataset import validate_dataframe

df_clean = validate_dataframe(df_raw)
# Checks: columns, dtypes, NaN rows, duplicate timestamps, OHLC relationships
```

---

## Troubleshooting

### Training Issues

| Issue                     | Solution                                                                          |
|---------------------------|-----------------------------------------------------------------------------------|
| Out of Memory (OOM)       | Reduce `batch_size`, use `--use-amp`, enable `--num-workers` for async loading    |
| Slow training             | Use `--num-workers 4 --pin-memory --prefetch-factor 4`, enable `--use-amp` on GPU |
| Poor accuracy             | Check data quality (validation output), increase `t_in`, tune learning rate       |
| Non-deterministic results | Call `set_seed(42)` at program start                                              |
| Model diverges            | Reduce `learning_rate`, enable `grad_clip`, lower batch size                      |

### Data Issues

| Issue                  | Solution                                       |
|------------------------|------------------------------------------------|
| Missing data           | Run `validate_dataframe()`, check for NaN rows |
| Time zone misalignment | Use `ensure_utc_timezone()` after loading      |
| Duplicate timestamps   | Use `deduplicate_on_datetime()`                |
| Unbalanced splits      | Check `train_ratio`, `val_ratio` parameters    |

### RL Issues

| Issue                | Solution                                                            |
|----------------------|---------------------------------------------------------------------|
| Policy doesn't learn | Reduce `entropy_coef`, check reward signal, verify environment      |
| Workers hang         | Set `daemon=True` (already set), check for deadlocks                |
| Low win rate         | Increase `total_steps`, tune `learning_rate`, adjust reward shaping |

### CLI Issues

| Issue                  | Solution                                                 |
|------------------------|----------------------------------------------------------|
| `ModuleNotFoundError`  | Verify imports, check `sys.path`, run from repo root     |
| Argument parsing fails | Check argument names: `--pairs` vs `--pair`, check types |
| CUDA not found         | Set `--device cpu`, verify PyTorch CUDA installation     |

---

## Performance Tips

1. **GPU Acceleration:**
    - Use `--device cuda --use-amp --num-workers 4`
    - Expected speedup: 3-5x

2. **Large-Scale Training:**
    - Enable `--pin-memory --prefetch-factor 4`
    - Use async checkpoint saving: `AsyncCheckpointSaver`

3. **Reproducibility:**
    - Always call `set_seed()` at program start
    - Log random seed for experiment tracking

4. **Memory Efficiency:**
    - Enable AMP: `--use-amp` (float16)
    - Use gradient accumulation (not yet implemented)
    - Reduce `t_in` window if memory constrained

5. **Model Convergence:**
    - Use cosine annealing scheduler
    - Early stopping with patience=3
    - Gradient clipping (default: 1.0)

---

## Phase 3: Production Enhancements

### Overview

Phase 3 adds production-ready features to bridge the gap between backtesting and live trading:

- **Transaction Cost Modeling**: Realistic commission, spreads, and slippage
- **Dynamic Position Sizing**: Kelly-criterion-inspired risk-based sizing
- **Risk Management**: Stop-loss, take-profit, and drawdown limits

### 3.1 Transaction Cost Modeling

#### ExecutionConfig - Transaction Cost Parameters

```python
from execution.simulated_retail_env import ExecutionConfig

config = ExecutionConfig(
    # Commission costs
    commission_per_lot=7.0,            # Fixed $7 per lot (FX retail)
    commission_pct=0.0001,             # OR 0.01% of notional

    # Variable spreads
    variable_spread=True,              # Enable volatility-based widening
    spread_volatility_multiplier=2.0,  # 2x spread during high volatility
)
```

**Cost Tracking**:

- `env._commission_paid`: Total commission costs
- `env._spread_paid`: Total spread costs
- `env._slippage_paid`: Total slippage costs

**Behavior**:

- Spreads widen when `volatility_ratio > 1.5`
- Uses EMA of recent price shocks to detect volatility regimes
- All costs deducted from cash on every fill

### 3.2 Dynamic Position Sizing

#### ActionConverter - Position Sizing Logic

```python
from train.core.env_based_rl_training import ActionConverter

converter = ActionConverter(
    lot_size=1.0,               # Base lot size
    max_position=10.0,          # Maximum inventory (long or short)
    risk_per_trade=0.02,        # Risk 2% of portfolio per trade
    use_dynamic_sizing=True,    # Enable Kelly-inspired sizing
)
```

**Sizing Formula** (when `use_dynamic_sizing=True`):

```python
portfolio_value = cash + (inventory * mid_price)
risk_amount = portfolio_value * risk_per_trade
size = risk_amount / mid_price
```

**Constraints Applied**:

1. **Position Limits**: Size capped to prevent exceeding `max_position`
2. **Cash Constraints**: Buy orders limited by available cash
3. **Lot Rounding**: Sizes rounded to `lot_size` increments

**Multi-Pair Behavior**:

- `max_position` is per-pair (independent inventory tracking)
- Enables diversification: long 8 EUR/USD AND long 6 GBP/USD simultaneously

### 3.3 Risk Management

#### Risk Control Parameters

```python
config = ExecutionConfig(
    # Stop-loss (per-position)
    enable_stop_loss=False,     # Disabled by default - let agent learn
    stop_loss_pct=0.02,         # Close at 2% loss from entry

    # Take-profit (per-position)
    enable_take_profit=False,   # Disabled by default
    take_profit_pct=0.04,       # Close at 4% gain from entry

    # Drawdown limit (portfolio-level)
    enable_drawdown_limit=True,  # Recommended for training
    max_drawdown_pct=0.20,       # Terminate episode at 20% drawdown
)
```

**Risk Metrics**:

- `env._stop_loss_triggered`: Count of stop-loss exits
- `env._take_profit_triggered`: Count of take-profit exits
- `env._peak_portfolio_value`: Peak portfolio for drawdown calculation

**Behavior**:

- Stop-loss/take-profit checked every step after price update
- Drawdown = `(peak_portfolio - current_portfolio) / peak_portfolio`
- Episode terminates early if `drawdown >= max_drawdown_pct`

### Production Configuration Examples

#### Balanced (Recommended)

```python
env_config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,          # 1 basis point
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

#### Conservative (High Safety)

```python
env_config = ExecutionConfig(
    initial_cash=50_000.0,
    commission_pct=0.0001,
    variable_spread=True,
    enable_stop_loss=True,
    stop_loss_pct=0.01,             # Tight 1% stop
    enable_take_profit=True,
    take_profit_pct=0.02,
    enable_drawdown_limit=True,
    max_drawdown_pct=0.10,          # Conservative 10%
)

converter = ActionConverter(
    max_position=5.0,               # Lower limit
    risk_per_trade=0.01,            # 1% risk
    use_dynamic_sizing=True,
)
```

See [Configuration Reference](../CONFIGURATION_REFERENCE.md) for complete options.

### Testing & Validation

All Phase 3 features validated with comprehensive test suite:

- **16/16 Phase 3 tests passing**
- **25/25 total tests passing** (Phases 1-3)

See [Testing & Validation Report](../TESTING_VALIDATION_REPORT.md) for details.

---

## References

- **Attention Mechanisms:** `utils/attention_optimization.py`
- **Risk Management:** `risk/risk_manager.py`
- **Feature Engineering:** `features/agent_features.py`
- **Sentiment Analysis:** `features/agent_sentiment.py`
- **Phase 3 Implementation:** See [Phase 3 Implementation Summary](../implementation/PHASE_3_IMPLEMENTATION_SUMMARY.md)
- **Configuration Guide:** See [Configuration Reference](../CONFIGURATION_REFERENCE.md)

---

*Document updated: 2025-12-29 | Implementation status: Complete (Phases 1-3)*
