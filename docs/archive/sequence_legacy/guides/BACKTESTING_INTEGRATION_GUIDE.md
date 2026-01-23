# Backtesting.py Integration Guide

## Overview

Backtesting.py is **fully integrated** into both the standalone A3C training script and the unified training pipeline.
This provides deterministic historical replay for reproducible training experiments.

## Integration Points

1. **Standalone RL Training**: `rl/run_a3c_training.py` - Direct RL training
2. **Unified Pipeline**: `utils/run_training_pipeline.py` - Data download → Supervised training → RL training (optional)

---

## Unified Pipeline (Recommended)

The unified pipeline handles the complete workflow: download → prepare → train supervised → train RL.

### Full Pipeline with RL Training

```bash
python utils/run_training_pipeline.py \
  --pairs gbpusd,eurusd \
  --run-histdata-download \
  --epochs 10 \
  --run-rl-training \
  --rl-env-mode backtesting \
  --rl-num-workers 4 \
  --rl-total-steps 100000
```

### Pipeline Stages

1. **Data Download** (optional): HistData, yfinance, GDELT
2. **Data Preparation**: Feature engineering, train/val/test splits
3. **Supervised Training**: Classification/regression model
4. **Evaluation**: Test set metrics
5. **RL Training** (optional): A3C policy with backtesting or simulation

### Key Pipeline Arguments

- `--run-rl-training`: Enable RL training after supervised training
- `--rl-env-mode {simulated,backtesting}`: Choose environment type
- `--rl-num-workers N`: Number of A3C workers
- `--rl-total-steps N`: Total training steps
- `--rl-learning-rate F`: A3C learning rate
- `--rl-entropy-coef F`: Exploration bonus
- `--rl-initial-balance F`: Starting account balance
- `--rl-checkpoint-dir PATH`: RL checkpoint directory

---

## Standalone RL Training

---

## Standalone RL Training

For direct RL training without the full pipeline:

### Simulated Mode (Default - Stochastic)

```bash
python rl/run_a3c_training.py \
  --pair gbpusd \
  --num-workers 4 \
  --total-steps 100000 \
  --device cuda
```

- Uses `SimulatedRetailExecutionEnv`
- Stochastic execution with spread, slippage, realistic fills
- Non-deterministic (different runs produce different results)
- Best for exploration during training

### Backtesting Mode (Deterministic)

```bash
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 4 \
  --total-steps 100000 \
  --device cuda
```

- Uses `BacktestingRetailExecutionEnv`
- Deterministic bar-by-bar historical replay
- Reproducible (same seed = same results)
- Best for validation and debugging

---

## Usage Examples

### Example 1: Complete Pipeline with Backtesting RL

```bash
python utils/run_training_pipeline.py \
  --pairs gbpusd \
  --run-histdata-download \
  --years 2023,2024 \
  --epochs 5 \
  --run-rl-training \
  --rl-env-mode backtesting \
  --rl-num-workers 8 \
  --rl-total-steps 200000 \
  --device cuda
```

**Pipeline stages:**

1. Downloads HistData for GBPUSD (2023-2024)
2. Prepares features and train/val/test splits
3. Trains supervised model (5 epochs)
4. Evaluates on test set
5. Trains A3C agent with backtesting (200k steps, 8 workers)

### Example 2: Multi-Pair Pipeline with Simulated RL

```bash
python utils/run_training_pipeline.py \
  --pairs gbpusd,eurusd,usdjpy \
  --epochs 3 \
  --run-rl-training \
  --rl-env-mode simulated \
  --rl-num-workers 4 \
  --rl-total-steps 50000
```

**Runs for each pair sequentially:**

- Supervised training → Evaluation → RL training (simulated)

### Example 3: Supervised Only (No RL)

```bash
python utils/run_training_pipeline.py \
  --pairs gbpusd \
  --epochs 10 \
  --batch-size 128
```

**Standard supervised training** (RL disabled by default)

```bash
python rl/run_a3c_training.py \
  --pair gbpusd \
  --num-workers 4 \
  --total-steps 100000 \
  --device cuda
```

- Uses `SimulatedRetailExecutionEnv`
- Stochastic execution with spread, slippage, realistic fills
- Non-deterministic (different runs produce different results)
- Best for exploration during training

### Backtesting Mode (Deterministic)

```bash
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 4 \
  --total-steps 100000 \
  --device cuda
```

- Uses `BacktestingRetailExecutionEnv`
- Deterministic bar-by-bar historical replay
- Reproducible (same seed = same results)
- Best for validation and debugging

## Key Differences

| Feature         | Simulated Mode       | Backtesting Mode                  |
|-----------------|----------------------|-----------------------------------|
| Execution       | Stochastic           | Deterministic                     |
| Spread/Slippage | Realistic variance   | Fixed from data                   |
| Reproducibility | Non-deterministic    | Fully reproducible                |
| Speed           | Fast                 | Slightly slower (replay overhead) |
| Data Required   | None (generates)     | Historical OHLCV CSV              |
| Use Case        | Exploration training | Validation/debugging              |

## CLI Arguments

### Environment Selection

- `--env-mode {simulated,backtesting}`: Choose environment type (default: `simulated`)
- `--historical-data PATH`: Path to OHLCV CSV (required for backtesting mode)
- `--data-root PATH`: Root directory to auto-locate data files (default: `data/data`)

### Automatic Data Location

If `--historical-data` is not provided in backtesting mode, the script tries:

1. `{data-root}/{pair}/{pair}_prepared.csv`
2. `{data-root}/{pair}.csv`
3. `data/{pair}/{pair}.csv`
4. `{pair}.csv`

## Data Format Requirements (Backtesting Mode)

Required columns (case-insensitive):

- `open`, `high`, `low`, `close`
- Optional: `volume`, `datetime` (for indexing)

Example CSV:

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,1.2650,1.2655,1.2648,1.2652,1000
2024-01-01 00:01:00,1.2652,1.2658,1.2651,1.2656,1200
...
```

## Usage Examples

### Train with Backtesting Mode (Auto-locate Data)

```bash
python rl/run_a3c_training.py \
  --pair eurusd \
  --env-mode backtesting \
  --data-root data/data \
  --num-workers 8 \
  --total-steps 200000
```

### Train with Backtesting Mode (Explicit Data Path)

```bash
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data /path/to/gbpusd_historical.csv \
  --num-workers 4 \
  --learning-rate 5e-5 \
  --entropy-coef 0.02
```

### Compare Simulated vs Backtesting

```bash
# Run 1: Simulated (stochastic)
python rl/run_a3c_training.py \
  --pair gbpusd \
  --checkpoint-path models/a3c_simulated.pt

# Run 2: Backtesting (deterministic)
python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --checkpoint-path models/a3c_backtesting.pt
```

## Troubleshooting

### Error: "backtesting.py is required"

**Solution:** Install the dependency

```bash
pip install backtesting>=0.3.2
```

### Error: "Historical data not found"

**Solution:** Provide explicit path or prepare data

```bash
# Option 1: Specify path
--historical-data /full/path/to/data.csv

# Option 2: Prepare data
python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10
```

### Error: "Missing required OHLCV columns"

**Solution:** Ensure CSV has open/high/low/close columns

```bash
# Check available columns
head -1 your_data.csv

# Rename columns if needed (case-insensitive matching)
# open -> Open, high -> High, etc.
```

## Implementation Details

### File Structure

- **`rl/run_a3c_training.py`**: Main CLI runner with environment selection
- **`execution/backtesting_env.py`**: BacktestingRetailExecutionEnv wrapper
- **`execution/simulated_retail_env.py`**: SimulatedRetailExecutionEnv (default)

### Environment Factory Pattern

Both environments share the same gym-like interface:

```python
def make_env():
    if args.env_mode == "simulated":
        return SimulatedRetailExecutionEnv(...)
    else:
        return BacktestingRetailExecutionEnv(...)
```

### Worker Compatibility

All A3C workers create independent environment instances, so backtesting mode works seamlessly with multi-worker
training.

## Best Practices

1. **Use simulated mode for training**: Better exploration due to stochasticity
2. **Use backtesting mode for validation**: Verify learned policies are robust
3. **Match data distributions**: Ensure backtesting data matches expected market conditions
4. **Set consistent seeds**: For reproducibility in backtesting mode
5. **Monitor both modes**: Compare performance across environment types

## References

- Official backtesting.py docs: https://kernc.github.io/backtesting.py/
- Implementation status: `BACKTESTING_IMPLEMENTATION_STATUS.md`
- Architecture reference: `ARCHITECTURE_API_REFERENCE.md`
