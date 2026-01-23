# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sequence** is a deep learning framework for multi-asset market prediction (FX, crypto, commodities) combining CNN-LSTM-Attention hybrid models, intrinsic time representations, sentiment analysis (GDELT + FinBERT), and reinforcement learning (A3C) for algorithmic trading.

## Environment Setup

```bash
# Conda environment is at /opt/miniconda3 (system-level, NOT ~/opt)
# Activate with:
source /opt/miniconda3/bin/activate

# Verify PyTorch (2.6.0+) is available
python -c "import torch; print(torch.__version__)"

# Install project dependencies
pip install -r requirements.txt

# Install fundamental data packages (editable installs)
bash run/scripts/install_data_sources.sh
```

**API Keys** (store in `.env` file, never commit):
```bash
FRED_API_KEY=your_key        # Required - get free at https://fred.stlouisfed.org/docs/api/api_key.html
COMTRADE_API_KEY=your_key    # Optional (500 records/request without key)
```

## Common Commands

### Data Preparation

```bash
# FX pair with intrinsic time transformation
python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 --t-out 10 \
  --task-type classification \
  --intrinsic-time --dc-threshold-up 0.0005

# Crypto pair (auto-detects asset class, applies crypto-specific indicators)
python data/prepare_dataset.py \
  --pairs btcusd \
  --t-in 120 --t-out 10 \
  --intrinsic-time --dc-threshold-up 0.005

# With GDELT sentiment enrichment
python data/prepare_dataset.py --pairs gbpusd --include-sentiment

# Collect fundamental economic data
python run/scripts/example_fundamental_integration.py \
  --pair EURUSD --start 2023-01-01 --end 2023-12-31
```

### Training

```bash
# Supervised learning
python train/run_training.py --pairs gbpusd --epochs 50 --batch-size 64

# Multi-task learning (price + volatility + regime)
python train/run_training_multitask.py --pairs gbpusd --epochs 50

# Reinforcement learning (A3C with backtesting)
python rl/run_a3c_training.py \
  --pair gbpusd --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 8 --total-steps 1000000

# Unified pipeline (download → prepare → train → RL)
python run/training_pipeline.py \
  --pairs gbpusd --run-histdata-download --epochs 50 \
  --run-rl-training --rl-env-mode backtesting

# Enable quantum reservoir features (experimental)
python train/run_training.py --pairs gbpusd --feature-groups quantum_reservoir
```

### Testing

```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_agent_features.py

# Run single test function
pytest tests/test_agent_features.py::test_compute_features_basic -v

# Run tests matching pattern
pytest -k "test_fred" tests/

# Run fast tests only (5s timeout)
pytest -m fast tests/

# Run with verbose output
pytest -v tests/
```

Test markers (defined in `conftest.py`):
- `@pytest.mark.fast` → 5 second timeout
- `@pytest.mark.timeout_300` → 300 second timeout for integration tests

### Code Quality

```bash
ruff check .           # Lint
ruff format .          # Format
ruff check --fix .     # Auto-fix issues
```

**Ruff Configuration** (see `pyproject.toml`):
- Line length: 100 characters
- Target: Python 3.10+
- Import sorting: Enabled (isort rules)
- Key exclusions:
  - `E501` (line length - handled by formatter)
  - `E402` (module-level imports after sys.path manipulation - intentional for Colab compatibility)
  - External code excluded: `models/timesFM`, `.venvx`, `build`, `dist`

Per-file rules:
- Test files: Allow unused imports (`F401`) and import order flexibility
- Model files: Allow PyTorch convention (`torch.nn.functional as F`)

### Utility Scripts

Helper scripts in `run/scripts/`:

```bash
# Validate prepared dataset integrity
python run/scripts/validate_training_data.py \
  --data-path data/data/gbpusd/gbpusd_prepared.csv

# Debug attention mechanism behavior
python run/scripts/debug_attention.py \
  --checkpoint models/gbpusd_best_model.pt

# Run hyperparameter search with Optuna
python run/scripts/run_hyperparameter_tuning.py \
  --pairs gbpusd \
  --n-trials 50 \
  --study-name gbpusd_optimization

# Full integration smoke test
python run/scripts/test_full_integration.py
```

## Architecture Overview

### Core Components

| Component | Location | Key Files |
|-----------|----------|-----------|
| **Data Pipeline** | `data/`, `train/features/` | `prepare_dataset.py`, `iterable_dataset.py`, `agent_features.py` |
| **Fundamental Data** | `data/downloaders/` | `fred_downloader.py`, `comtrade_downloader.py`, `ecb_shocks_downloader.py` |
| **Models** | `models/` | `agent_hybrid.py` (SharedEncoder base), `agent_multitask.py`, `regime_encoder.py` |
| **Training** | `train/` | `run_training.py`, `core/agent_train.py`, `training_manager.py` |
| **RL Agents** | `rl/` | `run_a3c_training.py`, `agents/a3c_agent.py`, `agents/sac_agent.py` |
| **Execution Envs** | `train/execution/` | `backtesting_env.py`, `simulated_retail_env.py` |
| **Evaluation** | `eval/` | `run_evaluation.py`, `ensemble_timesfm.py` |
| **Configuration** | `run/config/` | `config.py` (DataConfig, ModelConfig, TrainingConfig, etc.) |
| **Quantum (Experimental)** | `models/quantum_emulation/` | `reservoir/ising_qrc.py`, `walks/adg.py` |

### Key Architectural Patterns

**SharedEncoder Pattern**: All model variants (supervised, multi-task, RL) inherit from `SharedEncoder` in `models/agent_hybrid.py`, providing:
- Temporal local features via 1D CNN layers
- Sequential dependencies via bidirectional LSTM
- Context aggregation via multi-head attention
- Unified embedding for downstream tasks

**Configuration System**: The `run/config/config.py` module defines dataclass configs imported throughout:
- `DataConfig`: Dataset parameters (t_in, t_out, train/val/test splits)
- `FeatureConfig`: Feature engineering toggles (SMA/EMA windows, RSI, Bollinger bands)
- `ModelConfig`: Neural architecture hyperparameters (LSTM hidden size, CNN filters, attention dim)
- `TrainingConfig`: Training parameters (epochs, batch size, learning rate, optimizer)
- `RiskConfig`: Risk management constraints (position limits, drawdown thresholds)
- `ExecutionConfig`: Trading environment settings (spreads, slippage, latency)

**Intrinsic Time Transformation**: The `train/features/intrinsic_time.py` module implements directional-change (DC) bars, replacing fixed-time sampling with event-driven bars triggered by price movements exceeding thresholds. This provides scale-invariant market structure representation.

**Two-Stage Training**:
1. **Signal Model Pretraining** (`pretrain_signal_model` in `train/core/agent_train.py`): Train CNN-LSTM-Attention on supervised price prediction
2. **Execution Policy Training** (`train_execution_policy`): Train RL policy on top of frozen signal model embeddings for optimal execution

**Environment Modes**:
- **Simulated**: Stochastic retail execution with configurable spreads, slippage, latency (`train/execution/simulated_retail_env.py`)
- **Backtesting**: Deterministic historical replay using `backtesting.py` library (`train/execution/backtesting_env.py`)

## Data Flow

```
HistData CSVs (Central time)
  ↓ data/prepare_dataset.py
UTC conversion + deduplication
  ↓ train/features/agent_features.py
Technical indicators (SMA, EMA, RSI, ATR, Bollinger, etc.)
  ↓ train/features/intrinsic_time.py (optional)
Directional-change bars (if --intrinsic-time)
  ↓ train/features/agent_sentiment.py (optional)
GDELT sentiment features (if --include-sentiment)
  ↓ data/iterable_dataset.py
PyTorch IterableDataset (train/val/test splits)
  ↓ train/core/agent_train.py
Model training (supervised or RL)
  ↓ eval/agent_eval.py
Evaluation metrics
```

## Important Development Notes

### Configuration Import Pattern

The config lives in `run/config/config.py`. Scripts add `run/` to `sys.path`, enabling the short import:

```python
from config.config import ModelConfig, TrainingConfig  # Standard pattern
```

If you get `ModuleNotFoundError: No module named 'config'`, add `run/` to your path:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
```

### Feature Engineering Extensibility

The `train/features/agent_features.py` module supports dynamic feature loading from `train/features/generated/` for research-generated features. Add custom features by:
1. Creating a `.py` file in `train/features/generated/`
2. Defining a function with signature `fn(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame`
3. Features are auto-loaded via `_load_generated_features()`

### Intrinsic Time Thresholds

Directional-change thresholds are FX pair-specific. Typical values:
- Major pairs (EUR/USD, GBP/USD): 0.0005–0.001 (5–10 pips)
- Volatile pairs (GBP/JPY): 0.001–0.002 (10–20 pips)
- Crypto pairs: 0.005–0.01 (0.5%–1%)

### GDELT Sentiment Pipeline

Multi-stage process in `data/gdelt/`:
1. `consolidated_downloader.py`: Download GDELT GKG files
2. `parser.py`: Parse news events and filter for FX relevance
3. `train/features/agent_sentiment.py`: Run FinBERT sentiment analysis
4. `alignment.py`: Temporal alignment with OHLCV data

### Fundamental Data Pipeline

Three sources complement price data:

| Source | Data | Pairs | API Key |
|--------|------|-------|---------|
| **FRED** | Interest rates, CPI, GDP, unemployment | All (USD, EUR, GBP, JPY, AUD, CAD) | Required (free) |
| **UN Comtrade** | Monthly trade balance | Major pairs | Optional |
| **ECB Shocks** | Monetary policy surprises | EUR pairs only | Not needed (CSV) |

```python
from data.extended_data_collection import collect_all_forex_fundamentals

data = collect_all_forex_fundamentals(
    currency_pair="EURUSD",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

**Forward-fill caveat**: Fundamental data uses forward-fill for alignment with price data. During high volatility, stale monthly/quarterly data may not reflect current conditions. Consider staleness checks if >30 days old.

### Quantum Emulation (Experimental)

Opt-in quantum-inspired features in `models/quantum_emulation/`:

- **QRC (Quantum Reservoir Computing)**: Feature block for volatility/regime signals (`reservoir/ising_qrc.py`)
- **Quantum-walk ADG**: Distribution generator for risk/scenario analysis (`walks/adg.py`)
- **QUBO/N-Choose-K**: Discrete selection helpers (`optimization/`)

Enable with: `--feature-groups quantum_reservoir`

Defaults controlled in `FeatureConfig` (`qrc_*` knobs). Uses statevector simulation for n_qubits ≤ 10, lightweight RNN approximation otherwise.

### RL Environment Selection

- Use **backtesting mode** for reproducible experiments and paper results
- Use **simulated mode** for testing robustness to execution noise
- Historical data path format: `data/data/{pair}/{pair}_prepared.csv`

### Multi-GPU Training

Training scripts automatically detect and use all available GPUs via `torch.nn.DataParallel`. Batch size is split across GPUs. No code changes required.

### Checkpoint Management

Checkpoints are saved to `models/` by default:
- `best_model.pt`: Best validation loss checkpoint
- `checkpoint_epoch_{N}/`: Full training state for resumption
- Use `--resume-from-checkpoint` to continue training

### Tracing and Observability

OpenTelemetry tracing is available in `utils/tracing.py`. Initialize with:

```python
from utils.tracing import setup_tracing
setup_tracing(
    service_name="sequence-training",
    otlp_endpoint="http://localhost:4318",
    environment="development"
)
```

## Common Pitfalls

1. **Import Paths**: Always add `ROOT` to `sys.path` in scripts:
   ```python
   ROOT = Path(__file__).resolve().parents[1]
   if str(ROOT) not in sys.path:
       sys.path.insert(0, str(ROOT))
   ```

2. **Mock Patch Paths**: Patch where the object is *used*, not where it's defined:

   ```python
   # WRONG: Patching the definition location
   @patch("fred.Fred")

   # CORRECT: Patch where it's imported/used
   @patch("data.downloaders.fred_downloader.Fred")
   ```

3. **Time Zones**: HistData CSVs are in US Central time. Use `utils.datetime_utils.convert_to_utc_and_dedup()` for UTC conversion.

4. **Feature Column Order**: Feature columns must match between training and inference. Use `DataConfig.feature_columns` to enforce consistency.

5. **GDELT Mirror**: Default GDELT endpoint can be slow. Use `--gdelt-mirror` for custom mirrors.

6. **Memory Usage**: For large datasets, use `data/iterable_dataset.py` (streaming) instead of loading full data into memory.

7. **Attention Sequence Length**: For sequences >1024, enable `use_optimized_attention=True` in ModelConfig to avoid OOM errors.

8. **joblib.Memory**: The `bytes_limit` parameter is deprecated in recent versions. Use `mmap_mode` for memory control instead.

9. **Python Path Management**: Scripts use intentional `E402` pattern (imports after sys.path) for Colab compatibility. Don't "fix" these imports.
