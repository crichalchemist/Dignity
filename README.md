# Dignity Core – Privacy-Preserving Sequence Modeling for Transactional Behavior

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
[![CI](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml/badge.svg)](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Dignity** is a privacy-preserving deep learning framework for modeling transactional behavior patterns. It provides modular components for signal processing, data pipelines, and neural architectures with an input-level differential-privacy stage, k-anonymous generalization, and keyed pseudonymization — each backed by a test.

## Overview

Dignity Core implements privacy-first sequence modeling:

- **Privacy-Preserving**: Keyed pseudonymization, k-anonymous generalization, and input-level ε-DP with a budget ledger
- **Signal Processing**: 32 quant finance features — RSI, MACD, Bollinger Bands, ATR, stochastic, ADX, OBV, VWAP, volatility, momentum, regime detection, and more
- **Modular Architecture**: Clean separation between data pipeline, model components, and training infrastructure
- **Deployable**: ONNX export with verification and benchmarking for production inference
- **Multi-Task**: Configurable for risk scoring, forecasting, policy learning, or hierarchical cascade
- **Lightweight**: ~8,000 lines of focused, production-ready Python

## Key Features

- **Privacy-First**: Keyed pseudonymization, bounded Laplace noise with ε accounting, k-anonymous generalization — see docs/PRIVACY.md
- **32 Signal Features**: Full OHLCV-derived feature set for quant finance (RSI, MACD, Bollinger %B, ATR, stochastic, ADX, OBV, VWAP, realized volatility, DC state machine, and more)
- **Modular Design**: Core utilities, data pipeline, and model components are cleanly separated
- **Hybrid Backbone**: CNN1D + StackedLSTM + AdditiveAttention with composable task heads
- **4 Task Types**: Risk scoring, forecasting, policy learning (RL), and hierarchical cascade (Regime → Risk → Alpha → Policy)
- **ONNX Export**: Conversion with verification and inference benchmarking
- **Synthetic Data**: Built-in generators for testing and prototyping
- **Training Infrastructure**: AMP support, gradient clipping, cosine scheduling, checkpointing, CLI interface
- **384 Tests**: Comprehensive test suite across core utilities, data pipeline, models, training, export, and backtest
- **YAML Configuration**: Declarative configs for different tasks and environments

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/crichalchemist/Dignity.git
cd Dignity

# Create conda environment (recommended)
conda create -n dignity python=3.11
conda activate dignity

# Install dependencies
pip install -r requirements.txt

# Or install package (provides dignity-train, dignity-export, dignity-backtest)
pip install -e .
```

### Basic Usage

#### Generate Synthetic Data

```python
from data.source.synthetic import SyntheticGenerator

# Generate synthetic transaction sequences
gen = SyntheticGenerator(seed=42)
dataset = gen.generate_dataset(num_normal=800, num_anomalous=200, seq_len=100)

print(f"Generated {len(dataset)} rows")
```

#### Privacy Operations

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

# One budget per release; epsilons add and overspending raises BudgetExhausted.
pm = PrivacyManager(
    PrivacyBudget(epsilon_total=1.0), key=b"load-this-from-the-environment"
)

# Keyed pseudonymization (HMAC-SHA256): linkable under one key, unlinkable across keys
pseudonym = pm.pseudonymize("0x1234abcd5678ef90")

# Bounded Laplace noise: clipped to bounds, sensitivity = hi - lo, spends epsilon
amounts = np.array([123.456, 789.012, 456.789])
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))

# k-anonymous generalization: every output value is shared by >= k records
generalized = PrivacyManager.generalize_amounts(
    np.random.uniform(10, 100, 200), bins=10, k=5
)
```

In training, none of this is called by hand: a `privacy:` block in the config
turns on a pipeline stage that runs before signals are computed. **No block, no
stage.** See `config/train_risk.yaml` and [docs/PRIVACY.md](docs/PRIVACY.md).

#### Train a Model

```bash
# Train risk model
dignity-train --config config/train_risk.yaml

# Train forecasting model
dignity-train --config config/train_forecast.yaml

# Train cascade model (Regime → Risk → Alpha → Policy)
dignity-train --config config/train_quant_paper.yaml

# Train on Colab with optimized settings
dignity-train --config config/colab.yaml
```

#### Export to ONNX

```bash
# Export trained model to ONNX
python -m export.to_onnx \
    --checkpoint checkpoints/dignity_risk_best.pt \
    --output dignity_risk.onnx \
    --benchmark
```

## Architecture

### Data Pipeline Flow

```
Data Sources
├── Synthetic Generator (testing/prototyping)
├── Cryptocurrency APIs (CryptoSource, MetaApiSource)
└── Custom Sources (via data/source/ extensibility)
                    ↓
Privacy Stage (core/privacy.py — optional, from the `privacy:` config block)
├── Bounded Laplace noise (ε-DP, budget-ledgered)
└── k-anonymous generalization (quantile bins)
                    ↓
Signal Processing (core/signals.py)
├── 32 OHLCV-derived features
├── Technical indicators: RSI, MACD, Bollinger, ATR, Stochastic, ADX
├── Volume signals: OBV, VWAP, order flow imbalance
├── Volatility: realized vol, vol ratio
├── Momentum: ROC, price momentum
├── DC state machine: directional change, overshoot, bars since event
└── Regime detection (volatility clustering)
                    ↓
Data Pipeline (data/pipeline.py)
├── Signal computation (SignalProcessor.process_sequence)
├── Feature scaling (RobustScaler)
├── Sequence windowing
└── PyTorch DataLoader
                    ↓
Model Architecture (models/)
├── Backbone: CNN1D → StackedLSTM → AdditiveAttention
├── Task Heads: Risk | Forecast | Policy | Cascade
└── ~500K+ parameters (default config)
                    ↓
Training (train/engine.py + train/cli.py)
├── Mixed precision (AMP)
├── Gradient clipping
├── Cosine LR scheduling
├── Checkpointing
└── Guided Learning (cascade)
                    ↓
Export & Deploy (export/to_onnx.py)
├── ONNX conversion (with cascade wrapper)
├── Model verification
└── Inference benchmarking
```

### Neural Network Design

The core model implements a modular hybrid architecture:

**Backbone** (`models/backbone/hybrid.py`):
1. **CNN1D**: 2-layer 1D convolution for local pattern extraction
2. **StackedLSTM**: Unidirectional LSTM for temporal dependencies
3. **AdditiveAttention**: Attention mechanism for feature weighting

**Task Heads** (`models/head/`):
- **RiskHead**: Dual sigmoid outputs — VaR estimate and position limit
- **ForecastHead**: Multi-step time series forecasting (configurable features and horizon)
- **PolicyHead**: RL actor-critic — action logits and value estimate
- **RegimeHead**: Market regime classification (calm, trending, volatile, crisis)
- **AlphaHead**: Risk-adjusted return prediction (tanh-bounded score)

**Main Model** (`models/dignity.py`):
- Composable: `Dignity(task='risk')` assembles backbone + appropriate head
- Cascade mode: `Dignity(task='cascade')` chains Regime → Risk → Alpha → Policy heads
- 4 tasks: `risk`, `forecast`, `policy`, `cascade`
- `cascade_loss()` implements Guided Learning — auxiliary supervision at every head

### Privacy Operations

`core/privacy.py` claims exactly three things, each enforced by a test:

- **Keyed pseudonymization** (HMAC-SHA256, key required) — linkable under one key, unlinkable across keys
- **Input-level ε-differential privacy** — clipped Laplace noise with sensitivity derived from public bounds and ε spent against a ledger that refuses to overspend
- **k-anonymity** for generalized columns — quantile bins merged until every class has ≥ k records

The stage runs on raw columns *before* signal computation, so derived features inherit the DP guarantee. What it does **not** claim — and why — is in [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).

### Signal Processing

`SignalProcessor` in `core/signals.py` computes 32 features from OHLCV data:

**Technical Indicators**: RSI, MACD (line/signal/histogram), Bollinger Bands (%B and width), ATR, Stochastic (%K/%D), ADX

**Volume Signals**: OBV, VWAP, order flow imbalance

**Volatility**: Realized volatility (5/20-bar), volatility ratio

**Momentum**: ROC (5/20-bar), price momentum (10/20-bar)

**Directional Change**: DC state machine with overshoot and bars-since-event tracking

**Regime Detection**: Volatility-based classification into calm/normal/turbulent

**Asset-Aware**: `AssetConfig` calibrates thresholds per instrument class (forex, crypto, equity, commodity)

## Repository Structure

```
Dignity/
├── __init__.py               # Package root (v0.1.0)
├── setup.py                  # Installation config (console scripts: dignity-train, dignity-export, dignity-backtest)
├── requirements.txt          # Dependencies
├── pytest.ini                # Test configuration
├── pyproject.toml            # Ruff config
│
├── core/                     # Core utilities
│   ├── config.py            # YAML-based configuration (DignityConfig)
│   ├── signals.py           # 32-feature signal processor (SignalProcessor)
│   └── privacy.py           # Privacy primitives (pseudonymization, Laplace, k-anonymity)
│
├── data/                     # Data pipeline
│   ├── pipeline.py          # TransactionPipeline (signals, scaling, windowing)
│   ├── loader.py            # PyTorch Dataset and DataLoader
│   └── source/
│       ├── synthetic.py     # SyntheticGenerator (OHLCV + transaction data)
│       ├── crypto.py        # CryptoSource (CSV and exchange data)
│       └── metaapi.py       # MetaApiSource (live/historical forex data)
│
├── models/                   # Neural architectures
│   ├── dignity.py           # Main model (Dignity class, cascade loss)
│   ├── backbone/
│   │   ├── cnn1d.py        # 1D-CNN for local patterns
│   │   ├── lstm.py         # Stacked LSTM for temporal modeling
│   │   ├── attention.py    # Additive attention mechanism
│   │   └── hybrid.py       # DignityBackbone (CNN→LSTM→Attention)
│   └── head/
│       ├── risk.py         # Dual-output risk head (VaR + position limit)
│       ├── forecast.py     # Multi-step forecasting head
│       ├── policy.py       # Actor-critic policy head
│       ├── regime.py       # Market regime classification head
│       └── alpha.py        # Alpha scoring head
│
├── train/                    # Training infrastructure
│   ├── engine.py            # Training/validation loops with AMP
│   └── cli.py               # CLI entry point (dignity-train)
│
├── export/                   # Deployment
│   └── to_onnx.py           # ONNX export with verification and benchmarking
│
├── backtest/                 # Backtesting framework
│
├── config/                   # YAML configurations (6 files)
│   ├── base.yaml            # Default configuration template
│   ├── train_risk.yaml      # Risk scoring
│   ├── train_forecast.yaml  # Forecasting
│   ├── train_quant_paper.yaml  # Paper trading (cascade)
│   ├── train_quant.yaml     # Live execution (gated)
│   └── colab.yaml           # Google Colab optimized
│
├── tests/                    # Test suite (384 tests)
│   ├── conftest.py          # Pytest fixtures
│   ├── test_core.py         # Core utilities
│   ├── test_data.py         # Data pipeline
│   ├── test_models.py       # Model components
│   ├── test_train.py        # Training
│   ├── test_export.py       # ONNX export
│   └── test_backtest.py     # Backtesting
│
└── docs/                     # Documentation
```

**Package Stats:**
- 48 Python modules (~8,000 lines of code)
- 384 tests collected; 100% line coverage enforced on core/privacy.py in CI
- 6 YAML configs for different tasks and environments
- 32 input features per timestep (default config)

## Advanced Usage

### Custom Signal Processing

```python
from core.signals import SignalProcessor
import numpy as np

prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 103.0])
volumes = np.array([1000.0, 1200.0, 800.0, 1100.0, 900.0, 1300.0])

# Technical indicators
rsi = SignalProcessor.rsi(prices, period=14)
macd_line, macd_signal, macd_hist = SignalProcessor.macd(prices)
pct_b, bandwidth = SignalProcessor.bollinger_bands(prices, window=20)

# Volatility and momentum
vol = SignalProcessor.volatility(prices, window=3)
momentum = SignalProcessor.price_momentum(prices, window=2)
realized_vol = SignalProcessor.realized_volatility(prices, window=20)

# Directional change state machine
dc = SignalProcessor.dc_state_machine(prices, threshold=0.005)

# Regime detection
regime = SignalProcessor.regime_detection(vol)

# Full 32-feature computation
signals = SignalProcessor.process_sequence(volumes, prices)
```

### Configuration Management

```python
from core.config import DignityConfig

# Load config from YAML
config = DignityConfig.from_yaml("config/train_risk.yaml")

# Access nested configs
print(f"Task: {config.model.task}")
print(f"Hidden size: {config.model.hidden_size}")
print(f"Batch size: {config.data.batch_size}")
print(f"Learning rate: {config.train.lr}")

# Save modified config
config.train.epochs = 100
config.to_yaml("config/custom_config.yaml")
```

### Custom Data Source

```python
from data.source.crypto import CryptoSource
import pandas as pd


# Implement custom data source
class CustomSource:
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Must return DataFrame with columns: timestamp, open, high, low, close, volume
        pass
```

### Model Customization

```python
from models.dignity import Dignity
import torch

# Create model with custom config
model = Dignity(
    task="risk",
    input_size=32,  # Number of features
    hidden_size=256,  # Backbone hidden dimension
    n_layers=2,  # LSTM layers
    dropout=0.3,
)

# Forward pass
x = torch.randn(4, 100, 32)  # [batch, seq_len, features]
predictions, attention_weights = model(x)

# Cascade mode returns a dict
cascade_model = Dignity(task="cascade", input_size=32)
outputs = cascade_model(x)  # dict with regime_probs, var_estimate, alpha_score, etc.
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_core.py -v
pytest tests/test_data.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Design Philosophy

**Dignity** is built on three principles:

1. **Minimal**: ~8,000 lines of focused code. Every module has a single purpose.
2. **Deniable**: Local-only inference, self-contained artifact, nothing phones home — three properties, three tests. Defined precisely in [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).
3. **Deployable**: ONNX export, production-ready from day one.

You are not refactoring code. You are **distilling intent**.

## Contributing

Contributions are welcome. Please:
- Run tests before submitting: `pytest tests/ -v`
- Install the hooks once: `pip install pre-commit && pre-commit install`
- Follow existing code style: `ruff format .` and `ruff check .`
- Add tests for new features
- Update docs accordingly

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research or applications, please cite:

```bibtex
@software{dignity_core,
  title = {Dignity Core: Privacy-Preserving Sequence Modeling for Transactional Behavior},
  author = {crichalchemist},
  year = {2026},
  url = {https://github.com/crichalchemist/Dignity}
}
```

## Acknowledgments

- Refactored from [Sequence](https://github.com/crichalchemist/Sequence) research prototype
- PyTorch team for the deep learning framework
- Open source community for privacy-preserving ML techniques

---

**Built for deniability. Optimized for persistence.**
