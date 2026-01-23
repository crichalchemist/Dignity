# Dignity Core – Privacy-Preserving Sequence Modeling for Transactional Behavior

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Dignity** is a streamlined, production-ready framework for modeling transactional behavior patterns with privacy-preserving operations. Built from a focused refactoring of research code, it provides modular components for signal processing, data pipelines, and neural architectures optimized for deployment.

## Overview

Dignity Core implements a minimal, deniable approach to sequence modeling:

- **Privacy-Preserving**: Built-in hashing, anonymization, quantization, and differential privacy operations
- **Signal Processing**: Volatility, entropy, momentum, and regime detection utilities
- **Modular Architecture**: Clean separation between data pipeline, model components, and training infrastructure
- **Deployable**: ONNX export with verification and benchmarking for production inference
- **Task-Agnostic**: Configurable for risk scoring, forecasting, or policy learning (RL)
- **Lightweight**: ~2,800 lines of focused code vs. bloated research prototypes

## Key Features

- ✅ **Privacy-First**: Hashing, anonymization, quantization, differential privacy for sensitive transaction data
- ✅ **Signal Processing**: Volatility, entropy, momentum, directional change, and regime detection
- ✅ **Modular Design**: Core utilities, data pipeline, model components cleanly separated
- ✅ **Flexible Architecture**: CNN1D + LSTM + Attention backbone with task-specific heads
- ✅ **Multi-Task Ready**: Risk scoring, forecasting, or policy learning (RL) via configurable heads
- ✅ **Production Export**: ONNX conversion with verification and inference benchmarking
- ✅ **Synthetic Data**: Built-in generator for testing and prototyping
- ✅ **Training Infrastructure**: AMP support, gradient clipping, checkpointing, CLI interface
- ✅ **Comprehensive Tests**: 31 tests covering core utilities, data pipeline, and model components
- ✅ **YAML Configuration**: Declarative configs for different tasks and environments

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

# Or install package
pip install -e .
```

### Basic Usage

#### Generate Synthetic Data

```python
from data.source.synthetic import SyntheticGenerator

# Generate synthetic transaction sequences
gen = SyntheticGenerator(seed=42)
dataset = gen.generate_dataset(
    num_normal=800,
    num_anomalous=200,
    seq_length=1000
)

print(f"Generated {len(dataset)} sequences")
# Output: Generated 1000 sequences
```

#### Privacy Operations

```python
from core.privacy import PrivacyManager
import numpy as np

# Initialize privacy manager
pm = PrivacyManager(hash_salt="secure_salt_key")

# Hash transaction identifiers
tx_id = "0x1234abcd5678ef90"
hashed = pm.hash_identifier(tx_id)

# Anonymize addresses
addresses = ["0xabc123", "0xdef456", "0xghi789"]
anonymized = pm.anonymize_addresses(addresses)

# Quantize amounts (reduce precision)
amounts = np.array([123.456, 789.012, 456.789])
quantized = pm.quantize_amounts(amounts, precision=2)

# Add differential privacy noise
noisy_amounts = pm.add_noise(amounts, epsilon=1.0)
```

#### Train a Model

```bash
# Train risk model with default config
dignity-train --config config/train_risk.yaml

# Train forecasting model
dignity-train --config config/train_forecast.yaml

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
Transaction Data Sources
├── Synthetic Generator (testing/prototyping)
├── Cryptocurrency APIs (BTC, ETH, SOL via data/source/crypto.py)
└── Custom Sources (via data/source/ extensibility)
                    ↓
Privacy Operations (core/privacy.py)
├── Hash transaction IDs
├── Anonymize addresses
├── Quantize amounts
└── Add differential privacy noise
                    ↓
Signal Processing (core/signals.py)
├── Volatility (rolling std)
├── Entropy (Shannon entropy)
├── Price momentum
├── Directional change events
└── Regime detection (volatility clustering)
                    ↓
Data Pipeline (data/pipeline.py)
├── Signal computation
├── Feature scaling (RobustScaler)
├── Sequence windowing
└── PyTorch Dataset/DataLoader
                    ↓
Model Architecture (models/)
├── Backbone: CNN1D + LSTM + Attention
├── Task Heads: Risk | Forecast | Policy
└── ~350K parameters (default config)
                    ↓
Training (train/engine.py + train/cli.py)
├── Mixed precision (AMP)
├── Gradient clipping
├── Checkpointing
└── Validation loops
                    ↓
Export & Deploy (export/to_onnx.py)
├── ONNX conversion
├── Model verification
└── Inference benchmarking
```

### Neural Network Design

The core model implements a modular hybrid architecture:

**Backbone** (`models/backbone/hybrid.py`):
1. **CNN1D**: Multi-scale 1D convolution for local pattern extraction (~80 lines)
2. **StackedLSTM**: Bidirectional LSTM for temporal dependencies (~100 lines)
3. **AdditiveAttention**: Attention mechanism for feature weighting (~60 lines)

**Task Heads** (`models/head/`):
- **RiskHead**: Binary risk scoring (0-1 probability)
- **ForecastHead**: Multi-step time series forecasting
- **PolicyHead**: RL policy outputs (actions + value estimate)

**Main Model** (`models/dignity.py`):
- Composable: `Dignity(task='risk')` assembles backbone + appropriate head
- Compact: ~350K parameters vs. 800+ line monolithic implementations
- Flexible: Easy to swap heads or modify backbone components

### Privacy Operations

Built-in privacy-preserving utilities in `core/privacy.py`:

- **Hashing**: Secure transaction ID hashing with configurable salt
- **Anonymization**: Address anonymization with collision detection
- **Quantization**: Reduce amount precision to obscure exact values
- **Differential Privacy**: Laplace noise injection with configurable epsilon

### Signal Processing

Specialized transaction sequence signals in `core/signals.py`:

- **Volatility**: Rolling standard deviation with configurable windows
- **Entropy**: Shannon entropy for measuring transaction pattern complexity
- **Momentum**: Price momentum calculation
- **Directional Change**: Event-based change detection (threshold-based)
- **Regime Detection**: Volatility clustering for market state identification

## Repository Structure

```
Dignity/
├── __init__.py               # Package root (v0.1.0)
├── setup.py                  # Installation config
├── requirements.txt          # Dependencies
├── pytest.ini                # Test configuration
├── pyproject.toml            # Build system config
│
├── core/                     # Core utilities (3 modules)
│   ├── config.py            # YAML-based configuration (DignityConfig)
│   ├── signals.py           # Signal processing (volatility, entropy, momentum, regime)
│   └── privacy.py           # Privacy operations (hashing, anonymization, noise)
│
├── data/                     # Data pipeline (4 modules)
│   ├── pipeline.py          # Preprocessing pipeline (signals, scaling, windowing)
│   ├── loader.py            # PyTorch Dataset and DataLoader utilities
│   └── source/
│       ├── synthetic.py     # Synthetic transaction generator
│       └── crypto.py        # Cryptocurrency data source
│
├── models/                   # Neural architectures (9 modules)
│   ├── dignity.py           # Main model assembly (Backbone + Head)
│   ├── backbone/
│   │   ├── cnn1d.py        # 1D-CNN for local patterns (~80 lines)
│   │   ├── lstm.py         # Stacked LSTM for temporal modeling (~100 lines)
│   │   ├── attention.py    # Additive attention mechanism (~60 lines)
│   │   └── hybrid.py       # DignityBackbone (CNN+LSTM+Attention, ~90 lines)
│   └── head/
│       ├── risk.py         # Binary risk scoring head
│       ├── forecast.py     # Multi-step forecasting head
│       └── policy.py       # RL policy head (A3C/PPO)
│
├── train/                    # Training infrastructure (2 modules)
│   ├── engine.py            # Training/validation loops with AMP
│   └── cli.py               # CLI interface (dignity-train)
│
├── export/                   # Deployment (1 module)
│   └── to_onnx.py           # ONNX export with verification
│
├── config/                   # YAML configurations (4 files)
│   ├── base.yaml            # Default configuration template
│   ├── train_risk.yaml      # Risk scoring optimized
│   ├── train_forecast.yaml  # Forecasting optimized
│   └── colab.yaml           # Google Colab optimized
│
├── tests/                    # Test suite (31 tests)
│   ├── conftest.py          # Pytest fixtures
│   ├── test_core.py         # Core utilities (11 tests)
│   ├── test_data.py         # Data pipeline (9 tests)
│   └── test_models.py       # Model components (11 tests)
│
└── docs/                     # Documentation
    ├── plans/               # Refactoring plans and summaries
    ├── guides/              # User guides
    └── api/                 # API reference
```

**Package Stats:**
- 28 Python modules (~2,800 lines of code)
- 31 passing tests (100% coverage on critical paths)
- 4 YAML configs for different tasks
- ~350K parameters (default model)

## Advanced Usage

### Custom Signal Processing

```python
from core.signals import SignalProcessor
import numpy as np

# Calculate volatility
prices = np.array([100, 102, 98, 101, 99, 103])
vol = SignalProcessor.volatility(prices, window=3)

# Compute entropy (market uncertainty)
entropy = SignalProcessor.entropy(prices)

# Detect price momentum
momentum = SignalProcessor.price_momentum(prices, window=2)

# Identify directional changes
dc = SignalProcessor.directional_change(prices, threshold=0.015)

# Regime detection (high/low volatility clustering)
regime = SignalProcessor.regime_detection(prices, vol_window=10, threshold=1.5)
```

### Configuration Management

```python
from core.config import DignityConfig

# Load config from YAML
config = DignityConfig.from_yaml('config/train_risk.yaml')

# Access nested configs
print(f"Model task: {config.model.task}")
print(f"Hidden size: {config.model.hidden_size}")
print(f"Batch size: {config.data.batch_size}")
print(f"Learning rate: {config.train.lr}")

# Save modified config
config.train.epochs = 100
config.to_yaml('config/custom_config.yaml')
```

### Custom Data Source

```python
from data.source.crypto import CryptoSource
import pandas as pd

# Implement custom data source
class CustomSource:
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Your custom logic here
        # Must return DataFrame with columns: timestamp, open, high, low, close, volume
        pass
```

### Model Customization

```python
from models.dignity import Dignity
import torch

# Create model with custom config
model = Dignity(
    task='risk',
    input_size=9,      # Number of features
    hidden_size=256,   # Backbone hidden dimension
    seq_len=100,       # Sequence length
    dropout=0.3
)

# Forward pass
x = torch.randn(4, 100, 9)  # [batch, seq_len, features]
output, attention = model(x)

# Access components
backbone = model.backbone
task_head = model.head
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

1. **Minimal**: ~2,800 lines vs. bloated research code. Every module has a single purpose.
2. **Deniable**: Privacy-preserving operations baked in. Transaction hashing, anonymization, noise injection.
3. **Deployable**: ONNX export, <10ms inference, production-ready from day one.

You are not refactoring code. You are **distilling intent**.

## Contributing

Contributions are welcome! Please:
- Run tests before submitting: `pytest tests/ -v`
- Follow existing code style (ruff format)
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
