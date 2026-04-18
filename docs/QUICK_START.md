# Quick Start Guide

Get started with Dignity Core in 5 minutes.

## Installation

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/crichalchemist/Dignity.git
cd Dignity

# Create conda environment with Python 3.11
conda create -n dignity python=3.11 -y
conda activate dignity
```

### 2. Install

```bash
pip install -e .
```

This installs three console scripts: `dignity-train`, `dignity-export`, and `dignity-backtest`.

**Core dependencies:**
- PyTorch 2.1+
- pandas, numpy, scipy, scikit-learn
- ONNX and onnxruntime (for export)
- pytest, ruff (development)

## Basic Usage

### Generate Synthetic Data

```python
from data.source.synthetic import SyntheticGenerator

gen = SyntheticGenerator(seed=42)

# Transaction sequences (for risk/forecast/policy tasks)
df = gen.generate_dataset(num_normal=800, num_anomalous=200, seq_len=100)
print(f"Generated {len(df)} rows")

# OHLCV series (for cascade task)
ohlcv = gen.generate_ohlcv(n_bars=2000, start_date="2016-01-01")
```

### Train a Model

```bash
# Risk scoring
dignity-train --config config/train_risk.yaml

# Forecasting
dignity-train --config config/train_forecast.yaml

# Cascade (paper trading)
dignity-train --config config/train_quant_paper.yaml
```

### Minimal Training Script

```python
from core.config import DignityConfig
from data.pipeline import TransactionPipeline
from data.loader import create_dataloader
from models.dignity import Dignity
from train.engine import train_epoch, validate_epoch
import torch

# Load configuration
config = DignityConfig.from_yaml("config/train_risk.yaml")

# Prepare data
pipeline = TransactionPipeline(seq_len=config.data.seq_len, features=config.data.features)
pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
X_val = pipeline.transform(df_val)

train_loader = create_dataloader(X_train, y_train, batch_size=config.data.batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size=config.data.batch_size)

# Create model
model = Dignity(
    task=config.model.task,
    input_size=len(pipeline.available_features),
    hidden_size=config.model.hidden_size,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
criterion = torch.nn.BCELoss()

# Training loop
for epoch in range(10):
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, config.device)
    val_metrics = validate_epoch(model, val_loader, criterion, config.device)
    print(f"Epoch {epoch}: train={train_metrics['loss']:.4f}, val={val_metrics['loss']:.4f}")
```

## Privacy Features

```python
from core.privacy import PrivacyManager
import numpy as np

# Hash identifiers
hashed = PrivacyManager.hash_identifier("user_123", salt="secret_salt")

# Anonymize a list of addresses
addresses = ["0xabc123", "0xdef456", "0xghi789"]
anonymized = PrivacyManager.anonymize_addresses(addresses, salt="secret_salt")

# Quantize amounts (k-anonymity)
amounts = np.array([100.50, 250.75, 75.25])
quantized = PrivacyManager.quantize_amounts(amounts, bins=10)

# Add differential privacy noise
noisy = PrivacyManager.add_noise(amounts, epsilon=1.0, sensitivity=1.0)

# Full sanitization pipeline
result = PrivacyManager.sanitize_dataset(amounts, addresses, epsilon=0.1)
```

## Export to ONNX

```bash
python -m export.to_onnx \
    --checkpoint checkpoints/dignity_risk_best.pt \
    --output dignity_risk.onnx \
    --benchmark
```

```python
from export.to_onnx import export_to_onnx, benchmark_onnx_inference

export_to_onnx(model, "model.onnx", input_shape=(1, 100, 32))
stats = benchmark_onnx_inference("model.onnx")
print(f"Mean inference: {stats['mean_ms']:.2f} ms")
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md) – Customize model architecture and training
- [Privacy Operations](PRIVACY.md) – Deep dive into privacy features
- [Signal Processing](SIGNALS.md) – Understand the 32-feature signal set
- [Architecture Overview](ARCHITECTURE.md) – Model components and data flow

## Troubleshooting

**ImportError: No module named 'torch'**
Install PyTorch: `pip install torch>=2.1.0`

**Tests failing after installation**
Run `pytest tests/ -v` to verify all tests pass.

**ONNX export errors**
Ensure ONNX and onnxruntime are installed: `pip install onnx onnxruntime`
