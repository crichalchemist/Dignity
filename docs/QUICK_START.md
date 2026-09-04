# Quick Start Guide

Get started with Dignity Core in 5 minutes.

## Installation

### 1. Clone and Setup Environment

```bash
git clone <repository>
cd Dignity

# Create conda environment with Python 3.11
conda create -n dignity python=3.11 -y
conda activate dignity
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- PyTorch 2.2+
- pandas, numpy, scipy, scikit-learn
- ONNX and onnxruntime (for export)
- pytest, ruff (development)

## Basic Usage

### Generate Synthetic Data

```python
from data.source.synthetic import SyntheticGenerator

# Generate synthetic transaction data
generator = SyntheticGenerator(num_entities=100, num_transactions=10000, seed=42)

data = generator.generate()
print(f"Generated {len(data)} transactions")
```

### Configure and Train

```bash
# Train a risk modeling task
python -m train.cli \
    --config config/train_risk.yaml \
    --epochs 10 \
    --batch-size 32
```

### Minimal Training Script

```python
from core.config import DignityConfig
from data.pipeline import TransactionPipeline
from data.loader import create_dataloaders
from models.dignity import create_dignity_model
from train.engine import train_epoch, validate_epoch
import torch

# Load configuration
config = DignityConfig.from_yaml("config/base.yaml")

# Prepare data
pipeline = TransactionPipeline(config)
transactions = pipeline.load_and_process("data/transactions.csv")
train_loader, val_loader = create_dataloaders(transactions, config, split=0.8)

# Create model
model = create_dignity_model(config, task="risk")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, config.device)
    val_loss = validate_epoch(model, val_loader, config.device)
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
```

## Privacy Features

Turn on the privacy stage with a `privacy:` block in your config; it runs on raw
columns before signals are computed. See `config/train_risk.yaml` for a live
example and [PRIVACY.md](PRIVACY.md) for what each mechanism guarantees.

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0), key=b"16+-byte-secret-key")
pseudonyms = pm.pseudonymize_many(["user_a", "merchant_1"])
noisy = pm.add_laplace_noise(
    np.array([100.0, 250.0]), epsilon=1.0, bounds=(0.0, 1000.0)
)
generalized = PrivacyManager.generalize_amounts(
    np.random.uniform(10, 100, 200), bins=10, k=5
)
```

## Export to ONNX

```python
from export.to_onnx import export_dignity_to_onnx

# Export trained model
export_dignity_to_onnx(
    checkpoint_path="checkpoints/dignity_risk_best.pth",
    output_path="dignity_risk.onnx",
    opset_version=14,
)
```

## Next Steps

- **[Configuration Guide](CONFIGURATION.md)** - Customize model architecture and training
- **[Privacy Operations](PRIVACY.md)** - Deep dive into privacy features
- **[Training Guide](TRAINING.md)** - Advanced training techniques
- **[API Reference](API_REFERENCE.md)** - Complete API documentation

## Troubleshooting

**Issue:** `ImportError: No module named 'torch'`
- **Solution:** Install PyTorch: `pip install torch>=2.2.0`

**Issue:** Tests failing after installation
- **Solution:** Run `pytest tests/` to verify all tests pass

**Issue:** ONNX export errors
- **Solution:** Ensure ONNX and onnxruntime are installed: `pip install onnx onnxruntime`

For more help, see the [Testing Guide](TESTING.md).
