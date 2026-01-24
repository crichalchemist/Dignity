# Dignity Core Documentation

**Dignity Core** is a privacy-preserving transaction modeling framework designed for flexible, modular deep learning on sensitive data.

## Documentation Index

### Getting Started
- **[Quick Start Guide](QUICK_START.md)** - Installation, basic usage, and first model
- **[Configuration Guide](CONFIGURATION.md)** - YAML configuration reference

### Core Concepts
- **[Privacy Operations](PRIVACY.md)** - Hashing, anonymization, differential privacy
- **[Signal Processing](SIGNALS.md)** - Volatility, entropy, momentum, regime detection
- **[Architecture Overview](ARCHITECTURE.md)** - Models, backbones, and heads

### Reference
- **[Documentation Summary](DOCUMENTATION_SUMMARY.md)** - Complete documentation overview

## Quick Links

**Installation:**
```bash
# Clone and install
git clone <repository>
cd Dignity
conda create -n dignity python=3.11 -y
conda activate dignity
pip install -r requirements.txt
```

**Train a Model:**
```bash
python -m train.cli --config config/train_risk.yaml --epochs 10
```

**Export to ONNX:**
```python
from export.to_onnx import export_dignity_to_onnx

export_dignity_to_onnx(
    checkpoint_path="checkpoints/dignity_risk_best.pth",
    output_path="dignity_risk.onnx"
)
```

## Project Structure

```
dignity/
├── core/          # Configuration, signals, privacy operations
├── data/          # Data pipeline, loaders, synthetic generation
├── models/        # Neural network architectures
├── train/         # Training engine and CLI
├── export/        # ONNX export utilities
├── config/        # YAML configuration files
└── tests/         # Unit tests
```

## Support

For issues, questions, or contributions, please refer to the project repository.

