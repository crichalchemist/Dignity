# Dignity Core Documentation

**Dignity Core** is a privacy-preserving transaction modeling framework designed for flexible, modular deep learning on sensitive data.

## Documentation Index

### Getting Started
- **[Quick Start Guide](QUICK_START.md)** - Installation, basic usage, and first model
- **[Configuration Guide](CONFIGURATION.md)** - YAML configuration reference
- **[Training Guide](TRAINING.md)** - How to train models *(coming soon)*

### Core Concepts
- **[Privacy Operations](PRIVACY.md)** - Hashing, anonymization, noise injection
- **[Signal Processing](SIGNALS.md)** - Volatility, entropy, momentum, regime detection
- **[Architecture Overview](ARCHITECTURE.md)** - Models, backbones, and heads

### Advanced Topics
- **[API Reference](API_REFERENCE.md)** - Complete API documentation *(coming soon)*
- **[Deployment Guide](DEPLOYMENT.md)** - ONNX export and production deployment *(coming soon)*
- **[Testing Guide](TESTING.md)** - Running and writing tests *(coming soon)*
- **[Development Guide](DEVELOPMENT.md)** - Contributing and extending Dignity *(coming soon)*

### Project Information
- **[Plans](plans/)** - Refactor history and planning documents
- **[Archive](archive/)** - Historical documentation from Sequence project

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

