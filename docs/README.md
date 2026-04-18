# Dignity Core Documentation

**Dignity Core** is a privacy-preserving deep learning framework for modeling transactional behavior patterns.

## Documentation Index

### Getting Started
- [Quick Start Guide](QUICK_START.md) – Installation, basic usage, and first model
- [Configuration Guide](CONFIGURATION.md) – YAML configuration reference

### Core Concepts
- [Privacy Operations](PRIVACY.md) – Hashing, anonymization, differential privacy
- [Signal Processing](SIGNALS.md) – 32 OHLCV-derived features
- [Architecture Overview](ARCHITECTURE.md) – Models, backbones, and heads

## Quick Links

**Install:**
```bash
git clone https://github.com/crichalchemist/Dignity.git
cd Dignity
pip install -e .
```

**Train a model:**
```bash
dignity-train --config config/train_risk.yaml
```

**Export to ONNX:**
```bash
python -m export.to_onnx --checkpoint checkpoints/model.pt --output model.onnx
```

## Project Structure

```
Dignity/
├── core/          # Configuration, signals, privacy operations
├── data/          # Data pipeline, loaders, data sources
├── models/        # Neural network architectures (backbone + heads)
├── train/         # Training engine and CLI
├── export/        # ONNX export utilities
├── backtest/      # Backtesting framework
├── config/        # YAML configuration files (6 presets)
├── tests/         # 324 tests across 6 test files
└── docs/          # Documentation
```

## Support

For issues, questions, or contributions, refer to the project repository.
