# Configuration Guide

Dignity Core uses YAML configuration files for model and training setup. The `DignityConfig` dataclass parses YAML into typed Python objects.

## Configuration Files

Located in `config/`:

- **base.yaml** – Base model and data configuration
- **train_risk.yaml** – Risk scoring task
- **train_forecast.yaml** – Forecasting task
- **train_quant_paper.yaml** – Paper trading cascade (default for development)
- **train_quant.yaml** – Live execution (gated — see production path docs)
- **colab.yaml** – Google Colab optimized settings

## Configuration Structure

`DignityConfig` nests four sub-configs:

```python
@dataclass
class DignityConfig:
    model: ModelConfig       # Architecture parameters
    data: DataConfig         # Data pipeline parameters
    train: TrainConfig       # Training parameters
    execution: ExecutionConfig  # Live/paper trading parameters
    device: str = "cuda"
    seed: int = 42
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    task: str = "risk"            # risk, forecast, policy, cascade
    input_size: int = 32          # Number of input features
    hidden_size: int = 256        # Backbone hidden dimension
    n_layers: int = 2             # LSTM layers
    dropout: float = 0.1          # Dropout rate
    cnn_kernel_size: int = 3      # CNN kernel size
    task_weights: dict = {        # Cascade head weights
        "regime": 0.2,
        "risk": 0.3,
        "alpha": 0.3,
        "policy": 0.2,
    }
```

### DataConfig

```python
@dataclass
class DataConfig:
    source: str = "synthetic"     # synthetic, crypto, metaapi
    seq_len: int = 100            # Sequence length for windowing
    batch_size: int = 64          # Training batch size
    test_size: float = 0.2        # Validation split fraction
    num_workers: int = 4          # DataLoader workers
    start_date: str = "2016-01-01"  # Historical data start
    features: list = [...]        # 31 feature names (see below)
```

Default feature list:
```
volume, price, fee_rate, tx_count, rsi, macd_line, macd_signal, macd_hist,
bollinger_pct_b, bollinger_width, atr, stoch_k, stoch_d, adx, obv, vwap,
roc_5, roc_20, momentum_10, momentum_20, volatility_5, volatility_20,
vol_ratio, order_flow_imbalance, dc_direction, dc_overshoot,
dc_bars_since_event, volume_volatility, volume_entropy, price_change,
directional_change
```

### TrainConfig

```python
@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-4              # Learning rate
    weight_decay: float = 1e-5
    use_amp: bool = True          # Automatic mixed precision
    gradient_clip: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    save_interval: int = 5
    risk_gate_training: bool = True  # Suppress actions when VaR exceeds max_drawdown
```

### ExecutionConfig

```python
@dataclass
class ExecutionConfig:
    provider: str = "mock"        # metaapi or mock
    metaapi_token: str = ""
    account_id: str = ""
    symbols: list = ["EURUSD"]
    asset_class: str = "forex"    # forex, crypto, equity, commodity
    max_drawdown: float = 0.05
    max_position_size: float = 1.0
    risk_sdk_enabled: bool = True
    paper_trading: bool = True    # Must explicitly set False for live
```

## Task-Specific Configurations

### Risk Scoring (train_risk.yaml)

```yaml
model:
  task: "risk"
  hidden_size: 256
  n_layers: 2
  dropout: 0.1

data:
  seq_len: 100
  batch_size: 64

train:
  epochs: 50
  lr: 3e-4
```

### Forecasting (train_forecast.yaml)

```yaml
model:
  task: "forecast"
  hidden_size: 256
  n_layers: 2

data:
  seq_len: 100
  batch_size: 64

train:
  epochs: 100
  lr: 1e-4
```

### Cascade – Paper Trading (train_quant_paper.yaml)

```yaml
model:
  task: "cascade"
  hidden_size: 256
  n_layers: 2

data:
  source: "synthetic"
  seq_len: 100

execution:
  paper_trading: true
  risk_sdk_enabled: false
```

## Using Configurations

### In Python

```python
from core.config import DignityConfig

# Load from YAML
config = DignityConfig.from_yaml("config/train_risk.yaml")

# Access parameters
print(f"Task: {config.model.task}")
print(f"Hidden size: {config.model.hidden_size}")
print(f"Batch size: {config.data.batch_size}")
print(f"Learning rate: {config.train.lr}")

# Override values
config.train.epochs = 100
config.device = "cpu"
```

### CLI Training

```bash
# Use default config
dignity-train --config config/train_risk.yaml

# Resume from checkpoint
dignity-train --config config/train_risk.yaml --resume checkpoints/latest.pt
```

## Config Imports

Configs support an `imports` key for inheriting from a base config:

```yaml
# train_risk.yaml
imports:
  - base.yaml

model:
  task: "risk"
```

The imported base config provides defaults; the importing file overrides specific keys.

## Best Practices

1. Start with base.yaml – Use it as a template for new configurations.
2. Separate configs per task – Risk, forecast, policy, and cascade each need different hyperparameters.
3. Version control configs – Track changes in git alongside code.
4. Validate first – Test with small epoch counts before full training runs.
5. Keep credentials out of YAML – Use environment variables for API tokens.

## Next Steps

- [Quick Start Guide](QUICK_START.md) – Use configurations for first training run
- [Architecture Overview](ARCHITECTURE.md) – Understand model components
- [Privacy Operations](PRIVACY.md) – Configure privacy settings
