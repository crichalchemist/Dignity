# Configuration Guide

Dignity Core uses YAML configuration files for flexible model and training setup.

## Configuration Files

Located in `config/`:

- **base.yaml** - Base model and data configuration
- **train_risk.yaml** - Risk modeling task
- **train_forecast.yaml** - Forecasting task
- **colab.yaml** - Google Colab-specific settings

## Configuration Structure

### Complete Example (config/base.yaml)

```yaml
data:
  window_size: 20              # Sequence length
  feature_dim: 12              # Input features per timestep
  num_entities: 1000           # Number of unique entities
  stride: 1                    # Window stride for sampling
  
model:
  backbone:
    type: "dignity"            # Backbone architecture
    hidden_dim: 128            # Hidden layer dimension
    num_layers: 2              # LSTM layers
    dropout: 0.1               # Dropout rate
    attention: true            # Use attention mechanism
    
  head:
    type: "risk"               # Task head type: risk, forecast, or policy
    hidden_dim: 64             # Head-specific hidden dimension
    num_classes: 3             # For classification tasks
    
privacy:
  hash_ids: true               # Hash entity identifiers
  quantize_amounts: true       # Quantize transaction amounts
  num_bins: 20                 # Quantization bins
  add_noise: false             # Differential privacy noise
  epsilon: 1.0                 # Privacy budget (if noise enabled)
  
training:
  batch_size: 32               # Training batch size
  learning_rate: 0.001         # Initial learning rate
  epochs: 50                   # Training epochs
  val_split: 0.2               # Validation split fraction
  seed: 42                     # Random seed
  device: "cuda"               # Device: cuda, cpu, mps
  amp: true                    # Automatic mixed precision
  grad_clip: 1.0               # Gradient clipping threshold
  
signals:
  compute_volatility: true     # Compute rolling volatility
  compute_entropy: true        # Compute transaction entropy
  compute_momentum: true       # Compute momentum signals
  detect_regime: true          # Regime detection
  volatility_window: 10        # Volatility computation window
  entropy_bins: 5              # Entropy histogram bins
```

## Key Parameters

### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 20 | Sequence length for temporal modeling |
| `feature_dim` | int | 12 | Number of input features |
| `num_entities` | int | 1000 | Unique entities in dataset |
| `stride` | int | 1 | Sampling stride for windows |

### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone.type` | str | "dignity" | Backbone architecture |
| `backbone.hidden_dim` | int | 128 | Hidden dimension |
| `backbone.num_layers` | int | 2 | LSTM layers |
| `backbone.dropout` | float | 0.1 | Dropout rate |
| `backbone.attention` | bool | true | Enable attention |
| `head.type` | str | "risk" | Task: risk, forecast, policy |

### Privacy Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hash_ids` | bool | true | Hash entity identifiers |
| `quantize_amounts` | bool | true | Quantize amounts |
| `num_bins` | int | 20 | Quantization bins |
| `add_noise` | bool | false | Add DP noise |
| `epsilon` | float | 1.0 | Privacy budget |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Batch size |
| `learning_rate` | float | 0.001 | Learning rate |
| `epochs` | int | 50 | Training epochs |
| `val_split` | float | 0.2 | Validation split |
| `device` | str | "cuda" | Device: cuda/cpu/mps |
| `amp` | bool | true | Mixed precision |
| `grad_clip` | float | 1.0 | Gradient clipping |

## Task-Specific Configurations

### Risk Modeling (train_risk.yaml)

```yaml
model:
  head:
    type: "risk"
    hidden_dim: 64
    num_classes: 3  # low, medium, high risk
    
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
```

### Forecasting (train_forecast.yaml)

```yaml
model:
  head:
    type: "forecast"
    hidden_dim: 128
    forecast_horizon: 5  # Predict next 5 timesteps
    
training:
  batch_size: 64
  learning_rate: 0.0005
  epochs: 100
```

## Using Configurations

### In Python

```python
from core.config import DignityConfig

# Load from YAML
config = DignityConfig.from_yaml("config/train_risk.yaml")

# Access parameters
print(f"Hidden dim: {config.model.backbone.hidden_dim}")
print(f"Batch size: {config.training.batch_size}")

# Override specific values
config.training.epochs = 100
config.training.device = "cpu"
```

### CLI Training

```bash
# Use default config
python -m train.cli --config config/train_risk.yaml

# Override parameters
python -m train.cli \
    --config config/train_risk.yaml \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005
```

## Best Practices

1. **Start with base.yaml** - Use as template for new configurations
2. **Task-specific configs** - Create separate configs for different tasks
3. **Version control** - Track configuration changes in git
4. **Document changes** - Add comments explaining non-standard parameters
5. **Validate** - Test configurations with small epoch counts first

## Advanced: Custom Configurations

```python
from core.config import DignityConfig

# Create custom config programmatically
config = DignityConfig(
    data={"window_size": 30, "feature_dim": 15},
    model={
        "backbone": {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2
        }
    },
    training={"batch_size": 64, "epochs": 200}
)

# Save to YAML
config.to_yaml("config/custom.yaml")
```

## Next Steps

- **[Training Guide](TRAINING.md)** - Use configurations in training
- **[Architecture Overview](ARCHITECTURE.md)** - Understand model components
- **[API Reference](API_REFERENCE.md)** - Full API details
