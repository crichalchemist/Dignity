# Architecture Overview

Dignity Core uses a modular architecture with composable backbones and task-specific heads.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Dignity Core                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Data → Privacy → Signals → Pipeline → Model → Export │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Data Flow:
1. Load transaction data (CSV, database, synthetic)
2. Apply privacy operations (hashing, anonymization, DP noise)
3. Compute signals (volatility, entropy, momentum, regime)
4. Create sequences with sliding windows
5. Feed to neural network model
6. Train with task-specific objective
7. Export to ONNX for deployment
```

## Module Structure

### core/ - Foundation

**config.py** - Configuration management
```python
from core.config import DignityConfig

config = DignityConfig.from_yaml("config/base.yaml")
```

**signals.py** - Signal processing
```python
from core.signals import compute_volatility, compute_entropy

df = compute_volatility(df, window=20)
df = compute_entropy(df, window=50, bins=10)
```

**privacy.py** - Privacy operations
```python
from core.privacy import hash_identifiers, add_differential_privacy_noise

df = hash_identifiers(df, columns=["user_id"])
df = add_differential_privacy_noise(df, columns=["amount"], epsilon=1.0)
```

### data/ - Data Processing

**pipeline.py** - End-to-end data pipeline
```python
from data.pipeline import TransactionPipeline

pipeline = TransactionPipeline(config)
data = pipeline.load_and_process("transactions.csv")
```

**loader.py** - PyTorch data loading
```python
from data.loader import create_dataloaders

train_loader, val_loader = create_dataloaders(data, config, split=0.8)
```

**source/synthetic.py** - Synthetic data generation
```python
from data.source.synthetic import SyntheticGenerator

generator = SyntheticGenerator(num_entities=100, num_transactions=10000)
data = generator.generate()
```

**source/crypto.py** - Cryptocurrency data
```python
from data.source.crypto import load_crypto_prices

prices = load_crypto_prices(symbols=["BTC", "ETH"], start_date="2024-01-01")
```

### models/ - Neural Networks

#### Backbone Architectures

**backbone/cnn.py** - 1D Convolutional Network
```python
from models.backbone.cnn import CNN1D

backbone = CNN1D(
    input_dim=12,
    hidden_dim=128,
    num_layers=3,
    kernel_size=3
)
```

**backbone/lstm.py** - Stacked LSTM
```python
from models.backbone.lstm import StackedLSTM

backbone = StackedLSTM(
    input_dim=12,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1
)
```

**backbone/attention.py** - Additive Attention
```python
from models.backbone.attention import AdditiveAttention

attention = AdditiveAttention(hidden_dim=128)
```

**backbone/dignity.py** - Combined Backbone
```python
from models.backbone.dignity import DignityBackbone

# CNN + LSTM + Attention
backbone = DignityBackbone(
    input_dim=12,
    hidden_dim=128,
    num_layers=2,
    dropout=0.1,
    use_attention=True
)
```

#### Task Heads

**head/risk.py** - Risk Classification
```python
from models.head.risk import RiskHead

head = RiskHead(
    input_dim=128,
    hidden_dim=64,
    num_classes=3  # low, medium, high
)
```

**head/forecast.py** - Sequence Forecasting
```python
from models.head.forecast import ForecastHead

head = ForecastHead(
    input_dim=128,
    hidden_dim=128,
    forecast_horizon=5  # Predict next 5 timesteps
)
```

**head/policy.py** - Decision Policy
```python
from models.head.policy import PolicyHead

head = PolicyHead(
    input_dim=128,
    hidden_dim=64,
    num_actions=4
)
```

#### Complete Model

**dignity.py** - Model Assembly
```python
from models.dignity import create_dignity_model

# Automatically combines backbone + head
model = create_dignity_model(
    config,
    task="risk"  # or "forecast", "policy"
)
```

### train/ - Training

**engine.py** - Training loop
```python
from train.engine import train_epoch, validate_epoch

train_loss = train_epoch(model, train_loader, optimizer, device)
val_loss = validate_epoch(model, val_loader, device)
```

**cli.py** - Command-line interface
```bash
python -m train.cli \
    --config config/train_risk.yaml \
    --epochs 50 \
    --batch-size 32
```

### export/ - Deployment

**to_onnx.py** - ONNX export
```python
from export.to_onnx import export_dignity_to_onnx

export_dignity_to_onnx(
    checkpoint_path="checkpoints/dignity_risk_best.pth",
    output_path="dignity_risk.onnx"
)
```

## Model Architecture Details

### DignityBackbone

```
Input: (batch, seq_len, input_dim)
    ↓
Conv1D layers (temporal feature extraction)
    ↓
LSTM layers (sequence modeling)
    ↓
Attention mechanism (focus on important timesteps)
    ↓
Output: (batch, hidden_dim)
```

**Components:**
1. **CNN1D** - Extract local temporal patterns
2. **StackedLSTM** - Model long-range dependencies
3. **AdditiveAttention** - Weight important timesteps

### Task Heads

#### RiskHead
```
Input: (batch, hidden_dim)
    ↓
Dense layer 1 (hidden_dim → hidden_dim // 2)
    ↓
ReLU + Dropout
    ↓
Dense layer 2 (hidden_dim // 2 → num_classes)
    ↓
Softmax
    ↓
Output: (batch, num_classes)
```

#### ForecastHead
```
Input: (batch, hidden_dim)
    ↓
Dense layer 1 (hidden_dim → hidden_dim)
    ↓
ReLU + Dropout
    ↓
Dense layer 2 (hidden_dim → forecast_horizon)
    ↓
Output: (batch, forecast_horizon)
```

#### PolicyHead
```
Input: (batch, hidden_dim)
    ↓
Dense layer 1 (hidden_dim → hidden_dim // 2)
    ↓
Tanh + Dropout
    ↓
Dense layer 2 (hidden_dim // 2 → num_actions)
    ↓
Softmax
    ↓
Output: (batch, num_actions)
```

## Training Pipeline

```python
from core.config import DignityConfig
from data.pipeline import TransactionPipeline
from data.loader import create_dataloaders
from models.dignity import create_dignity_model
from train.engine import train_epoch, validate_epoch
import torch

# 1. Load configuration
config = DignityConfig.from_yaml("config/train_risk.yaml")

# 2. Prepare data
pipeline = TransactionPipeline(config)
data = pipeline.load_and_process("transactions.csv")

# 3. Create data loaders
train_loader, val_loader = create_dataloaders(data, config, split=0.8)

# 4. Create model
model = create_dignity_model(config, task="risk")
model = model.to(config.training.device)

# 5. Setup optimizer and loss
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate
)
criterion = torch.nn.CrossEntropyLoss()

# 6. Training loop
for epoch in range(config.training.epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, config.training.device)
    val_loss = validate_epoch(model, val_loader, criterion, config.training.device)
    
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        best_val_loss = val_loss
```

## Data Flow Example

```python
# 1. Raw transaction data
raw_data = {
    "user_id": ["alice", "bob", "charlie"],
    "merchant_id": ["shop_1", "shop_2", "shop_1"],
    "amount": [100.50, 250.75, 75.25],
    "timestamp": ["2024-01-01 10:00", "2024-01-01 10:30", "2024-01-01 11:00"]
}

# 2. Apply privacy
data = hash_identifiers(data, ["user_id", "merchant_id"])
data = add_differential_privacy_noise(data, ["amount"], epsilon=1.0)

# 3. Compute signals
data = compute_volatility(data, window=20)
data = compute_entropy(data, window=50, bins=10)
data = detect_regime(data, method="volatility")

# 4. Create sequences
# Shape: (num_sequences, window_size, num_features)
sequences = create_sequences(data, window_size=20, stride=1)

# 5. Feed to model
model_output = model(sequences)

# 6. Compute loss and update
loss = criterion(model_output, targets)
loss.backward()
optimizer.step()
```

## Customization Examples

### Custom Backbone

```python
import torch.nn as nn

class CustomBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Average pool across sequence
        pooled = x.mean(dim=1)  # (batch, input_dim)
        return self.encoder(pooled)  # (batch, hidden_dim)
```

### Custom Task Head

```python
class CustomHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.head(x)
```

## Performance Considerations

### Memory Optimization
- Use gradient checkpointing for large models
- Enable AMP (automatic mixed precision)
- Batch size tuning based on GPU memory

### Speed Optimization
- DataLoader with `num_workers > 0`
- Pin memory for GPU training
- Compile model with `torch.compile()` (PyTorch 2.0+)

### Distributed Training
```python
# Multi-GPU training
model = nn.DataParallel(model)

# Or with DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
```

## Next Steps

- **[Configuration Guide](CONFIGURATION.md)** - Configure architecture parameters
- **[Training Guide](TRAINING.md)** - Train and evaluate models
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Deployment Guide](DEPLOYMENT.md)** - Export and deploy models
