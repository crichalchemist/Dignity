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
2. Apply the privacy stage (Laplace noise, k-anonymous generalization)
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
from core.signals import SignalProcessor

# All static: they take arrays, not frames. TransactionPipeline.compute_signals
# applies them column by column.
volatility = SignalProcessor.volatility(df["price"].to_numpy(), window=20)
entropy = SignalProcessor.entropy(df["volume"].to_numpy(), bins=10)
```

**privacy.py** - Privacy primitives
```python
from core.privacy import PrivacyBudget, PrivacyManager

pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0), key=b"16+-byte-secret-key")
pseudonym = pm.pseudonymize("user_id_value")
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))
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

**Calibration** - `data/calibration/reference_stats.json` holds the bands the
synthetic generator's statistics must fall inside. `python -m scripts.calibrate`
builds it from public Bitstamp hourly bars (Kraken's OHLC endpoint ignores
`since` and serves only the latest 720 candles, so no 12-month window is
reachable). It covers BTC/USD and ETH/USD over 2023-11..2024-11 and
2024-11..2025-11, takes the median of each statistic over 160-bar chunks
(`data/stats.py`) after dropping zero-volume bars, and widens the min/max of
the four values by 20% of their range. The `fetched=` dates in the sources are UTC. `fee_rate` and `tx_count` are
only plausibility-checked (v1.1: DGNTY-44).

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

backbone = CNN1D(input_dim=12, hidden_dim=128, num_layers=3, kernel_size=3)
```

**backbone/lstm.py** - Stacked LSTM
```python
from models.backbone.lstm import StackedLSTM

backbone = StackedLSTM(input_dim=12, hidden_dim=128, num_layers=2, dropout=0.1)
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
    input_dim=12, hidden_dim=128, num_layers=2, dropout=0.1, use_attention=True
)
```

#### Task Heads

**head/risk.py** - Risk Classification
```python
from models.head.risk import RiskHead

head = RiskHead(
    input_dim=128,
    hidden_dim=64,
    num_classes=3,  # low, medium, high
)
```

**head/forecast.py** - Sequence Forecasting
```python
from models.head.forecast import ForecastHead

head = ForecastHead(
    input_dim=128,
    hidden_dim=128,
    forecast_horizon=5,  # Predict next 5 timesteps
)
```

**head/policy.py** - Decision Policy
```python
from models.head.policy import PolicyHead

head = PolicyHead(input_dim=128, hidden_dim=64, num_actions=4)
```

#### Complete Model

**dignity.py** - Model Assembly
```python
from models.dignity import create_dignity_model

# Automatically combines backbone + head
model = create_dignity_model(
    config,
    task="risk",  # or "forecast", "policy"
)
```

### train/ - Training

**engine.py** - Training loop
The training loop is `train/cli.py` (`train_epoch` + `predict`); this section is rewritten from the code in
v1.0's documentation milestone.

**cli.py** - Command-line interface
```bash
python -m train.cli \
    --config config/train_risk.yaml \
    --epochs 50 \
    --batch-size 32
```

### export/ - Deployment

**to_onnx.py** - ONNX export
```bash
python -m export.to_onnx --checkpoint runs/risk/dignity_risk_best.pt --output dignity_risk.onnx
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

The training loop is `train/cli.py` (`train_epoch` + `predict`); this section is rewritten from the code in
v1.0's documentation milestone.

## Data Flow Example

```python
import torch

from core.privacy import PrivacyBudget, PrivacyManager
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity

# 1. Raw transaction frame: volume, price, fee_rate, tx_count per step, plus a label
df = SyntheticGenerator(seed=42).generate_dataset(
    num_normal=800, num_anomalous=200, seq_len=120
)

# 2. Privacy (see docs/PRIVACY.md). In training this is TransactionPipeline's
#    privacy stage, driven by the `privacy:` config block and run once before
#    signals. Standalone, the same primitive looks like this:
pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0))
noisy_volume = pm.add_laplace_noise(
    df["volume"].to_numpy(), epsilon=1.0, bounds=(0.0, 1000.0)
)

# 3. Signals, scaling and sliding windows
#    Shape: (num_windows, seq_len, num_features)
pipeline = TransactionPipeline(
    seq_len=100, features=["volume", "fee_rate", "tx_count", "volatility", "momentum"]
)
sequences, targets = pipeline.process(df.drop(columns="label"), df["label"].to_numpy())

# 4. Feed to model: Dignity.forward returns (predictions, attention_weights)
model = Dignity(task="risk", input_size=len(pipeline.available_features))
x = torch.as_tensor(sequences[:64], dtype=torch.float32)
y = torch.as_tensor(targets[:64], dtype=torch.float32)
predictions, attention = model(x)

# 5. Compute loss and update
loss = criterion(predictions.squeeze(-1), y)
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
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(128, output_dim),
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
