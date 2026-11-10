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
from core.config import DignityConfig
from data.source.synthetic import SyntheticGenerator

cfg = DignityConfig.from_yaml("config/synthetic_easy.yaml")
df = SyntheticGenerator(cfg.data.synthetic).generate()
```

The generator writes an event record: one row per step, with `block_id`,
`is_anomalous`, `anomaly_type`, `anomaly_id` and `block_label` beside the raw columns.
`data/labels.py` derives the risk and forecast labels of each window from those rows.

**Calibration** - `data/calibration/reference_stats.json` holds the bands the
synthetic generator's statistics must fall inside. `python -m scripts.calibrate`
builds it from public Bitstamp hourly bars (Kraken's OHLC endpoint ignores
`since` and serves only the latest 720 candles, so no 12-month window is
reachable). It covers BTC/USD and ETH/USD over 2023-11..2024-11 and
2024-11..2025-11, takes the median of each statistic over 160-bar chunks
(`data/stats.py`) after dropping zero-volume bars, and widens the min/max of
the four values by 20% of their range. The `fetched=` dates in the sources are UTC. `fee_rate` and `tx_count` are
only plausibility-checked (v1.1: DGNTY-44).

**Presets and the data gate** - `config/synthetic_{easy,medium,hard}.yaml` are the
three difficulty presets. `python -m eval.gates` generates each one, splits it by block
and fits two baselines on the train blocks: `window_logistic` (scored on val windows)
and `block_tree` (scored on val blocks). The gate passes only if all four checks hold:
each baseline's AUROC on `hard` lies in [0.55, 0.80]; AUROC falls strictly from
`easy` to `medium` to `hard` for both baselines; and the `window_logistic` recall at
10% FPR on `hard`, taken per onset-position bucket (five buckets, windows whose anomaly
starts inside the window), has a max-min spread of at most 0.1, so position alone does
not predict the label. The gate is a tuning tool: it defaults to the tuning seed 1000, the
CLI refuses the evaluation seed 2026 (exit code 2), and the test split is never read.
The values the shipped presets reach on seed 1000 are recorded in
[`results/v1.0/data_gate_tuning.json`](../results/v1.0/data_gate_tuning.json).

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

**head/forecast.py** - Early Warning
```python
from models.head.forecast import ForecastHead

head = ForecastHead(input_size=256, horizon=10)  # P(onset within h rows), h = 1..10
```

**head/policy.py** - Decision Policy
```python
from models.head.policy import PolicyHead

head = PolicyHead(input_size=256, n_actions=3)
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
The training loop is `train/loop.py::fit` (`train_epoch` + `predict`); this section is rewritten from the code in
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
python -m export.to_onnx --checkpoint runs/risk/best.pt --output dignity_risk.onnx
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
Dense layer 2 (hidden_dim → horizon)
    ↓
Cumulative hazard: 1 - exp(-cumsum(softplus(z)))
    ↓
Output: (batch, horizon) probabilities
```

#### PolicyHead
```
Input: (batch, hidden_dim)
    ↓
Dense layer 1 (hidden_dim → 128)
    ↓
ReLU + Dropout
    ↓
Linear(hidden → 3)
    ↓
Output: (batch, 3) logits
```

## Training Pipeline

The training loop is `train/loop.py::fit` (`train_epoch` + `predict`); this section is rewritten from the code in
v1.0's documentation milestone.

## Data Flow Example

```python
import numpy as np
import torch

from core.config import DignityConfig
from core.privacy import PrivacyBudget, PrivacyManager
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity

cfg = DignityConfig.from_yaml("config/synthetic_easy.yaml")

# 1. Event-record frame: one row per step, raw columns plus block_id and the labels
df = SyntheticGenerator(cfg.data.synthetic).generate()

# 2. Privacy (see docs/PRIVACY.md). In training this is TransactionPipeline's
#    privacy stage, driven by the `privacy:` config block and run once before
#    signals. Standalone, the same primitive looks like this:
pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0))
noisy_volume = pm.add_laplace_noise(
    df["volume"].to_numpy(), epsilon=1.0, bounds=(0.0, 1000.0)
)

# 3. Block split 70/15/15: the scaler is fit on train blocks, and signals and
#    windows are computed inside each block. Labels come from the event record.
pipeline = TransactionPipeline(seq_len=cfg.data.seq_len, features=cfg.data.features)
data = pipeline.process_blocks(
    df,
    cfg.data.split,
    cfg.model.horizon,
    cfg.data.label_overlap,
    np.random.default_rng(cfg.data.synthetic.seed),
)

# 4. Feed to model: Dignity.forward returns (predictions, attention_weights)
#    data.train.X: (num_windows, seq_len, num_features)
model = Dignity(task="risk", input_size=len(data.feature_names))
x = torch.as_tensor(data.train.X[:64])
y = torch.as_tensor(data.train.y_risk[:64], dtype=torch.float32)
predictions, attention = model(x)

# 5. Compute loss and update
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
