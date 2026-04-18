# Architecture Overview

Dignity Core uses a modular architecture with composable backbones and task-specific heads.

## System Architecture

```
Input Data → Privacy → Signals → Pipeline → Model → Export
```

Data flows left to right. Each stage is independent and testable.

1. Load transaction data (synthetic, CSV, MetaApi)
2. Apply privacy operations (hashing, anonymization, DP noise)
3. Compute 32 signal features (RSI, MACD, Bollinger, volatility, regime, etc.)
4. Scale features and create sliding-window sequences
5. Feed sequences to neural network model
6. Train with task-specific objective
7. Export to ONNX for deployment

## Module Structure

### core/ – Foundation

**config.py** – Configuration management via `DignityConfig` dataclass.

```python
from core.config import DignityConfig

config = DignityConfig.from_yaml("config/base.yaml")
print(config.model.task)         # "risk"
print(config.model.hidden_size)  # 256
print(config.data.batch_size)    # 64
```

**signals.py** – 32-feature signal processor.

```python
from core.signals import SignalProcessor
import numpy as np

prices = np.array([100.0, 102.0, 98.0, 101.0])
volumes = np.array([1000.0, 1200.0, 800.0, 1100.0])

# Individual indicators
rsi = SignalProcessor.rsi(prices, period=14)
macd_line, signal, hist = SignalProcessor.macd(prices)
vol = SignalProcessor.volatility(prices, window=3)

# Full 32-feature computation
signals = SignalProcessor.process_sequence(volumes, prices)
```

**privacy.py** – Privacy operations via `PrivacyManager` class.

```python
from core.privacy import PrivacyManager
import numpy as np

hashed = PrivacyManager.hash_identifier("user_123", salt="secret")
amounts = np.array([100.0, 250.0, 75.0])
quantized = PrivacyManager.quantize_amounts(amounts, bins=10)
noisy = PrivacyManager.add_noise(amounts, epsilon=1.0)
```

### data/ – Data Processing

**pipeline.py** – End-to-end data pipeline.

```python
from data.pipeline import TransactionPipeline

pipeline = TransactionPipeline(seq_len=100, features=config.data.features)
pipeline.fit(df_train)
X_scaled = pipeline.transform(df_test)
X_seq, y_seq = pipeline.create_sequences(X_scaled, labels, stride=1)
```

**loader.py** – PyTorch data loading.

```python
from data.loader import create_dataloader

train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
```

**source/synthetic.py** – Synthetic data generation.

```python
from data.source.synthetic import SyntheticGenerator

gen = SyntheticGenerator(seed=42)

# Transaction sequences (for risk/forecast/policy tasks)
df = gen.generate_dataset(num_normal=800, num_anomalous=200, seq_len=100)

# OHLCV series (for cascade task)
ohlcv = gen.generate_ohlcv(n_bars=2000, start_date="2016-01-01")
```

**source/crypto.py** – Cryptocurrency data.

```python
from data.source.crypto import CryptoSource

src = CryptoSource(pair="BTC/USD")
df = src.load_from_csv("data/btc.csv")
```

**source/metaapi.py** – Live/historical forex data via MetaApi.

```python
from data.source.metaapi import MetaApiSource

src = MetaApiSource(token="...", account_id="...", symbol="EURUSD")
await src.connect()
df = await src.get_history(bars=5000)
```

### models/ – Neural Networks

#### Backbone

**backbone/cnn1d.py** – 1D Convolutional Network.

```python
from models.backbone.cnn1d import CNN1D

cnn = CNN1D(input_size=32, hidden_size=256, kernel_size=3, num_layers=2)
```

**backbone/lstm.py** – Stacked LSTM.

```python
from models.backbone.lstm import StackedLSTM

lstm = StackedLSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.1)
```

**backbone/attention.py** – Additive Attention.

```python
from models.backbone.attention import AdditiveAttention

attn = AdditiveAttention(hidden_size=256)
```

**backbone/hybrid.py** – Combined Backbone (CNN → LSTM → Attention).

```python
from models.backbone.hybrid import DignityBackbone

backbone = DignityBackbone(
    input_size=32,
    hidden_size=256,
    n_layers=2,
    dropout=0.1,
)
# Returns: (context [B, H], attention_weights [B, T])
```

#### Task Heads

**head/risk.py** – Dual-output risk head.

```python
from models.head.risk import RiskHead

head = RiskHead(input_size=256, hidden_size=128)
# Returns: (var_estimate [B, 1], position_limit [B, 1])
```

**head/forecast.py** – Sequence forecasting.

```python
from models.head.forecast import ForecastHead

head = ForecastHead(input_size=256, pred_len=5, num_features=3)
# Returns: [B, pred_len, num_features]
```

**head/policy.py** – Actor-critic policy.

```python
from models.head.policy import PolicyHead

head = PolicyHead(input_size=256, n_actions=3)
# Returns: (action_logits [B, n_actions], value [B, 1])
```

**head/regime.py** – Regime classification.

```python
from models.head.regime import RegimeHead

head = RegimeHead(input_size=256, n_regimes=4)
# Returns: regime_probs [B, n_regimes] (softmax)
```

**head/alpha.py** – Alpha scoring.

```python
from models.head.alpha import AlphaHead

head = AlphaHead(input_size=256)
# Returns: alpha_score [B, 1] in [-1, 1] (tanh)
```

#### Complete Model

**dignity.py** – Model assembly.

```python
from models.dignity import Dignity

# Single-head tasks
model = Dignity(task="risk", input_size=32, hidden_size=256)
model = Dignity(task="forecast", input_size=32, hidden_size=256)
model = Dignity(task="policy", input_size=32, hidden_size=256)

# Cascade: Regime → Risk → Alpha → Policy
model = Dignity(task="cascade", input_size=32, hidden_size=256)
outputs = model(x)  # dict with regime_probs, var_estimate, alpha_score, action_logits, value
```

### train/ – Training

**engine.py** – Training and validation loops.

```python
from train.engine import train_epoch, validate_epoch, train_cascade_epoch

# Single-head
train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
val_metrics = validate_epoch(model, val_loader, criterion, device)

# Cascade (Guided Learning)
train_metrics = train_cascade_epoch(model, train_loader, optimizer, task_weights, device=device)
```

**cli.py** – Command-line interface.

```bash
dignity-train --config config/train_risk.yaml
dignity-train --config config/train_quant_paper.yaml --resume checkpoints/latest.pt
```

### export/ – Deployment

**to_onnx.py** – ONNX export.

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
```

## Model Architecture Details

### DignityBackbone

```
Input: (batch, seq_len, input_size)
    ↓
CNN1D (2 layers) → local temporal feature extraction
    ↓
StackedLSTM (n_layers, unidirectional) → sequence modeling
    ↓
Dropout
    ↓
AdditiveAttention → weighted context vector
    ↓
Output: context [batch, hidden_size], attention_weights [batch, seq_len]
```

### Task Heads

#### RiskHead

Dual sigmoid outputs for regime-conditioned risk assessment:

```
Input: [batch, input_size]
    ↓
Linear → ReLU → Dropout → Linear → ReLU → Dropout (shared trunk)
    ↓                           ↓
Linear → Sigmoid            Linear → Sigmoid
    ↓                           ↓
var_estimate [B, 1]        position_limit [B, 1]
```

#### ForecastHead

Multi-step multi-feature forecasting:

```
Input: [batch, input_size]
    ↓
Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
    ↓
Reshape to [batch, pred_len, num_features]
```

#### PolicyHead

Actor-critic architecture for RL:

```
Input: [batch, input_size]
    ↓
Linear → ReLU → Dropout (shared)
    ↓               ↓
Linear (actor)   Linear (critic)
    ↓               ↓
action_logits     value
[B, n_actions]    [B, 1]
```

#### Cascade Architecture

```
Backbone context [B, H]
    ↓
RegimeHead → regime_probs [B, 4]
    ↓
cat(context, regime_probs) → RiskHead → (var_estimate, position_limit)
    ↓
cat(context, regime_probs) → AlphaHead → alpha_score [B, 1]
    ↓
cat(context, alpha_score, var_estimate) → PolicyHead → (action_logits, value)
```

Each head receives the backbone context plus upstream outputs, so earlier predictions condition downstream decisions.

## Training Pipeline

```python
from core.config import DignityConfig
from data.pipeline import TransactionPipeline
from data.loader import create_dataloader
from models.dignity import Dignity
from train.engine import train_epoch, validate_epoch
import torch

# 1. Load configuration
config = DignityConfig.from_yaml("config/train_risk.yaml")

# 2. Prepare data
pipeline = TransactionPipeline(seq_len=config.data.seq_len, features=config.data.features)
pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
X_val = pipeline.transform(df_val)

train_loader = create_dataloader(X_train, y_train, batch_size=config.data.batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size=config.data.batch_size)

# 3. Create model
model = Dignity(
    task=config.model.task,
    input_size=len(pipeline.available_features),
    hidden_size=config.model.hidden_size,
).to(config.device)

# 4. Setup optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
criterion = torch.nn.BCELoss()

# 5. Training loop
for epoch in range(config.train.epochs):
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, config.device)
    val_metrics = validate_epoch(model, val_loader, criterion, config.device)
    print(f"Epoch {epoch}: train={train_metrics['loss']:.4f}, val={val_metrics['loss']:.4f}")
```

## Performance Considerations

### Memory Optimization
- Enable AMP (`config.train.use_amp = True`)
- Gradient clipping (`config.train.gradient_clip`)
- Batch size tuning based on available memory

### Speed Optimization
- DataLoader with `num_workers > 0`
- Pin memory for GPU training
- Compile model with `torch.compile()` (PyTorch 2.0+)

## Next Steps

- [Configuration Guide](CONFIGURATION.md) – Configure architecture parameters
- [Quick Start Guide](QUICK_START.md) – Get running in 5 minutes
- [Privacy Operations](PRIVACY.md) – Deep dive into privacy features
- [Signal Processing](SIGNALS.md) – Understand the 32-feature signal set
