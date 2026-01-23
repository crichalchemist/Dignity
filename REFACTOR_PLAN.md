# 🔧 **Refactoring Plan: Sequence → Dignity**
## *From Bulky Research Prototype to Streamlined Production Core*

### **Core Problem**
The current `Sequence` codebase is:
- **Over-engineered**: Mixes research experimentation (multi-agent A3C, sentiment fusion) with core sequence modeling.
- **Tightly Coupled**: Data, model, and training logic are interwoven.
- **Resource-Intensive**: Assumes multi-GPU, large memory, persistent storage.
- **Overfitted to FX**: Designed for currency pairs, not transaction behavior.

### **Refactoring Goal**
Transform `Sequence` into **Dignity Core**:  
A **modular, minimal, and deployable** framework for training and serving sequence models on **transactional behavior patterns**, optimized for:
- Low-resource training (Colab T4/A100)
- Fast inference (ONNX, <50ms)
- Privacy-preserving data flow
- Easy integration into payment gateways

---

## 🧱 **1. Structural Overhaul: From Monolith to Modular**

### **Before (Current Structure)**
```
Sequence/
├── models/
│   └── agent_hybrid.py          # 800+ lines, mixes CNN, LSTM, A3C, attention
├── train/
│   └── run_training.py          # Hardcoded paths, multiple entrypoints
├── rl/
│   └── run_a3c_training.py      # Separate RL loop
├── data/
│   └── prepare_dataset.py       # FX-focused, Yahoo Finance dependency
```

### **After (Refactored: Dignity Core)**
```
dignity/
├── core/                        # Shared utilities
│   ├── config.py                # Unified config (YAML/JSON)
│   ├── signals.py               # Signal processing (volatility, entropy)
│   └── privacy.py               # Hashing, anonymization
│
├── data/                        # Clean, modular data pipeline
│   ├── source/                  # Raw data interfaces
│   │   ├── synthetic.py         # Generate merchant sequences
│   │   ├── crypto.py            # CCXT wrapper for BTC/XMR
│   │   └── gdeltsent.py         # Optional sentiment (GDELT)
│   ├── pipeline.py              # Unified preprocessing pipeline
│   └── loader.py                # PyTorch Dataset + DataLoader
│
├── models/                      # Lean, composable models
│   ├── backbone/                # Replace agent_hybrid.py
│   │   ├── cnn1d.py             # 1D-CNN for local patterns
│   │   ├── lstm.py              # Stacked LSTM
│   │   ├── attention.py         # Additive attention
│   │   └── hybrid.py            # CNN + LSTM + Attention (main)
│   ├── head/                    # Task-specific heads
│   │   ├── risk.py              # Risk score (0–1)
│   │   ├── forecast.py          # Volume/success prediction
│   │   └── policy.py            # A3C policy (optional)
│   └── dignity.py               # Final model: Backbone + Head
│
├── train/                       # Unified training
│   ├── engine.py                # Training loop (AMP, DDP-ready)
│   ├── a3c.py                   # Lightweight A3C (if used)
│   └── cli.py                   # CLI interface: `dignity train --config cfg.yaml`
│
├── export/                      # Deployment
│   └── to_onnx.py               # Export to ONNX
│
├── config/                      # Configs per use case
│   ├── base.yaml
│   ├── train_risk.yaml
│   └── train_forecast.yaml
│
└── tests/                       # Minimal validation
    ├── test_data.py
    └── test_model.py
```

---

## 🗑️ **2. Eliminate Bloat: What’s Removed**

| Component | Reason for Removal |
|---------|----------------------|
| `agent_hybrid.py` (original) | Monolithic; mixes concerns; too many entrypoints |
| Hardcoded Yahoo Finance calls | Not relevant to transaction data |
| Multi-agent A3C framework | Overkill; single-agent is sufficient |
| Redundant logging modules | Replace with `logging` + `rich` |
| Unnecessary visualization | Move to optional `notebooks/` |
| FX-specific feature engineering | Replace with transaction-specific signals |

---

## 🔨 **3. Refactor Key Modules**

### **A. `models/backbone/hybrid.py` — The New Core**
Replaces `agent_hybrid.py` with a clean, composable architecture.

```python
import torch
import torch.nn as nn
from dignity.models.backbone.cnn1d import CNN1D
from dignity.models.backbone.lstm import StackedLSTM
from dignity.models.backbone.attention import AdditiveAttention

class DignityBackbone(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3)
        self.lstm = StackedLSTM(hidden_size, hidden_size, n_layers, dropout)
        self.attn = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, T, F]
        x = self.cnn(x)  # [B, T, H]
        x, _ = self.lstm(x)  # [B, T, H]
        x = self.dropout(x)
        ctx, weights = self.attn(x, mask)  # [B, H], [B, T]
        return ctx, weights  # context vector + attention weights
```

> **Size**: <100 lines.  
> **Focus**: Extract temporal context from transaction sequences.

---

### **B. `models/dignity.py` — Task-Specific Assembly**

```python
class Dignity(nn.Module):
    def __init__(self, task='risk', seq_len=100):
        super().__init__()
        self.backbone = DignityBackbone()
        self.task = task

        if task == 'risk':
            self.head = RiskHead(256)
        elif task == 'forecast':
            self.head = ForecastHead(256, pred_len=5)
        elif task == 'policy':
            self.head = PolicyHead(256, n_actions=3)

    def forward(self, x):
        ctx, attn = self.backbone(x)
        out = self.head(ctx)
        return out, attn
```

> Enables **modular training**: one backbone, multiple heads.

---

### **C. `data/pipeline.py` — Unified Preprocessing**

```python
class TransactionPipeline:
    def __init__(self, config):
        self.scaler = RobustScaler()
        self.seq_len = config['seq_len']

    def fit_transform(self, df):
        # Select features
        X = df[['volume', 'fee_rate', 'entropy', 'tx_count']].values
        self.scaler.fit(X)
        return self.scaler.transform(X)

    def to_sequence(self, X, y_col=None):
        # Generate sliding windows
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len - 5):
            X_seq.append(X[i:i+self.seq_len])
            if y_col is not None:
                y_seq.append(y_col[i+self.seq_len:i+self.seq_len+5])
        return np.array(X_seq), np.array(y_seq)
```

> Decouples data logic from model.

---

### **D. `train/engine.py` — Lean Training Loop**

```python
def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out, _ = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

> No bloat. No multiple entrypoints. One loop.

---

## 🧩 **4. Configuration-Driven Operation**

### `config/train_risk.yaml`
```yaml
model:
  task: risk
  hidden_size: 256
  n_layers: 2

data:
  source: synthetic
  seq_len: 100
  batch_size: 64
  test_size: 0.2

train:
  epochs: 50
  lr: 3e-4
  use_amp: true
  checkpoint_dir: /content/drive/MyDrive/dignity/checkpoints

device: cuda
```

Run training:
```bash
python -m dignity.train.cli --config config/train_risk.yaml
```

---

## 🚀 **5. Deployment: ONNX Export**

### `export/to_onnx.py`
```python
model.eval()
dummy = torch.randn(1, 100, 9)
torch.onnx.export(
    model,
    dummy,
    "dignity_risk.onnx",
    input_names=["input"],
    output_names=["risk_score", "attention"],
    dynamic_axes={"input": {0: "batch", 1: "sequence"}},
    opset_version=13
)
```

> Output: `dignity_risk.onnx` — embeddable in any gateway.

---

## ✅ **6. Migration Plan**

| Step | Action |
|------|--------|
| 1 | Clone `Sequence` → rename to `dignity` |
| 2 | Delete `rl/`, `train/`, `models/` contents |
| 3 | Scaffold new structure (above) |
| 4 | Implement `core/`, `data/`, `models/backbone/` |
| 5 | Port synthetic data generator |
| 6 | Train `Dignity(risk)` on Colab |
| 7 | Export to ONNX, integrate into ShadowPay |

---

## 🧭 **Final Insight**

You are not refactoring code.  
You are **distilling intent**.

The original `Sequence` is a research artifact — exploratory, verbose, academic.  
**Dignity** is an operational reflex — minimal, fast, deniable.

It does not need to be clever.  
It needs to **persist**.

This refactoring strips away everything that does not serve that goal.
