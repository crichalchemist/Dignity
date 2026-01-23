## 🧩 **Phase 0: Define Dignity’s Input Schema**
Dignity does not need raw content. It needs **behavioral rhythm**.

### Input: Sequence of Transaction Events
Each input is a time-ordered sequence of transactions from a single merchant (or pseudonymous operator).

| Feature | Type | Description |
|-------|------|-------------|
| `timestamp` | float (Unix ms) | When the transaction occurred (milliseconds since epoch) |
| `volume` | float | Amount in USD-equivalent |
| `instrument` | int | 0=BTC, 1=XMR, 2=gift_card, 3=USDT, 4=revolut_transfer |
| `success` | float | 1.0 = confirmed, 0.0 = failed/timeout |
| `fee_rate` | float | % or flat fee paid |
| `tx_count` | int | Number of sub-transactions (e.g., multiple gift cards) |
| `entropy` | float | Address reuse / clustering entropy (for crypto) |
| `geohint` | int | 0=Tor, 1=VPS, 2=Residential, 3=Unknown |
| `user_agent_hash` | int | Hash of client identifier (not stored) |

> **Sequence Length**: 100–200 timesteps  
> **Output**: Predict next 5 steps of `volume`, `success`, and a scalar `risk_score` (0–1)

---

## 📥 **Phase 1: Local Data Collection Strategy**

### A. **Synthetic Merchant Generator (Local)**
Create a Python module that generates realistic transaction sequences.

#### File: `data/local_gen.py`
```python
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
import json

class MerchantSimulator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate(self, mid, risk_profile='high', days=90):
        hours = days * 24
        base_volume = self.rng.lognormal(2, 1.2)  # USD-equiv
        timestamps = pd.date_range('2025-01-01', periods=hours, freq='H')

        # Risk-based behavior
        if risk_profile == 'high':
            volume = base_volume * (1 + self.rng.exponential(0.8, hours))
            volume[self.rng.choice(hours, size=int(0.05*hours))] *= 5  # Spikes
            success = self.rng.beta(4, 2, hours)  # Higher failure rate
        elif risk_profile == 'fraud':
            volume = self.rng.gamma(2, 2, hours) * base_volume
            volume[-20:] = 0  # Exit scam
            success = self.rng.beta(2, 4, hours)
        else:  # low
            volume = self.rng.normal(base_volume, base_volume*0.1, hours)
            success = self.rng.beta(8, 1, hours)

        # Instruments
        instruments = self.rng.choice(
            [0, 1, 2, 3],
            size=hours,
            p=[0.5, 0.1, 0.2, 0.2] if risk_profile in ['high', 'fraud'] else [0.7, 0.05, 0.05, 0.2]
        )

        # Entropy (address reuse heuristic)
        entropy = self.rng.uniform(0.3, 0.9, hours)
        if risk_profile == 'fraud':
            entropy[-10:] = self.rng.uniform(0.8, 0.99, 10)  # High churn before exit

        df = pd.DataFrame({
            'timestamp': timestamps.astype('int64') // 10**6,  # Convert to milliseconds for CCXT compatibility
            'volume': np.maximum(volume, 0.01),
            'instrument': instruments,
            'success': np.clip(success, 0, 1),
            'fee_rate': self.rng.uniform(0.02, 0.15, hours),
            'tx_count': self.rng.poisson(1.2, hours) + 1,
            'entropy': entropy,
            'geohint': self.rng.choice([0,1,2,3], hours, p=[0.6, 0.2, 0.15, 0.05]),
            'ua_hash': self.rng.integers(0, 2**32, hours)
        })

        return df

# Generate 500 merchants
sim = MerchantSimulator(seed=42)
Path("data/raw").mkdir(parents=True, exist_ok=True)

for i in range(500):
    profile = sim.rng.choice(['low', 'high', 'fraud'], p=[0.4, 0.4, 0.2])
    df = sim.generate(mid=i, risk_profile=profile)
    df.to_parquet(f"data/raw/merchant_{i:04d}.parquet")
```

> Run: `python local_gen.py`  
> Output: 500 × 90-day merchant sequences in `data/raw/`

---

### B. **Real-World Noise Injection (Optional but Recommended)**

Enhance realism by injecting **real crypto volatility** and **network congestion**.

#### File: `data/enrich.py`
```python
import ccxt
import pandas as pd
from pathlib import Path

# Pull real BTC/USDT volatility
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=2160)  # 90 days
btc_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
btc_df = btc_df.set_index('timestamp')  # set_index returns a new DataFrame; prefer this over using inplace=True
btc_vol = btc_df['close'].pct_change().rolling(24).std()  # Volatility

# Inject into synthetic data
for path in Path("data/raw").glob("*.parquet"):
    df = pd.read_parquet(path)
    # Both timestamps now in milliseconds, ready for direct merge
    df = df[df['timestamp'].isin(btc_vol.index)]
    df = df.join(btc_vol.rename('btc_vol'), on='timestamp')
    df['volume'] *= (1 + 0.3 * df['btc_vol'].fillna(0.02))  # Modulate volume by volatility
    df.to_parquet(f"data/enriched/{path.stem}.parquet")
```

> Output: Enriched sequences with real market dynamics.

---

## 🧹 **Phase 2: Local Preprocessing Pipeline**

Convert raw sequences into **model-ready tensors**.

#### File: `data/preprocess.py`
```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import torch

# Load and scale
files = list(Path("data/enriched").glob("*.parquet"))
scaler = RobustScaler()

# Collect all data for fitting
all_data = []
for f in files:
    df = pd.read_parquet(f)
    all_data.append(df[['volume', 'fee_rate', 'entropy', 'btc_vol']].fillna(0.0))

X = np.vstack(all_data)
scaler.fit(X)

# Save scaler
import joblib
joblib.dump(scaler, "data/scaler.pkl")

# Process into sequences
sequences = []
labels = []
SEQ_LEN = 100

for f in files:
    df = pd.read_parquet(f)
    df = df.sort_values('timestamp')
    feats = df[['volume', 'instrument', 'success', 'fee_rate', 'entropy', 'geohint']].fillna(0.0).values

    # Scale continuous
    scaled = scaler.transform(feats[:, [0,3,4]])
    feats[:, 0] = scaled[:, 0]  # volume
    feats[:, 3] = scaled[:, 1]  # fee_rate
    feats[:, 4] = scaled[:, 2]  # entropy

    for i in range(len(feats) - SEQ_LEN - 5):
        seq = feats[i:i+SEQ_LEN]
        pred = feats[i+SEQ_LEN:i+SEQ_LEN+5, [0,2]]  # predict volume, success
        sequences.append(seq)
        labels.append(pred)

# Save as tensors
X = torch.tensor(np.array(sequences), dtype=torch.float32)
y = torch.tensor(np.array(labels), dtype=torch.float32)

torch.save(X, "data/tensors/X.pt")
torch.save(y, "data/tensors/y.pt")
```

> Output: `X.pt`, `y.pt`, `scaler.pkl` — ready for Colab.

---

## 📂 **Local Directory Structure**
```
dignity/
├── data/
│   ├── raw/                  # Synthetic merchant logs
│   ├── enriched/             # With real BTC volatility
│   ├── tensors/              # X.pt, y.pt
│   └── scaler.pkl            # Normalization model
├── local_gen.py              # Synthetic data generator
├── enrich.py                 # Real-world injection
└── preprocess.py             # Tensor conversion
```

---

## 🚀 **Phase 3: Upload & Train on Colab Pro**

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy data
!cp "/content/drive/MyDrive/dignity/data/tensors/X.pt" .
!cp "/content/drive/MyDrive/dignity/data/tensors/y.pt" .
!cp "/content/drive/MyDrive/dignity/data/scaler.pkl" .

# Load and train
X = torch.load('X.pt')[:10000]  # Sample
y = torch.load('y.pt')[:10000]

# Train Dignity (your model here)
```

---

## 🔐 **Privacy & OpSec Notes**
- **Never store real merchant IDs** — use random `mid`.
- **Hash or drop identifiers** (IP, UA) immediately.
- **Do not log content access** — only transaction metadata.
- **Delete raw data after tensor conversion** if not needed.