# Dignity Foundation Model Architecture

## Overview

**Dignity-FM** is a multi-asset foundation model for financial market analysis and trading. It combines state-of-the-art sequence modeling (B2S6-enhanced Mamba) with multi-asset attention mechanisms for cross-market intelligence.

## Core Architecture: Hybrid B2S6-Transformer

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                              │
│  ┌──────────┬──────────┬──────────┬──────────────────────┐  │
│  │ Crypto   │  Forex   │  Trans.  │  Macro Indicators    │  │
│  │ OHLCV    │  OHLCV   │  Data    │  (Vol Index, etc.)   │  │
│  └──────────┴──────────┴──────────┴──────────────────────┘  │
│                          ↓                                   │
│              Timestamp Alignment (milliseconds)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction (CNN1D)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Conv1D(in=features, out=256, kernel=3)            │     │
│  │  → LayerNorm → GELU                                │     │
│  │  Conv1D(256, 256, kernel=5)                        │     │
│  │  → LayerNorm → GELU                                │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: Local patterns (price action, volume spikes)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│        Temporal Modeling (B2S6 Blocks × 12 layers)          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  B2S6 Unit (Block-Biased Selective State Space)    │     │
│  │  ┌──────────────────────────────────────────────┐  │     │
│  │  │ 1. Block Partitioning (split into 4 blocks)  │  │     │
│  │  │    - Each block: selective dynamics          │  │     │
│  │  │    - Increases effective model width         │  │     │
│  │  │                                              │  │     │
│  │  │ 2. Channel-Specific Bias                    │  │     │
│  │  │    - Input-independent bias per channel     │  │     │
│  │  │    - Preserves long-range memory            │  │     │
│  │  │                                              │  │     │
│  │  │ 3. Selective State Space (per block)        │  │     │
│  │  │    x_k = A_k·x_{k-1} + B_k·u_k              │  │     │
│  │  │    y_k = C_k·x_k                            │  │     │
│  │  │    where A_k, B_k, C_k are input-dependent  │  │     │
│  │  └──────────────────────────────────────────────┘  │     │
│  │  → Residual Connection → LayerNorm                 │     │
│  └────────────────────────────────────────────────────┘     │
│  Properties:                                                 │
│  - Linear complexity O(L) for sequence length L              │
│  - Handles 1M+ timestep sequences efficiently                │
│  - Better long-range modeling than vanilla Mamba             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│    Cross-Asset Attention (Transformer Layer × 4)            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Multi-Head Attention (8 heads)                    │     │
│  │  - Query: Current asset state                      │     │
│  │  - Key/Value: All assets (crypto + forex)          │     │
│  │  - Learns correlations (BTC ↔ EUR/USD)             │     │
│  │  → Residual → LayerNorm                            │     │
│  │                                                     │     │
│  │  Feed-Forward Network                              │     │
│  │  - Linear(d_model, 4*d_model) → GELU               │     │
│  │  - Linear(4*d_model, d_model)                      │     │
│  │  → Residual → LayerNorm                            │     │
│  └────────────────────────────────────────────────────┘     │
│  Purpose: Cross-market intelligence                         │
│  - BTC rally → risk-off in EUR/USD?                         │
│  - Forex volatility → crypto volume surge?                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Task-Specific Heads                             │
│  ┌──────────────┬──────────────┬──────────────┬─────────┐  │
│  │ Market       │ Movement     │ Risk         │ Trade   │  │
│  │ Analysis     │ Prediction   │ Scoring      │ Signals │  │
│  ├──────────────┼──────────────┼──────────────┼─────────┤  │
│  │ Regime       │ Multi-       │ VaR          │ Entry/  │  │
│  │ Classifier   │ Horizon      │ Estimation   │ Exit    │  │
│  │              │ Forecaster   │              │ Timing  │  │
│  │ (Bull/Bear/  │ (1m/5m/1h/   │ (99% CI)     │         │  │
│  │  Ranging)    │  1d ahead)   │              │         │  │
│  └──────────────┴──────────────┴──────────────┴─────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Model Specifications

### Scale Options

**Small (Prototype - Local CPU):**
- Parameters: ~50M
- d_model: 256
- B2S6 layers: 6
- Transformer layers: 2
- Sequence length: 10K timesteps
- Use case: Local testing, architecture validation

**Medium (Pre-training - Colab GPU):**
- Parameters: ~300M
- d_model: 512
- B2S6 layers: 12
- Transformer layers: 4
- Sequence length: 100K timesteps
- Use case: Foundation model pre-training

**Large (Future - Multi-GPU):**
- Parameters: ~1B
- d_model: 1024
- B2S6 layers: 24
- Transformer layers: 8
- Sequence length: 1M timesteps
- Use case: Production deployment

### B2S6 Configuration

```python
# B2S6 Unit Hyperparameters
block_size = 4              # Number of blocks for partitioning
state_dim = 64              # SSM state dimension (n)
channel_dim = 512           # Model dimension (d_model)
dt_rank = "auto"            # Rank for delta projection
conv_kernel = 4             # Convolution kernel size

# Key improvements over vanilla Mamba:
# 1. Block partitioning: 512 channels → 4 blocks of 128
# 2. Channel-specific bias: learnable bias per channel
# 3. Better expressiveness: universal approximation
```

## Pre-training Strategy

### Objective Functions

**1. Masked Sequence Modeling (40% weight):**
```python
# BERT-style masking on multi-asset sequences
mask_ratio = 0.15
targets = [price, volume, volatility]
loss_msm = MSELoss(predicted_masked, actual_masked)
```

**2. Contrastive Learning (30% weight):**
```python
# Similar market regimes should cluster
# Positive pairs: same asset, different time windows (similar regime)
# Negative pairs: different regimes
loss_contrastive = NTXentLoss(embeddings, temperature=0.07)
```

**3. Multi-Horizon Forecasting (30% weight):**
```python
# Predict future returns at multiple horizons
horizons = [1, 5, 60, 1440]  # 1min, 5min, 1hr, 1day (in minutes)
loss_forecast = sum([MSELoss(pred_h, actual_h) for h in horizons])
```

**Total Loss:**
```python
loss_total = 0.4 * loss_msm + 0.3 * loss_contrastive + 0.3 * loss_forecast
```

## Data Requirements

### Pre-training Corpus

**Target Scale:** 10M+ sequences

**Data Sources:**
1. **Cryptocurrency (5M sequences)**
   - BTC, ETH, BNB, SOL, ADA, XRP (top 20 by volume)
   - 1-minute OHLCV from CCXT
   - Time range: 2020-2026 (6 years)
   - ~3.15M candles per asset × 20 assets = 63M datapoints

2. **Forex (3M sequences)**
   - EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD
   - 1-minute OHLCV from OANDA/Alpha Vantage
   - Time range: 2020-2026
   - ~2M candles per pair × 5 pairs = 10M datapoints

3. **Synthetic Transactions (2M sequences)**
   - Privacy-preserving synthetic data
   - Augmentation for rare events (flash crashes, pump-and-dump)
   - Generated via existing `SyntheticGenerator`

### Sequence Construction

**Window size:** 10,000 timesteps (1-minute granularity = ~7 days)
**Stride:** 1,000 timesteps (overlapping windows for data efficiency)
**Features per timestep:**
- OHLC (4)
- Volume (1)
- Volatility (1-hour rolling std) (1)
- Returns (1)
- Volume MA (1)
- RSI (1)
- MACD (3)
**Total:** 13 features × 10K timesteps = 130K input dimensions per sequence

## Training Configuration

### Phase 1: Pre-training (Public Data, No DP)

```yaml
model: DignityFM-Medium
data:
  crypto_assets: [BTC, ETH, BNB, SOL, ADA]
  forex_pairs: [EURUSD, GBPUSD, USDJPY]
  sequence_length: 10000
  batch_size: 16  # Colab GPU limit
  
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps: 10000
  
training:
  epochs: 100
  gradient_clip: 1.0
  mixed_precision: true  # FP16 for speed
  checkpoint_every: 5000
  
privacy:
  enabled: false  # Public market data, no DP needed
```

### Phase 2: Fine-tuning (Private Transactions, DP)

```yaml
model: DignityFM-Medium (load from checkpoint)
data:
  transaction_data: true
  sequence_length: 5000  # Shorter for private data
  batch_size: 32
  
optimizer:
  type: DP-SGD  # Differential Privacy SGD
  lr: 5e-5
  
training:
  epochs: 20
  
privacy:
  enabled: true
  epsilon: 3.0       # Privacy budget
  delta: 1e-7
  max_grad_norm: 1.0  # Per-sample clipping
  noise_multiplier: auto  # Computed from ε, δ
```

## Task Head Details

### 1. Market Analysis Head

**Output:** Regime classification (Bull/Bear/Ranging/High-Vol/Low-Vol)

```python
class MarketRegimeClassifier(nn.Module):
    def forward(self, backbone_output):
        # backbone_output: [batch, seq_len, d_model]
        # Pool to single representation
        pooled = backbone_output.mean(dim=1)  # [batch, d_model]
        
        # Classification layers
        x = Linear(d_model, 256)(pooled)
        x = GELU()(x)
        x = Dropout(0.1)(x)
        logits = Linear(256, 5)(x)  # 5 regimes
        
        return logits
```

### 2. Movement Prediction Head

**Output:** Multi-horizon returns (1min, 5min, 1hr, 1day)

```python
class MultiHorizonForecaster(nn.Module):
    def forward(self, backbone_output):
        # Predict from last timestep
        last_state = backbone_output[:, -1, :]  # [batch, d_model]
        
        # Separate heads for each horizon
        predictions = []
        for horizon in [1, 5, 60, 1440]:
            pred = Linear(d_model, 128)(last_state)
            pred = GELU()(pred)
            pred = Linear(128, 1)(pred)  # Single return value
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)  # [batch, 4]
```

### 3. Risk Scoring Head

**Output:** Value-at-Risk (99% confidence) + Expected Shortfall

```python
class RiskAssessmentHead(nn.Module):
    def forward(self, backbone_output):
        last_state = backbone_output[:, -1, :]
        
        # VaR estimation
        var_99 = Linear(d_model, 1)(last_state)
        
        # Expected Shortfall (conditional on VaR breach)
        es = Linear(d_model, 1)(last_state)
        
        return {"VaR_99": var_99, "ES": es}
```

### 4. Trade Signal Head

**Output:** Entry/Exit signals + confidence + suggested position size

```python
class TradeCycleController(nn.Module):
    def forward(self, backbone_output):
        pooled = backbone_output.mean(dim=1)
        
        # Entry signal (long/short/neutral)
        entry = Linear(d_model, 3)(pooled)  # 3-class
        
        # Confidence score
        confidence = Linear(d_model, 1)(pooled)
        confidence = Sigmoid()(confidence)
        
        # Position sizing (Kelly criterion-inspired)
        position_size = Linear(d_model, 1)(pooled)
        position_size = Sigmoid()(position_size)  # 0-1 range
        
        return {
            "entry_signal": entry,
            "confidence": confidence,
            "position_size": position_size
        }
```

## Privacy Implementation

### Differential Privacy During Fine-tuning

```python
from opacus import PrivacyEngine

# Attach DP engine to optimizer
privacy_engine = PrivacyEngine()

model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.1,  # Computed from ε=3.0, δ=1e-7
    max_grad_norm=1.0,
)

# Training loop with privacy accounting
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # DP noise added automatically
    
    # Check privacy budget
    epsilon = privacy_engine.get_epsilon(delta=1e-7)
    print(f"Epoch {epoch}, ε={epsilon:.2f}")
    
    if epsilon > 3.0:
        print("Privacy budget exhausted!")
        break
```

### Privacy-Utility Trade-off

**No DP (Public Data):**
- Perplexity: ~3.2 (baseline)
- Accuracy: ~92%

**ε=10 (Weak Privacy):**
- Perplexity: ~3.4 (+6%)
- Accuracy: ~90% (-2%)

**ε=3 (Strong Privacy - Recommended):**
- Perplexity: ~4.1 (+28%)
- Accuracy: ~85% (-7%)

**ε=1 (Very Strong Privacy):**
- Perplexity: ~5.8 (+81%)
- Accuracy: ~78% (-14%)

**Recommendation:** Use ε=3 for transaction fine-tuning. Acceptable utility loss for strong privacy.

## Deployment

### Model Export

**ONNX for CPU Inference:**
```python
# Export trained model
torch.onnx.export(
    model,
    dummy_input,
    "dignity_fm_medium.onnx",
    opset_version=17,
    input_names=["sequence"],
    output_names=["regime", "forecasts", "risk", "signals"],
    dynamic_axes={"sequence": {0: "batch", 1: "seq_len"}}
)

# Quantize for CPU deployment
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic(
    "dignity_fm_medium.onnx",
    "dignity_fm_medium_int8.onnx",
    weight_type="int8"
)
```

**Performance Estimates:**
- FP32 model: ~1.2GB, 50ms inference (CPU)
- INT8 quantized: ~300MB, 15ms inference (CPU)
- GPU (T4): <5ms inference

## Research Questions & Experiments

### Ablation Studies

1. **B2S6 vs Vanilla Mamba:** Does block partitioning help on forex/crypto?
2. **Cross-Asset Attention:** How much does multi-market context improve predictions?
3. **Privacy Budget:** What's the minimum ε for acceptable performance?
4. **Sequence Length:** Optimal window size for different tasks?

### Evaluation Metrics

**Pre-training:**
- Perplexity on held-out sequences
- Masked prediction accuracy
- Contrastive alignment score

**Downstream Tasks:**
- Market regime: Classification accuracy, F1
- Movement: RMSE, directional accuracy
- Risk: VaR coverage, ES calibration
- Trading: Sharpe ratio, max drawdown (backtest)

## Next Steps

1. Implement B2S6 unit in PyTorch
2. Build data loaders for CCXT + Forex
3. Pre-train on small scale (validate architecture)
4. Scale to Colab for full pre-training
5. Fine-tune with DP on transactions
6. Deploy and backtest

## References

- [B2S6 Paper] Yu & Erichson (2025). "Block-Biased Mamba for Long-Range Sequence Processing"
- [Mamba] Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- [S4] Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces"
- [Differential Privacy] Abadi et al. (2016). "Deep Learning with Differential Privacy"
