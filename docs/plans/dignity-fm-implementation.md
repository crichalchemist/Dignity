# Dignity Foundation Model Implementation Plan

## Overview

Implement a multi-asset foundation model for financial market analysis using B2S6-enhanced Mamba architecture with cross-asset attention.

**Target:** Working foundation model prototype trainable on local CPU + Colab GPU

## Context

- **Architecture:** CNN1D → B2S6 (×12 layers) → Transformer (×4 layers) → Task Heads
- **Data Sources:** Cryptocurrency (CCXT), Forex (Alpha Vantage), Synthetic
- **Privacy:** DP-SGD for transaction fine-tuning (ε=3.0)
- **Scale:** Small (50M params) for prototype, Medium (300M) for pre-training

## Tasks

### Task 1: B2S6 Core Unit Implementation

**File:** `.worktree/Dignity-Model/models/backbone/b2s6.py`

Implement the Block-Biased S6 (B2S6) recurrent unit based on the paper "Block-Biased Mamba for Long-Range Sequence Processing".

**Requirements:**
- Block partitioning: Split input channels into configurable blocks (default: 4)
- Channel-specific bias: Learnable per-channel bias for long-range memory
- Selective state space dynamics: Input-dependent A, B, C matrices
- Input-dependent sampling interval (delta): Controls forgetting rate
- Efficient computation: Use parallel scan for training, recurrent for inference

**Key Components:**
```python
class B2S6Unit(nn.Module):
    def __init__(
        self,
        d_model: int = 512,        # Channel dimension
        state_dim: int = 64,       # SSM state dimension (n)
        num_blocks: int = 4,       # Number of blocks for partitioning
        conv_kernel: int = 4,      # Convolution kernel size
        dt_rank: str = "auto"      # Rank for delta projection
    )
```

**Implementation Details:**
1. Block Partitioning Layer: Split d_model into num_blocks
2. Per-block selective SSM:
   - A matrix: Diagonal, input-dependent via delta
   - B matrix: Input projection, shared within block
   - C matrix: Output projection, shared within block
   - Delta: softplus(linear(input) + bias) for sampling interval
3. Channel-specific bias: Add learnable bias per channel (not input-dependent)
4. Combine blocks: Concatenate outputs from all blocks

**Math Reference:**
- x_k = A_k * x_{k-1} + B_k * u_k  (state update)
- y_k = C_k * x_k  (output)
- A_k = exp(delta_k * A_diag)  (discretization)
- delta_k = softplus(w^T * u_k + b)  (input-dependent)

**Tests:**
- Forward pass shape check: [batch, seq_len, d_model] → [batch, seq_len, d_model]
- Block partitioning correctness: num_blocks splits
- Gradient flow: Check backprop through selective dynamics
- Long sequence handling: Test with 10K+ timesteps

---

### Task 2: B2S6 Layer with Residual & Normalization

**File:** `.worktree/Dignity-Model/models/backbone/b2s6.py`

Wrap B2S6Unit in a full layer with residual connections, layer normalization, and optional convolution.

**Requirements:**
- Pre-normalization: LayerNorm before B2S6Unit (like modern transformers)
- Residual connection: x + B2S6(LayerNorm(x))
- Optional 1D convolution: Short-range modeling before SSM
- Dropout: Configurable dropout rate

**Key Components:**
```python
class B2S6Layer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        state_dim: int = 64,
        num_blocks: int = 4,
        use_conv: bool = True,
        conv_kernel: int = 4,
        dropout: float = 0.1
    )
```

**Implementation Details:**
1. LayerNorm: Pre-normalization before B2S6
2. Optional Conv1D: Causal convolution for local patterns
3. B2S6Unit: Core selective state space
4. Dropout: Applied after B2S6
5. Residual: Add input to output

**Tests:**
- Residual connection preserves gradients
- LayerNorm stabilizes training
- Conv1D captures local patterns
- Dropout mask is different per forward pass

---

### Task 3: B2S6 Backbone (Multi-Layer Stack)

**File:** `.worktree/Dignity-Model/models/backbone/b2s6.py`

Stack multiple B2S6 layers to create the temporal modeling backbone.

**Requirements:**
- Configurable depth: 6-24 layers
- Gradient checkpointing: For memory efficiency on long sequences
- Final layer normalization: Stabilize output before cross-attention

**Key Components:**
```python
class B2S6Backbone(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 12,
        state_dim: int = 64,
        num_blocks: int = 4,
        use_gradient_checkpointing: bool = False
    )
```

**Implementation Details:**
1. Input projection: Linear(input_features, d_model) if needed
2. Layer stack: nn.ModuleList of B2S6Layer
3. Gradient checkpointing: Use torch.utils.checkpoint for memory
4. Final LayerNorm: Normalize before output

**Tests:**
- Multi-layer forward pass
- Gradient checkpointing reduces memory
- Output shape matches d_model
- Deep network (24 layers) doesn't explode/vanish

---

### Task 4: Cross-Asset Transformer Layer

**File:** `.worktree/Dignity-Model/models/backbone/cross_asset_attention.py`

Implement cross-asset attention to learn correlations between crypto and forex markets.

**Requirements:**
- Multi-head attention: Learn different correlation patterns (8 heads)
- Residual connections: Stabilize deep network
- Feed-forward network: Non-linear transformation
- Layer normalization: Pre-norm architecture

**Key Components:**
```python
class CrossAssetTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    )
```

**Implementation Details:**
1. Multi-head attention: Query/Key/Value projections
2. Attention mechanism: Scaled dot-product
3. Residual + LayerNorm: x + Dropout(Attention(LayerNorm(x)))
4. Feed-forward: Linear → GELU → Linear
5. Second residual: x + Dropout(FFN(LayerNorm(x)))

**Tests:**
- Attention weights sum to 1
- Multi-head outputs are different
- Residual connections work
- Handles variable sequence lengths

---

### Task 5: CNN1D Feature Extractor

**File:** `.worktree/Dignity-Model/models/backbone/cnn_feature_extractor.py`

Create CNN1D layers for local pattern extraction (price action, volume spikes).

**Requirements:**
- Two convolutional layers: kernel sizes 3 and 5
- Layer normalization: Stabilize activations
- GELU activation: Smooth non-linearity
- Causal padding: No future information leakage

**Key Components:**
```python
class CNN1DFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_features: int = 13,  # OHLCV + indicators
        d_model: int = 512,
        kernel_sizes: list[int] = [3, 5]
    )
```

**Implementation Details:**
1. First conv: Conv1D(input_features, d_model, kernel=3)
2. LayerNorm + GELU
3. Second conv: Conv1D(d_model, d_model, kernel=5)
4. LayerNorm + GELU
5. Causal padding: Pad left only, no right padding

**Tests:**
- Output shape: [batch, seq_len, d_model]
- Causal property: Output at time t doesn't depend on t+1
- Feature extraction: Different kernels capture different patterns

---

### Task 6: Task-Specific Heads

**File:** `.worktree/Dignity-Model/models/head/foundation_heads.py`

Implement all four task heads: market analysis, movement prediction, risk scoring, trade signals.

**Requirements:**
- Market regime classifier: 5 classes (Bull/Bear/Ranging/High-Vol/Low-Vol)
- Multi-horizon forecaster: 4 horizons (1min, 5min, 1hr, 1day)
- Risk assessment: VaR (99%) + Expected Shortfall
- Trade signal: Entry/exit + confidence + position size

**Key Components:**
```python
class MarketRegimeHead(nn.Module): ...
class MultiHorizonForecastHead(nn.Module): ...
class RiskAssessmentHead(nn.Module): ...
class TradeSignalHead(nn.Module): ...
```

**Implementation Details:**
1. **Market Regime:** Pool sequence → Linear(d_model, 256) → GELU → Linear(256, 5)
2. **Forecaster:** Last timestep → 4 separate linear heads for each horizon
3. **Risk:** Separate heads for VaR and ES, both continuous outputs
4. **Trade Signal:** 3 outputs (entry class, confidence, position size)

**Tests:**
- Market regime: Softmax outputs sum to 1
- Forecaster: 4 separate predictions
- Risk: VaR >= ES (mathematical constraint)
- Trade signal: Confidence and position in [0, 1]

---

### Task 7: Complete Foundation Model

**File:** `.worktree/Dignity-Model/models/dignity_fm.py`

Integrate all components into the complete Dignity Foundation Model.

**Requirements:**
- Modular architecture: CNN → B2S6 → Transformer → Heads
- Configurable scale: Small (50M), Medium (300M), Large (1B)
- Multi-task outputs: All 4 heads in single forward pass
- Efficient: Support gradient checkpointing and mixed precision

**Key Components:**
```python
class DignityFoundationModel(nn.Module):
    def __init__(
        self,
        input_features: int = 13,
        d_model: int = 512,
        b2s6_layers: int = 12,
        transformer_layers: int = 4,
        num_heads: int = 8,
        task_heads: list[str] = ["regime", "forecast", "risk", "trade"]
    )
```

**Implementation Details:**
1. CNN feature extractor
2. B2S6 backbone (temporal modeling)
3. Cross-asset transformer stack
4. All task heads (selected via config)
5. Forward pass returns dict of task outputs

**Tests:**
- End-to-end forward pass
- Multi-task outputs correct shapes
- Parameter count matches target (50M/300M/1B)
- Supports long sequences (10K+ timesteps)

---

### Task 8: CCXT Historical Data Loader

**File:** `.worktree/Dignity-Model/data/source/crypto_historical.py`

Implement cryptocurrency data loader using CCXT library.

**Requirements:**
- Support multiple exchanges: Binance, Coinbase, Kraken
- Fetch historical OHLCV: 1-minute granularity
- Caching: Save to parquet files to avoid re-fetching
- Rate limiting: Respect exchange API limits
- Multiple symbols: Batch loading

**Key Components:**
```python
class CCXTHistoricalLoader:
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_date: str = "2020-01-01",
        end_date: str = "2026-01-01"
    ) -> pd.DataFrame
```

**Implementation Details:**
1. Initialize CCXT exchange
2. Convert dates to millisecond timestamps
3. Fetch in batches (1000 candles per request)
4. Handle rate limits with sleep
5. Cache to parquet
6. Return DataFrame with [timestamp, open, high, low, close, volume]

**Tests:**
- Fetches real data from Binance
- Caching works (second load is instant)
- Handles date ranges correctly
- Multiple symbols can be loaded
- Timestamps are in milliseconds

---

### Task 9: Forex Historical Data Loader

**File:** `.worktree/Dignity-Model/data/source/forex_historical.py`

Implement forex data loader using Alpha Vantage API.

**Requirements:**
- Support Alpha Vantage (free tier available)
- Fetch historical OHLC: 1-minute granularity
- Caching: Save to parquet
- Rate limiting: 5 requests/min for free tier
- Multiple pairs: Sequential loading with delays

**Key Components:**
```python
class ForexHistoricalLoader:
    def fetch_ohlc(
        self,
        pair: str,
        interval: str = "1min",
        start_date: str = "2020-01-01",
        end_date: str = "2026-01-01"
    ) -> pd.DataFrame
```

**Implementation Details:**
1. API key from environment variable
2. Request FX_INTRADAY data
3. Parse JSON response
4. Convert timestamps to milliseconds
5. Cache to parquet
6. Return DataFrame with same format as crypto

**Tests:**
- Fetches real forex data (with API key)
- Caching works
- Rate limiting respects 5 req/min
- Timestamps aligned with crypto (milliseconds)
- Volume column exists (set to 0 for forex)

---

### Task 10: Timestamp Alignment Pipeline

**File:** `.worktree/Dignity-Model/data/pipeline/alignment.py`

Align multi-source data (crypto + forex) to common timestamp grid.

**Requirements:**
- Reference grid: 1-minute intervals
- Fill missing values: Forward-fill by default
- Multi-source support: Handle different timestamp ranges
- Efficient: Use pandas reindex

**Key Components:**
```python
class TimestampAligner:
    def align(
        self,
        data_sources: dict[str, pd.DataFrame],
        method: str = "ffill"
    ) -> pd.DataFrame
```

**Implementation Details:**
1. Find global min/max timestamps across sources
2. Create reference grid (60000 ms intervals)
3. Reindex each source to grid
4. Apply fill method (ffill/bfill/interpolate)
5. Concatenate with multi-level columns
6. Return aligned DataFrame

**Tests:**
- Aligns crypto and forex to same grid
- Missing values filled correctly
- Multi-level columns: [source][feature]
- Timestamps are exact 1-minute intervals

---

### Task 11: Feature Engineering Pipeline

**File:** `.worktree/Dignity-Model/data/pipeline/features.py`

Compute technical indicators for all assets.

**Requirements:**
- Returns: Log returns
- Volatility: Rolling standard deviation (60-min window)
- RSI: Relative Strength Index (14-period)
- MACD: Moving Average Convergence Divergence
- Volume MA: Moving average of volume (20-period)

**Key Components:**
```python
class FeatureEngineer:
    def compute_all_features(
        self,
        df: pd.DataFrame,
        price_cols: list[str] = None,
        volume_cols: list[str] = None
    ) -> pd.DataFrame
```

**Implementation Details:**
1. Auto-detect price/volume columns
2. Compute returns: log(price_t / price_{t-1})
3. Compute volatility: rolling std of returns
4. Compute RSI: gain/loss ratio
5. Compute MACD: EMA difference + signal
6. Compute volume MA: rolling mean
7. Fill NaN with appropriate defaults

**Tests:**
- All indicators computed correctly
- No NaN values after processing
- Shapes match input
- Works with multi-level columns

---

### Task 12: Sequence Windowing Dataset

**File:** `.worktree/Dignity-Model/data/pipeline/windowing.py`

Create PyTorch dataset for sliding window sequences.

**Requirements:**
- Configurable window size: 10,000 timesteps
- Configurable stride: 1,000 timesteps (overlap)
- PyTorch Dataset: Compatible with DataLoader
- Return tensors: Float32 format

**Key Components:**
```python
class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10000,
        stride: int = 1000
    )
```

**Implementation Details:**
1. Store data array
2. Compute number of windows: (len - seq_len) // stride + 1
3. __getitem__: Slice window, convert to tensor
4. __len__: Return number of windows

**Tests:**
- Correct number of windows computed
- Window slicing works
- Tensors are correct shape
- Compatible with DataLoader (batching)

---

### Task 13: Complete Data Pipeline

**File:** `.worktree/Dignity-Model/data/pipeline/foundation_data.py`

Integrate all data components into end-to-end pipeline.

**Requirements:**
- Load from multiple sources
- Align timestamps
- Engineer features
- Create train/val split (80/20, chronological)
- Return PyTorch datasets

**Key Components:**
```python
class FoundationDataPipeline:
    def create_dataset(
        self,
        sequence_length: int = 10000,
        stride: int = 1000,
        train_split: float = 0.8
    ) -> tuple[SequenceWindowDataset, SequenceWindowDataset]
```

**Implementation Details:**
1. Initialize loaders (crypto, forex)
2. Load raw data
3. Align timestamps
4. Engineer features
5. Drop timestamp column
6. Train/val split (chronological)
7. Create window datasets
8. Return (train_dataset, val_dataset)

**Tests:**
- End-to-end pipeline runs
- Train/val split is chronological
- Datasets have correct shapes
- Can create DataLoader from datasets

---

### Task 14: Configuration System

**File:** `.worktree/Dignity-Model/config/foundation_model.yaml`

Create YAML configuration for foundation model training.

**Requirements:**
- Model architecture config: Dimensions, layers, heads
- Data config: Symbols, date ranges, sequence params
- Training config: Optimizer, learning rate, epochs
- Privacy config: DP-SGD parameters

**Structure:**
```yaml
model:
  scale: medium  # small, medium, large
  d_model: 512
  b2s6_layers: 12
  transformer_layers: 4
  
data:
  crypto_symbols: [BTC/USDT, ETH/USDT]
  forex_pairs: [EUR/USD]
  sequence_length: 10000
  
training:
  batch_size: 16
  learning_rate: 1e-4
  epochs: 100
  
privacy:
  enabled: false
  epsilon: 3.0
```

**Tests:**
- Config loads correctly
- All parameters accessible
- Validation: Required fields present

---

### Task 15: Pre-training Script (Public Data, No DP)

**File:** `.worktree/Dignity-Model/train/pretrain.py`

Create training script for foundation model pre-training on public market data.

**Requirements:**
- Multi-task loss: MSM (40%) + Contrastive (30%) + Forecast (30%)
- AdamW optimizer with warmup
- Mixed precision (FP16) for speed
- Checkpoint saving every N steps
- Validation loop
- TensorBoard logging

**Key Components:**
```python
def pretrain(
    model: DignityFoundationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict
):
    # Training loop
```

**Implementation Details:**
1. Load config
2. Initialize model
3. Setup optimizer (AdamW) + scheduler (cosine with warmup)
4. Setup mixed precision (torch.cuda.amp)
5. Training loop:
   - Masked sequence modeling loss
   - Contrastive learning loss
   - Multi-horizon forecast loss
   - Combined loss with weights
6. Validation loop
7. Checkpoint saving
8. Logging to TensorBoard

**Tests:**
- Training loop runs
- Loss decreases over epochs
- Validation loss computed
- Checkpoints saved correctly
- Logs visible in TensorBoard

---

## Success Criteria

**Prototype Complete When:**
1. All 15 tasks implemented and tested
2. End-to-end training runs on small dataset
3. Model checkpoints can be saved/loaded
4. Validation metrics are reasonable
5. Data pipeline handles real CCXT data

**Quality Standards:**
- All functions have docstrings
- Type hints on all function signatures
- Unit tests pass for each component
- Code follows PEP 8 (ruff/black compliant)
- No hard-coded paths (use config)

**Performance Targets (Prototype):**
- Small model (50M params) trains on CPU
- Data loading: <1 min for 1 week of crypto data
- Forward pass: <100ms for 10K sequence on CPU
- Memory: <8GB RAM for training

## Next Steps After Prototype

1. Scale to medium model (300M params)
2. Train on Colab GPU with full dataset
3. Implement DP-SGD fine-tuning
4. Add evaluation scripts
5. ONNX export for deployment
6. Backtest trading signals
