# Forex Time-Series Foundation Model: Training Curriculum

A structured 5-phase curriculum for building a pre-trained time-series foundation model, progressing from general time series understanding to production-ready forex trading.

---

## Curriculum Overview

```
Phase 1: Foundation        → Phase 2: Domain          → Phase 3: Cross-Asset
(Generic Time Series)        (Forex Specialist)         (Multi-Domain Transfer)
     ↓                            ↓                            ↓
Phase 4: Regime & Context  → Phase 5: Production Fine-Tuning
(Long Context + Macro)        (RL + Live Validation)
```

| Phase | Duration | Focus | Key Metric |
|-------|----------|-------|------------|
| 1 | 2-4 weeks | Universal TS patterns | MASE < 1.0 |
| 2 | 3-6 weeks | Forex specialization | Direction Acc > 52% |
| 3 | 4-6 weeks | Cross-asset transfer | Retain > 95% forex performance |
| 4 | 3-4 weeks | Regime + long context | Regime F1 > 0.7 |
| 5 | 2-3 weeks | Production fine-tuning | Sharpe > 1.0 |

---

## Phase 1: Foundation (General Time Series Pre-training)

**Objective:** Learn universal time series representations before domain specialization—trends, seasonality, noise patterns, missing value handling.

### Configuration

| Aspect | Specification |
|--------|---------------|
| **Duration** | 2-4 weeks |
| **Datasets** | Benchmark datasets only (see below) |
| **Tasks** | Masked reconstruction, next-step prediction, denoising |
| **Context Length** | 96 → 192 → 336 tokens (progressive) |
| **Supervision** | Dense (every timestep) |
| **Learning Rate** | High initial (1e-3), 5% warmup, cosine decay |
| **Batch Size** | Maximize GPU utilization |

### Datasets for Phase 1

| Dataset | Source | Size | Resolution | Purpose |
|---------|--------|------|------------|---------|
| **Monash Time Series Archive** | [forecastingdata.org](https://forecastingdata.org/) | 30 datasets, ~5GB | Various | Multi-domain benchmark |
| **M4 Competition** | [github.com/Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods) | 100K series, ~500MB | Hourly-Yearly | Forecasting benchmarks |
| **M5 Competition (Walmart)** | [kaggle.com/c/m5-forecasting-accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) | ~2GB | Daily | Hierarchical forecasting |
| **ETTh/ETTm (Electricity)** | [github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) | ~50MB | 15min/1h | Standard TSF benchmark |
| **Weather (Autoformer)** | [github.com/thuml/Autoformer](https://github.com/thuml/Autoformer) | ~500MB | 10min-1h | Physical process modeling |
| **Traffic (PEMS-BAY)** | [zenodo.org/record/5724362](https://zenodo.org/record/5724362) | ~500MB | 5min | Spatial-temporal patterns |
| **UCR Archive** | [cs.ucr.edu/~eamonn/time_series_data_2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | 128 datasets, ~1GB | Various | Classification benchmarks |

**Why these datasets:** Diverse domains (retail, energy, weather, traffic) teach the encoder to recognize universal patterns without financial market bias. This prevents overfitting to forex-specific quirks.

### Sub-phase Schedule

| Week | Context Length | Focus | Validation |
|------|----------------|-------|------------|
| 1 | 96 tokens | Basic pattern recognition | Reconstruction loss < 0.1 |
| 2 | 192 tokens | Multi-scale dependencies | Attention shows periodicity |
| 3-4 | 336 tokens | Long-range patterns | Transfer to held-out domains |

### Validation Checkpoints

```python
phase_1_criteria = {
    "monash_mase": 1.0,           # Mean Absolute Scaled Error < 1.0 (beats naive)
    "m4_smape": 0.9,              # < 90% of naive baseline
    "reconstruction_loss": 0.05,  # Masked reconstruction convergence
    "transfer_score": 0.10,       # Zero-shot on held-out domains > 10% improvement
}
```

### Advancement Criteria

```python
advance_to_phase_2 = (
    monash_mase < 1.0 and
    m4_smape < naive_smape * 0.9 and
    training_loss_plateau(patience=5_epochs)
)
```

---

## Phase 2: Domain Specialization (Forex Focus)

**Objective:** Deep expertise in forex market dynamics—trading sessions, news reactions, carry trade patterns, cross-pair correlations.

### Configuration

| Aspect | Specification |
|--------|---------------|
| **Duration** | 3-6 weeks |
| **Datasets** | Existing 46 pairs + supplemental forex data |
| **Tasks** | Multi-horizon forecasting, volatility prediction, regime detection |
| **Context Length** | 336 → 512 → 720 tokens |
| **Supervision** | Dense → Semi-dense (key levels) |
| **Auxiliary Tasks** | Session classification, spread prediction, direction |
| **News Integration** | GDELT embeddings as conditional input |

### Datasets for Phase 2

**Primary (Already Available):**
| Dataset | Pairs | Time Range | Resolution | Size |
|---------|-------|------------|------------|------|
| **HistData (existing)** | 36 FX | 2010-2025 | 1min | ~280M samples |
| **Crypto (existing)** | 10 pairs | 2019-2025 | 1min | ~31.5M samples |
| **GDELT (existing)** | - | 2015-present | 15min | Sentiment features |

**Supplemental (Add for Validation/Depth):**
| Dataset | Source | Type | Purpose | Priority |
|---------|--------|------|---------|----------|
| **Dukascopy Tick Data** | [dukascopy.com](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | Tick-level | Microstructure learning, HistData validation | HIGH |
| **TrueFX** | [truefx.com](https://www.truefx.com/) | Bid/ask tick | Spread dynamics, liquidity patterns | MEDIUM |
| **FXCM Historical** | [fxcm.com/markets](https://www.fxcm.com/markets/) | 1min OHLCV | Cross-validation with HistData | MEDIUM |
| **ForexFactory Calendar** | [forexfactory.com/calendar](https://www.forexfactory.com/calendar) | Events | High-impact event timestamps | HIGH |

### Sub-phase Schedule

**Phase 2a: Major Pairs Foundation (40% of phase)**
```
Duration: 1-1.5 weeks
Pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
Resolution: 1H → 15min → 5min → 1min (progressive)
Objective: Learn core G7 currency dynamics
Validation: Direction accuracy > 51% on 1H EUR/USD
```

**Phase 2b: Cross Pairs Expansion (30% of phase)**
```
Duration: 1-2 weeks
Pairs: EUR/GBP, EUR/JPY, GBP/JPY, EUR/AUD, AUD/JPY, etc. (22 crosses)
Task: Cross-pair correlation learning
Objective: Triangular arbitrage relationships
Validation: Correlation prediction R² > 0.5
```

**Phase 2c: Full Pair Coverage (20% of phase)**
```
Duration: 1 week
Pairs: All 46 pairs including exotics (USD/TRY, USD/ZAR, etc.)
Task: Low-liquidity pattern recognition
Objective: Thin market dynamics, wider spreads
Validation: No performance degradation on majors
```

**Phase 2d: GDELT News Conditioning (10% of phase)**
```
Duration: 0.5-1 week
Task: Condition forecasts on sentiment embeddings
Objective: Learn news → price causality
Validation: NFP/FOMC event reactions captured
```

### Validation Checkpoints

```python
phase_2_criteria = {
    "direction_accuracy_eurusd": 0.52,    # Statistically significant > 50%
    "direction_accuracy_all": 0.51,        # All pairs average
    "mape_1h": 0.002,                      # 0.2% MAPE on 1H forecast
    "regime_f1": 0.60,                     # Regime classification F1
    "cross_pair_correlation_r2": 0.50,    # Predict EUR/GBP from EUR/USD + GBP/USD
    "session_classification_acc": 0.85,   # London/NY/Tokyo/Sydney
}
```

### Advancement Criteria

```python
advance_to_phase_3 = (
    direction_accuracy_eurusd >= 0.52 and
    regime_classification_f1 >= 0.60 and
    no_catastrophic_forgetting_on_benchmarks
)
```

---

## Phase 3: Cross-Asset Transfer

**Objective:** Transfer forex knowledge to related asset classes—cryptocurrencies, commodities, equity indices—while retaining forex performance.

### Configuration

| Aspect | Specification |
|--------|---------------|
| **Duration** | 4-6 weeks |
| **Datasets** | Forex (existing) + Crypto + Commodities + Equities |
| **Tasks** | Unified forecasting, cross-asset correlation, lead-lag detection |
| **Context Length** | 720 → 1024 → 2048 tokens |
| **Supervision** | Semi-dense → Sparse (key events only) |
| **Architecture** | Shared encoder, asset-specific output heads |

### Datasets for Phase 3

**Cryptocurrency (40% of phase):**
| Dataset | Source | Coverage | Resolution | License |
|---------|--------|----------|------------|---------|
| **Binance Public Data** | [data.binance.vision](https://data.binance.vision/) | 500+ pairs | 1s-1D | Free |
| **CryptoDataDownload** | [cryptodatadownload.com](https://www.cryptodatadownload.com/) | Multi-exchange | 1min-1D | Free |
| **Glassnode** | [glassnode.com](https://glassnode.com/) | On-chain metrics | Various | Free tier |

**Commodities (30% of phase):**
| Dataset | Source | Coverage | Resolution | License |
|---------|--------|----------|------------|---------|
| **Nasdaq Data Link (Quandl)** | [data.nasdaq.com](https://data.nasdaq.com/) | Continuous futures | 1D | Free tier |
| **FRED Commodities** | [fred.stlouisfed.org](https://fred.stlouisfed.org/) | Gold, Oil, Ag | 1D | Free |
| **Investing.com** | [investing.com/commodities](https://www.investing.com/commodities/) | 40+ commodities | 1D | Free |

**Equity Indices (30% of phase):**
| Dataset | Source | Coverage | Resolution | License |
|---------|--------|----------|------------|---------|
| **Yahoo Finance (yfinance)** | [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) | All US stocks | 1min-1D | Free |
| **Kaggle US Stocks** | [kaggle.com/borismarjanovic](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) | 8000+ tickers | 1D | CC0 |
| **CBOE VIX** | [cboe.com/vix](https://www.cboe.com/tradable_products/vix/) | VIX, VVIX | 1min-1D | Free |

### Sub-phase Schedule

**Phase 3a: Cryptocurrency Integration (40%)**
```
Duration: 2-2.5 weeks
Assets: BTC/USD, ETH/USD + top 20 alts
Rationale: 24/7 markets, similar tick structure to forex
Special Focus: Weekend/holiday patterns (forex gaps), high volatility regimes
Validation: Crypto direction accuracy > 51%
```

**Phase 3b: Commodities (30%)**
```
Duration: 1.5-2 weeks
Assets: Gold (XAU/USD), Silver, Crude Oil, Natural Gas, Copper
Rationale: USD correlation (gold inverse), macro sensitivity
Special Focus: Safe-haven flows, inflation hedging, OPEC events
Validation: Gold-USD inverse correlation captured (R² > 0.3)
```

**Phase 3c: Equity Indices (30%)**
```
Duration: 1.5-2 weeks
Assets: SPY, QQQ, DXY (Dollar Index), VIX
Rationale: Risk sentiment driver for forex
Special Focus: Risk-on/risk-off regime detection, earnings calendar
Validation: VIX regime signal improves forex predictions
```

### Validation Checkpoints

```python
phase_3_criteria = {
    "forex_retention": 0.95,              # < 5% performance drop vs Phase 2
    "crypto_direction_accuracy": 0.51,    # Competitive with forex
    "gold_usd_correlation_r2": 0.30,      # Capture inverse relationship
    "risk_on_off_f1": 0.65,               # Regime classification
    "vix_signal_improvement": 0.02,       # VIX improves forex by 2%
}
```

### Advancement Criteria

```python
advance_to_phase_4 = (
    forex_performance_retained > 0.95 and
    crypto_direction_accuracy > 0.51 and
    cross_asset_correlation_r2 > 0.30
)
```

---

## Phase 4: Regime Awareness & Long Context

**Objective:** Handle regime changes and extended context windows, incorporating macroeconomic data for multi-week forecasting.

### Configuration

| Aspect | Specification |
|--------|---------------|
| **Duration** | 3-4 weeks |
| **Datasets** | All previous + Macro + Alternative data |
| **Tasks** | Regime classification, long-horizon (1D-1W), conditional generation |
| **Context Length** | 2048 → 4096 → 8192 tokens |
| **Supervision** | Sparse (regime labels, weekly targets) |
| **Architecture** | Efficient attention (FlashAttention), potential state space hybrid |

### Datasets for Phase 4

**Macroeconomic Data:**
| Dataset | Source | Coverage | Resolution | License |
|---------|--------|----------|------------|---------|
| **FRED Economic Data** | [fred.stlouisfed.org](https://fred.stlouisfed.org/) | 800K+ series | Various | Free |
| **World Bank Open Data** | [data.worldbank.org](https://data.worldbank.org/) | 200+ countries | Annual | Free |
| **BIS Statistics** | [bis.org/statistics](https://www.bis.org/statistics/) | FX turnover, rates | Quarterly | Free |
| **ECB Statistical Data** | [sdw.ecb.europa.eu](https://sdw.ecb.europa.eu/) | Euro area macro | Various | Free |
| **Treasury Yield Curves** | [treasury.gov](https://home.treasury.gov/resource-center/data-chart-center) | US yields | Daily | Free |

**Alternative Data:**
| Dataset | Source | Type | License |
|---------|--------|------|---------|
| **COT Reports (CFTC)** | [cftc.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders/) | Positioning | Free |
| **Economic Calendar** | [investing.com/economic-calendar](https://www.investing.com/economic-calendar/) | Events | Free |
| **VIX Term Structure** | [cboe.com](https://www.cboe.com/) | Volatility curve | Free |
| **Put/Call Ratios** | [cboe.com/us/options/market_statistics](https://www.cboe.com/us/options/market_statistics/) | Sentiment | Free |

### Regime Categories

| Regime | Characteristics | Detection Signals |
|--------|-----------------|-------------------|
| **1. Low Vol Trending** | Carry trades, momentum | VIX < 15, trend strength > 0.7 |
| **2. High Vol Trending** | Risk-off/on flows | VIX > 20, directional |
| **3. Low Vol Ranging** | Consolidation, summer doldrums | VIX < 15, ADX < 20 |
| **4. High Vol Ranging** | Uncertainty, indecision | VIX > 20, ADX < 20 |
| **5. Crisis/Dislocation** | Flash crashes, interventions | VIX > 30, correlation breakdown |

### Sub-phase Schedule

**Phase 4a: Macro Integration (40%)**
```
Duration: 1.5 weeks
Data: FRED core series (GDP, unemployment, CPI, rates)
Task: Macro-conditioned forecasting
Validation: Weekly forecasts improve with macro features
```

**Phase 4b: Long Context Training (35%)**
```
Duration: 1-1.5 weeks
Context: 2048 → 4096 → 8192 tokens
Task: Utilize full history for regime detection
Validation: Attention entropy shows long-range utilization
Memory: FlashAttention-2, gradient checkpointing
```

**Phase 4c: Regime Classification (25%)**
```
Duration: 1 week
Task: 5-class regime classification
Features: VIX + realized vol + COT + macro
Validation: Regime F1 > 0.7
```

### Validation Checkpoints

```python
phase_4_criteria = {
    "regime_f1": 0.70,                    # 5-class regime classification
    "long_context_utilization": 0.30,     # Attention entropy reduction
    "macro_improvement_weekly": 0.03,     # 3% improvement with macro
    "memory_efficiency": 0.80,            # < 80% target GPU VRAM
    "crisis_detection_recall": 0.80,      # Catch 80% of crisis regimes
}
```

### Advancement Criteria

```python
advance_to_phase_5 = (
    regime_f1 > 0.70 and
    long_context_improves_weekly_forecast and
    memory_usage < target_gpu_vram * 0.8
)
```

---

## Phase 5: Production Fine-Tuning

**Objective:** Production-ready model optimized for specific trading strategies with RL-based reward shaping.

### Configuration

| Aspect | Specification |
|--------|---------------|
| **Duration** | 2-3 weeks |
| **Datasets** | Most recent 6 months (all assets) + live paper trading |
| **Tasks** | Target-specific (e.g., 4H EUR/USD directional) |
| **Context Length** | Optimal from Phase 4 (typically 1024-4096) |
| **Supervision** | Trading rewards (Sharpe, win rate, max drawdown) |
| **Training** | RL fine-tuning (PPO/A3C on trading environment) |

### RL Reward Engineering

```python
def compute_reward(action, outcome, position):
    pnl_reward = outcome.pnl * 100  # Scale returns
    
    # Risk penalties
    drawdown_penalty = -abs(outcome.max_drawdown) * 50 if outcome.max_drawdown > 0.05 else 0
    overtrading_penalty = -0.01 if action != 'hold' else 0  # Discourage churn
    
    # Sharpe bonus
    sharpe_bonus = max(0, outcome.rolling_sharpe - 1.0) * 10
    
    return pnl_reward + drawdown_penalty + overtrading_penalty + sharpe_bonus
```

### Validation Checkpoints

```python
phase_5_criteria = {
    "paper_trading_sharpe": 1.0,          # 30-day rolling Sharpe > 1.0
    "max_drawdown": 0.10,                 # < 10% max drawdown
    "win_rate": 0.52,                     # Win rate > 52%
    "profit_factor": 1.5,                 # Gross profit / gross loss > 1.5
    "inference_latency_ms": 100,          # < 100ms for inference
    "consistency": 0.80,                  # 80% of weeks profitable
}
```

### Production Readiness Checklist

- [ ] Sharpe > 1.0 on 30-day paper trading
- [ ] Max drawdown < 10% over evaluation period
- [ ] Consistent performance across recent regime changes
- [ ] Inference latency < 100ms
- [ ] Model exported to ONNX for deployment
- [ ] Risk manager integration tested
- [ ] Graceful degradation on missing data

---

## Compute-Aware Configurations

### Consumer GPU (1x RTX 4090, 24GB)

| Phase | Batch Size | Context | Est. Duration |
|-------|------------|---------|---------------|
| 1 | 16 | 336 | 1 week |
| 2 | 12 | 512 | 2 weeks |
| 3 | 8 | 1024 | 3 weeks |
| 4 | 4 | 2048 | 2 weeks |
| 5 | 8 | 1024 | 1 week |

**Total: 9-12 weeks**

**Optimizations Required:**
- BF16 mixed precision (mandatory)
- 8-bit AdamW optimizer
- Gradient checkpointing (Phase 3+)
- FlashAttention-2
- Gradient accumulation (8 steps)

### Workstation (4x A6000, 192GB)

| Phase | Batch Size | Context | Est. Duration |
|-------|------------|---------|---------------|
| 1 | 128 | 336 | 3 days |
| 2 | 96 | 720 | 1 week |
| 3 | 64 | 1024 | 2 weeks |
| 4 | 32 | 4096 | 1.5 weeks |
| 5 | 64 | 2048 | 4 days |

**Total: 5-6 weeks**

**Configuration:**
- FSDP data parallelism
- ZeRO Stage 2
- Selective activation checkpointing

### Cloud Cluster (8x A100 80GB)

| Phase | Batch Size | Context | Est. Duration |
|-------|------------|---------|---------------|
| 1 | 512 | 512 | 1 day |
| 2 | 384 | 1024 | 3 days |
| 3 | 256 | 2048 | 1 week |
| 4 | 128 | 8192 | 1 week |
| 5 | 256 | 4096 | 2 days |

**Total: 2-3 weeks**

**Configuration:**
- FSDP + Tensor Parallelism (2x4)
- ZeRO Stage 3
- torch.compile for speed
- Full dataset utilization

---

## Dataset Acquisition Priority

### Tier 1: Immediate (Free, High Value)
1. **Binance Public Data** — Comprehensive crypto coverage
2. **FRED Economic Data** — Complete macro foundation
3. **Monash Archive + M4/M5** — Benchmark validation
4. **CBOE VIX** — Volatility/regime signals
5. **COT Reports** — Institutional positioning

### Tier 2: Short-Term (Free, Moderate Effort)
1. **Dukascopy Tick Data** — Forex validation & microstructure
2. **Yahoo Finance (yfinance)** — US equities baseline
3. **CryptoDataDownload** — Multi-exchange crypto
4. **Economic Calendar (ForexFactory)** — Event timestamps

### Tier 3: Medium-Term (Some Cost)
1. **Polygon.io** ($99+/mo) — Real-time production feeds
2. **Alpha Vantage** (Free tier) — Intraday equities
3. **Nasdaq Data Link** — Continuous futures

### Tier 4: Long-Term (Commercial/Academic)
1. **Refinitiv** — Institutional deployment
2. **Kaiko** — Crypto institutional grade
3. **WRDS** — Academic research

---

## Quick Start: First 4 Weeks

**Week 1:**
- Download Monash Archive, M4, ETT datasets
- Begin Phase 1 pre-training on benchmarks
- Acquire FRED API key, download core macro series
- Set up Binance data pipeline

**Week 2:**
- Continue Phase 1 (validate MASE < 1.0)
- Download VIX historical data
- Register for Dukascopy, begin forex tick download
- Integrate COT report parsing

**Week 3:**
- Complete Phase 1, validate transfer scores
- Begin Phase 2a with major forex pairs
- Add economic calendar integration
- Validate against Dukascopy tick data

**Week 4:**
- Complete Phase 2a (majors)
- Begin Phase 2b (cross pairs)
- Add Binance crypto to data pipeline
- First cross-pair correlation experiments
