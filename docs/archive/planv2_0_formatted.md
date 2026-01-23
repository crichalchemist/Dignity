# 3-Month Implementation Plan: Foundation Model for Social Impact

**Target:** Working model by April 2026  
**Priority:** Data acquisition → Pre-training → Fine-tuning → Deployment

---

## Executive Summary

Based on your resources (Colab 12h/day + 4 Tesla GPUs) and timeline (3 months), here's a data-first, incremental approach that builds value at each stage while ensuring you have all 46 pairs trained by April.

---

## Phase 1: Complete Data Acquisition (Weeks 1-4)

### Current Data Status

| Asset Class | Available | Missing | Total | Quality |
|-------------|-----------|---------|-------|---------|
| FX | 10 pairs | 21 pairs | 31 | Good (2010-2023) |
| Crypto | 7 pairs (2023 only) | 3 missing + full history | 10 | Poor (1 year only) |
| Commodities | 0 pairs | 1 (gold) | 1 | None |

### Gaps to Fill

- 21 missing FX pairs (USD/JPY, USD/CHF, USD/CAD, AUD/USD, etc.)
- 3 missing crypto pairs (LINK, MATIC, AVAX)
- 10 crypto pairs need 2019-2025 history (currently only 2023)
- 1 gold pair (XAU/USD, 2010-2025)

### Week 1-2: FX Data Acquisition

#### Command: Use existing histdata.py downloader

```bash
# This reads pairs.csv and downloads all pairs automatically
cd /Volumes/Containers/Sequence
python data/downloaders/histdata.py \
  --start-year 2010 \
  --end-year 2025 \
  --max-downloads 1000 \
  --output data/histdata
```

#### Expected Timeline

- Each FX pair: ~15 years × 12 months = 180 files
- HistData has no API limits, just download time
- With good internet: 1-2 hours for all 31 pairs

#### Validation

After download, verify each pair has 15 files (2010-2024):

```bash
for pair in data/histdata/*/; do
  count=$(find "$pair" -name "*.csv" 2>/dev/null | wc -l)
  echo "$(basename $pair): $count files"
done
```

#### Risk Management

- **Risk:** HistData may block aggressive downloads
- **Mitigation:** Download 5 pairs at a time with 1-hour breaks

### Week 2-3: Crypto Data Acquisition

#### Command: Use yfinance_downloader.py

```bash
# Download all crypto pairs from pairs.csv
python data/downloaders/yfinance_downloader.py \
  --pairs-csv pairs.csv \
  --start 2019-09-01 \
  --end 2025-12-31 \
  --interval 1m \
  --output-dir data/crypto_full
```

#### Expected Timeline

- Each crypto pair: ~6 years of minute data
- yfinance is free but rate-limited (unknown limits)
- Estimate: 2-3 hours per pair → 20-30 hours total

#### Data Format

Crypto data format matches FX (semicolon-delimited CSV with `datetime;O;H;L;C;V`)

### Week 3: Gold Data Acquisition

#### Option A: yfinance (faster, futures data)

```bash
python data/downloaders/yfinance_downloader.py \
  --pairs xauusd \
  --start 2010-01-01 \
  --end 2025-12-31 \
  --interval 1m \
  --output-dir data/commodities
```

#### Option B: HistData (more reliable, spot data)

- Gold available as XAUUSD from HistData
- Same process as FX pairs

### Week 4: Data Validation & Consolidation

#### Automated Validation Script

Create `scripts/validate_all_data.py`:

```python
def validate_pair_data(pair_dir: Path):
    """Check data quality for a single pair."""
    files = list(pair_dir.glob("*.csv"))
    
    # Check file count
    expected_years = 15  # 2010-2024
    if len(files) < expected_years * 0.8:  # Allow some missing years
        return False, "Insufficient years of data"
    
    # Check data format
    sample = pd.read_csv(files[0], sep=';', header=None)
    if len(sample.columns) != 6:
        return False, f"Expected 6 columns, got {len(sample.columns)}"
    
    # Check for NaN
    if sample.isnull().any().any():
        return False, "NaN values found"
    
    # Check OHLC relationships
    invalid = (sample[3] < sample[2]) | (sample[3] < sample[4])
    if invalid.any():
        return False, f"{invalid.sum()} invalid OHLC rows"
    
    return True, "OK"
```

#### Run Validation

```bash
python scripts/validate_all_data.py \
  --data-root data \
  --report validation_report.csv
```

#### Data Consolidation Structure

Create consolidated dataset directory:

```
data/consolidated_dataset/
  ├── fx/
  │   ├── eurusd/
  │   ├── gbpusd/
  │   └── ... (29 more)
  ├── crypto/
  │   ├── btcusd/
  │   ├── ethusd/
  │   └── ... (7 more)
  └── commodities/
      └── xauusd/
```

---

## Phase 2: Cross-Pair Dataset & Infrastructure (Weeks 5-6)

### Week 5: Build CrossPairDataset

Create `data/cross_pair_dataset.py`:

```python
class CrossPairDataset(IterableDataset):
    """
    Multi-pair streaming dataset with configurable sampling.
    
    Supports:
    - Proportional sampling (√n_pairs)
    - Uniform sampling (equal samples per pair)
    - Custom sampling weights
    """
    
    def __init__(
        self,
        pairs: list[str],
        data_root: Path,
        sampling: str = "sqrt_proportional",
        t_in: int = 120,
        t_out: int = 10,
        pair_embeddings: bool = True,
    ):
        self.pairs = pairs
        self.sampling = sampling
        
        # Load individual pair datasets
        self.pair_datasets = {
            pair: IterableFXDataset(
                pair=pair,
                data_root=data_root,
                t_in=t_in,
                t_out=t_out,
            )
            for pair in pairs
        }
        
        # Calculate sampling weights
        if sampling == "sqrt_proportional":
            pair_sizes = {
                p: len(ds) 
                for p, ds in self.pair_datasets.items()
            }
            self.weights = {
                p: math.sqrt(size) / sum(math.sqrt(s) for s in pair_sizes.values())
                for p, size in pair_sizes.items()
            }
        elif sampling == "uniform":
            weight = 1.0 / len(pairs)
            self.weights = {p: weight for p in pairs}
        elif sampling == "proportional":
            total = sum(len(ds) for ds in self.pair_datasets.values())
            self.weights = {
                p: len(ds) / total 
                for p, ds in self.pair_datasets.items()
            }
    
    def __iter__(self):
        """Iterate with weighted sampling."""
        while True:
            # Select pair based on weights
            pair = random.choices(
                self.pairs, 
                weights=[self.weights[p] for p in self.pairs],
                k=1
            )[0]
            
            # Get sample from selected pair
            dataset = self.pair_datasets[pair]
            try:
                seq, targets = next(iter(dataset))
                yield seq, targets, pair
            except StopIteration:
                continue
```

### Pair Embedding Support

Add pair embedding support to `models/agent_hybrid.py`:

```python
class SharedEncoderWithPairEmbedding(nn.Module):
    def __init__(self, num_features, num_pairs=46, embedding_dim=16):
        super().__init__()
        # Base encoder
        self.encoder = SharedEncoder(num_features)
        
        # Learnable pair embeddings
        self.pair_embedding = nn.Embedding(num_pairs, embedding_dim)
        
        # Option C: FiLM conditioning (recommended in plan)
        self.film_scale = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder.output_dim),
            nn.Sigmoid()
        )
        self.film_shift = nn.Linear(embedding_dim, self.encoder.output_dim)
    
    def forward(self, x, pair_idx):
        # Get base encoding
        context, attn = self.encoder(x)
        
        # Get pair embedding
        pair_emb = self.pair_embedding(pair_idx)
        
        # Apply FiLM modulation
        scale = self.film_scale(pair_emb)
        shift = self.film_shift(pair_emb)
        conditioned = scale * context + shift
        
        return conditioned, attn
```

### Week 6: Pre-Processing All 46 Pairs

Create `run/prepare_foundation_dataset.py`:

```python
def prepare_all_pairs(
    pairs: list[str],
    data_root: Path,
    output_root: Path,
    t_in: int = 120,
    t_out: int = 10,
):
    """
    Prepare all 46 pairs for pre-training.
    
    For each pair:
    1. Load raw OHLCV data
    2. Convert to UTC and deduplicate
    3. Build technical indicators
    4. Save processed data (train/val/test splits)
    """
    
    for pair in pairs:
        print(f"Processing {pair}...")
        
        # Load raw data
        raw_files = list((data_root / pair).glob("*.csv"))
        df = load_and_concatenate(raw_files)
        
        # Preprocess
        df = convert_to_utc_and_dedup(df)
        df = build_feature_frame(df, cfg=feature_config)
        df = df.dropna()
        
        # Split train/val/test (time-ordered)
        n = len(df)
        train_df = df.iloc[:int(0.7*n)]
        val_df = df.iloc[int(0.7*n):int(0.85*n)]
        test_df = df.iloc[int(0.85*n):]
        
        # Save
        output_dir = output_root / pair
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(output_dir / "train.parquet")
        val_df.to_parquet(output_dir / "val.parquet")
        test_df.to_parquet(output_dir / "test.parquet")
        
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

#### Run on Lightning.ai

```bash
# Prepare all pairs in parallel
python run/prepare_foundation_dataset.py \
  --pairs-file pairs.csv \
  --data-root data/consolidated_dataset \
  --output-root data/processed_foundation \
  --workers 4
```

**Expected timeline:** 4-8 hours for all 46 pairs

---

## Phase 3: Pre-Training Experiments (Weeks 7-8)

### Week 7: Masked Prediction Pre-Training

Create `run/pretrain_foundation.py`:

```python
def masked_prediction_loss(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
    mask_ratio: float = 0.15,
    patch_size: int = 15,
) -> torch.Tensor:
    """
    Masked timestep prediction loss.
    
    Strategy: Mask patches (15-timestep chunks) rather than points.
    """
    seq, targets, pair_idx = batch
    seq = seq.to(device)
    pair_idx = pair_idx.to(device)
    
    batch_size, seq_len, num_features = seq.shape
    
    # Mask patches
    num_patches = seq_len // patch_size
    num_mask = int(num_patches * mask_ratio)
    mask_indices = random.sample(range(num_patches), num_mask)
    
    # Create mask tensor [B, T]
    mask = torch.ones(batch_size, seq_len, device=device)
    for idx in mask_indices:
        start = idx * patch_size
        end = min(start + patch_size, seq_len)
        mask[:, start:end] = 0
    
    # Apply mask
    masked_seq = seq * mask.unsqueeze(-1)
    
    # Forward pass
    context, attn = model(masked_seq, pair_idx)
    
    # Decode to reconstruct masked patches
    reconstruction = model.decode_to_features(context)
    
    # Compute MSE loss on masked patches only
    masked_positions = mask == 0
    loss = F.mse_loss(
        reconstruction[masked_positions],
        seq[masked_positions]
    )
    
    return loss, mask
```

#### Training Loop (Small-Scale Validation)

```python
def pretrain_phase1(
    pairs: list[str],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda"),
):
    # Initialize model
    encoder = SharedEncoderWithPairEmbedding(
        num_features=60,  # OHLCV + technicals
        num_pairs=46,
    )
    decoder = nn.Linear(encoder.encoder.output_dim, 60)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    
    # Dataset
    dataset = CrossPairDataset(
        pairs=pairs[:5],  # Start with 5 pairs for validation
        data_root="data/processed_foundation",
        sampling="sqrt_proportional",
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # Training
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            loss, mask = masked_prediction_loss(encoder + decoder, batch, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_acc = linear_probe(encoder, val_loader)
            print(f"  Linear Probe Acc: {val_acc:.2%}")
            if val_acc > 0.50:
                print("  ✓ Reached target accuracy!")
                break
```

#### Run on Lightning.ai

```bash
# Small-scale validation (5 pairs, 10% data)
python run/pretrain_foundation.py \
  --pairs eurusd,gbpusd,btcusd,ethusd,xauusd \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-4 \
  --data-root data/processed_foundation \
  --output checkpoints/pretrain_phase1 \
  --gpus 0,1,2,3
```

**Expected timeline:** 20-30 hours on 4 GPUs

**Success criteria:**
- Mask loss decreases monotonically
- Linear probe accuracy >50% (random = 33%)
- No OOM errors or crashes

### Week 8: Scaling to All Pairs

If Phase 1 succeeds, scale up:

```bash
# Full pre-training on all 46 pairs, full dataset
python run/pretrain_foundation.py \
  --pairs-file pairs.csv \
  --epochs 50 \
  --batch-size 512 \
  --lr 1e-4 \
  --data-root data/processed_foundation \
  --output checkpoints/foundation_model \
  --gpus 0,1,2,3
```

#### Checkpointing Strategy

Handle 12-hour Colab limits with regular checkpoints:

```python
# Save checkpoint every 5 epochs
if (epoch + 1) % 5 == 0:
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'loss': epoch_loss,
    }, f"checkpoints/pretrain/checkpoint_epoch_{epoch}.pt")
```

**Expected timeline:** 150-200 hours on 4 GPUs

**Reality check:** At 12 hours/day on Colab, this would take ~15-17 days. Spread over Month 2 (February), this is feasible.

---

## Phase 4: Fine-Tuning for Profitability (Weeks 9-11)

### Week 9: Baseline Comparison

Create `run/fine_tune_comparison.py`:

```python
def compare_strategies(
    target_pair: str,
    pretrained_encoder_path: str,
    epochs: int = 20,
):
    """
    Compare 3 fine-tuning strategies:
    1. From scratch (baseline)
    2. Pre-trained, encoder frozen
    3. Pre-trained, gradual unfreeze
    """
    
    results = {}
    
    # Strategy 1: From scratch
    scratch_model = DignityModel(cfg)
    results['scratch'] = train_model(
        scratch_model, 
        target_pair, 
        epochs, 
        lr=1e-4
    )
    
    # Strategy 2: Pre-trained, frozen encoder
    frozen_model = load_pretrained_model(pretrained_encoder_path)
    freeze_encoder(frozen_model)
    results['frozen'] = train_model(
        frozen_model, 
        target_pair, 
        epochs, 
        lr=5e-5
    )
    
    # Strategy 3: Gradual unfreeze
    unfreeze_model = load_pretrained_model(pretrained_encoder_path)
    results['gradual'] = train_model_with_unfreeze(
        unfreeze_model,
        target_pair,
        epochs,
        lr=5e-5,
        unfreeze_schedule={5: 'attention', 10: 'lstm', 15: 'cnn'}
    )
    
    return results
```

#### Test on High-Liquidity Pairs

```bash
# Test on 3 pairs: EURUSD (FX), BTCUSD (crypto), XAUUSD (gold)
for pair in eurusd btcusd xauusd; do
  python run/fine_tune_comparison.py \
    --target-pair $pair \
    --pretrained-path checkpoints/foundation_model/best_encoder.pt \
    --output results/fine_tune_$pair.json
done
```

**Expected timeline:** 40-60 hours on 4 GPUs (3 pairs × 3 strategies)

### Week 10: Profitability Testing

Create `eval/profitability_benchmark.py`:

```python
def backtest_profitability(
    model: nn.Module,
    pair: str,
    test_data: pd.DataFrame,
    transaction_cost: float = 0.0002,  # 2 pips for FX
):
    """
    Backtest model for profitability.
    
    Returns:
    - Total PnL
    - Sharpe ratio
    - Max drawdown
    - Win rate
    """
    
    model.eval()
    positions = []
    cumulative_pnl = 0
    
    for i in range(0, len(test_data) - 120):
        # Get prediction
        window = test_data.iloc[i:i+120]
        features = preprocess_window(window)
        with torch.no_grad():
            logits = model(features)
        
        # Convert to position
        pred_class = torch.argmax(logits, dim=1).item()
        if pred_class == 0:  # Down
            position = -1
        elif pred_class == 2:  # Up
            position = 1
        else:  # Flat
            position = 0
        
        # Execute trade
        future_return = test_data.iloc[i+130]['close'] / test_data.iloc[i+120]['close'] - 1
        trade_pnl = position * future_return - transaction_cost * abs(position)
        cumulative_pnl += trade_pnl
        
        positions.append({
            'timestamp': test_data.iloc[i+130]['datetime'],
            'position': position,
            'pnl': trade_pnl,
            'cumulative_pnl': cumulative_pnl,
        })
    
    # Calculate metrics
    returns = [p['pnl'] for p in positions]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    drawdown = max(cumulative_pnl - p['cumulative_pnl'] for p in positions)
    win_rate = sum(r > 0 for r in returns) / len(returns)
    
    return {
        'total_pnl': cumulative_pnl,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
        'win_rate': win_rate,
    }
```

#### Run on All Fine-Tuned Models

```bash
for model in results/fine_tune_*.json; do
  python eval/profitability_benchmark.py \
    --model-path $model \
    --test-data data/processed_foundation/*/test.parquet \
    --output results/profitability_$model.json
done
```

### Week 11: Select Best Models

Analyze results:

```python
def select_best_models(profitability_results: dict) -> list[str]:
    """
    Select models with:
    - Sharpe ratio > 1.5
    - Max drawdown < 0.15
    - Win rate > 0.52
    """
    
    best_models = []
    for model_name, metrics in profitability_results.items():
        if (metrics['sharpe_ratio'] > 1.5 and 
            metrics['max_drawdown'] < 0.15 and
            metrics['win_rate'] > 0.52):
            best_models.append(model_name)
    
    # Sort by Sharpe ratio
    best_models.sort(
        key=lambda m: profitability_results[m]['sharpe_ratio'],
        reverse=True
    )
    
    return best_models[:5]  # Top 5 models
```

---

## Phase 5: Production Deployment (Weeks 12-13)

### Week 12: Model Optimization & Production Prep

#### Quantization for Deployment

```python
# Reduce model size for faster inference
from torch.quantization import quantize_dynamic
model = load_best_model("results/best_model.pt")
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
torch.save(quantized_model, "models/production_model_quantized.pt")
```

#### Inference Script

Create `deploy/infer.py`:

```python
def predict_live(
    model: nn.Module,
    pair: str,
    last_n_minutes: int = 120,
):
    """Get prediction for live trading."""
    
    # Fetch latest data
    live_data = fetch_latest_ohlcv(pair, n=last_n_minutes)
    features = preprocess_live_data(live_data)
    
    # Predict
    with torch.no_grad():
        logits = model(features)
        probs = F.softmax(logits, dim=1)
    
    return {
        'pair': pair,
        'prediction': torch.argmax(logits).item(),
        'confidence': probs.max().item(),
        'probabilities': {
            'down': probs[0].item(),
            'flat': probs[1].item(),
            'up': probs[2].item(),
        },
        'timestamp': datetime.now(),
    }
```

### Week 13: Final Testing & Social Impact Integration

#### Paper Trading Validation

```bash
# Run model in paper trading mode
python deploy/paper_trader.py \
  --model models/production_model_quantized.pt \
  --pairs eurusd,btcusd,xauusd \
  --duration 168h  # 1 week
  --output results/paper_trading.csv
```

#### Social Impact Calculator

Create metrics for social impact:

```python
def calculate_social_impact(
    trading_results: dict,
    donation_percentage: float = 0.10,  # 10% of profits
) -> dict:
    """
    Calculate potential social impact from trading profits.
    
    Assuming $10,000 starting capital:
    - 10% Sharpe ratio → ~25% annual return → $2,500/year
    - Donate 10% → $250/year
    - Scale to $100K capital → $2,500/year
    """
    
    capital = 10000
    annual_return = capital * trading_results['sharpe_ratio'] * 0.25
    donation = annual_return * donation_percentage
    
    # Impact examples
    meals_provided = donation / 0.50  # $0.50 per meal
    children_schooled = donation / 365  # $365/year per child
    trees_planted = donation / 10  # $10 per tree
    
    return {
        'annual_profit': annual_return,
        'annual_donation': donation,
        'meals_provided': meals_provided,
        'children_schooled': children_schooled,
        'trees_planted': trees_planted,
    }
```

---

## Monthly Breakdown

### Month 1 (January): Data Acquisition

- **Week 1-2:** Download 21 missing FX pairs via HistData
- **Week 2-3:** Download 10 crypto pairs (full history 2019-2025) via yfinance
- **Week 3:** Download gold data (XAU/USD) via yfinance
- **Week 4:** Validate all 46 pairs, consolidate into `data/processed_foundation/`

**Deliverables:**
- All 46 pairs with complete data
- Validation report confirming data quality
- Consolidated dataset structure ready for pre-processing

**Resource:** Local machine (GitHub workspace)

### Month 2 (February): Infrastructure & Pre-Training

- **Week 5:** Build CrossPairDataset with 3 sampling strategies
- **Week 5:** Add pair embedding support to SharedEncoder
- **Week 6:** Pre-process all 46 pairs (train/val/test splits, features)
- **Week 7:** Phase 1 pre-training validation (5 pairs, 10% data)
- **Week 8:** Scale to all 46 pairs, full pre-training (50 epochs)

**Deliverables:**
- CrossPairDataset working with all 46 pairs
- Pre-trained encoder saved as `checkpoints/foundation_model/best_encoder.pt`
- Linear probe accuracy >50% (validation of pre-training)

**Resource:** Lightning.ai (4 Tesla GPUs) + Colab (for smaller experiments)  
**GPU Hours:** ~200 hours (50 epochs × 46 pairs)

### Month 3 (March): Fine-Tuning & Production

- **Week 9:** Compare 3 fine-tuning strategies on 3 test pairs
- **Week 10:** Profitability benchmarking on all fine-tuned models
- **Week 11:** Select top 5 models by Sharpe ratio
- **Week 12:** Model quantization and inference script
- **Week 13:** Paper trading validation (1 week) + social impact calculations

**Deliverables:**
- 5 production-ready models with quantized weights
- Profitability metrics (Sharpe, drawdown, win rate)
- Social impact calculator and projections
- Paper trading results for 1 week

**Resource:** Lightning.ai (4 GPUs) + Local for testing  
**GPU Hours:** ~100 hours

---

## Resource Allocation Summary

| Phase | Colab Hours | Lightning.ai Hours | Local Hours |
|-------|-------------|-------------------|-------------|
| Data Acquisition | 0 | 0 | 40-60 |
| Infrastructure | 10 | 0 | 20 |
| Pre-Training | 20 | 180 | 0 |
| Fine-Tuning | 10 | 90 | 10 |
| Production | 5 | 10 | 20 |
| **Total** | **45** | **280** | **150** |

**Colab Usage:** 45 hours ÷ 12 hours/day = ~4 days (spread over 3 months)  
**Lightning.ai:** 280 hours on 4 GPUs = ~70 days of single-GPU equivalent

---

## Success Criteria

### Data Acquisition (Week 4)

- ✅ All 46 pairs downloaded with 2010-2025 (FX/gold) or 2019-2025 (crypto)
- ✅ Validation report shows <5% invalid OHLC rows per pair
- ✅ Consolidated dataset structure ready for pre-processing

### Pre-Training (Week 8)

- ✅ Mask loss decreases monotonically over 50 epochs
- ✅ Linear probe accuracy >50% on held-out pairs
- ✅ Encoder checkpoints saved every 5 epochs

### Fine-Tuning (Week 11)

- ✅ Pre-trained + frozen encoder outperforms from-scratch by ≥2%
- ✅ Sharpe ratio >1.5 on at least 3 pairs
- ✅ Win rate >52% on test set

### Production (Week 13)

- ✅ 5 quantized models ready for deployment
- ✅ Inference latency <10ms per prediction
- ✅ Paper trading shows positive PnL for 1 week
- ✅ Social impact calculator shows $250+ annual donation potential

---

## Risk Mitigation

### Risk 1: HistData Blocks Downloads

**Likelihood:** Medium  
**Impact:** High (data acquisition blocked)  
**Mitigation:**
- Download 5 pairs at a time with 1-hour breaks
- Use yfinance/TwelveData as fallback for missing pairs

### Risk 2: Pre-Training Doesn't Converge

**Likelihood:** Medium  
**Impact:** High (wasted 180 GPU-hours)  
**Mitigation:**
- Start with 5-pair validation (Week 7) before full training
- Reduce to 20 epochs if loss plateaus early
- Consider using existing DignityModel as baseline

### Risk 3: Fine-Tuned Models Not Profitable

**Likelihood:** High (markets are unpredictable)  
**Impact:** Medium (social impact goal delayed)  
**Mitigation:**
- Test on 3 diverse pairs (FX, crypto, gold)
- Use risk management (position limits, drawdown stops)
- Consider ensemble of top 5 models instead of single best

### Risk 4: Lightning.ai Runs Out of Credits

**Likelihood:** Low  
**Impact:** Medium (training delays)  
**Mitigation:**
- Pre-training already uses daily checkpoints
- Fall back to Colab 12-hour/day runs
- Reduce batch size to fit on single GPU

---

## Questions for You

### 1. Data Acquisition Strategy

- Should we download ALL 31 FX pairs via HistData, or focus on top 20 most liquid?
- For crypto, should we use yfinance (free, unknown limits) or TwelveData (800 calls, more reliable)?
- Do you have API keys for TwelveData, or should we rely on yfinance?

### 2. Pre-Training Scope

- Should we start with 15-pair validation (all available data) or 5-pair validation (faster)?
- If 5-pair validation fails, should we abort or troubleshoot?
- What's your minimum acceptable linear probe accuracy to proceed to full pre-training?

### 3. Fine-Tuning Priorities

- Which pairs are most important for your social impact goals?
- Should we optimize for Sharpe ratio, total return, or win rate?
- Do you want to backtest on all 46 pairs or focus on 3-5 high-liquidity pairs?

### 4. Production Timeline

- Is April deadline hard or soft? (Can we go into May if needed?)
- Do you want a single "best" model, or an ensemble of 5 models?
- What's your expected starting capital for trading?

### 5. Social Impact Integration

- Which causes are you targeting? (Education, hunger, environment, etc.)
- Do you want automated donation integration, or manual?
- Should the impact calculator use fixed percentages or allow user input?

---

## Next Steps (This Week)

Based on your priorities, I recommend starting with:

### Immediate Actions

1. Validate current data gaps - Run inventory on all 46 pairs
2. Start FX downloads - Begin with 5 pairs to test HistData limits
3. Set up Lightning.ai environment - Verify 4 GPU access

### This Week's Goals

- Download 10 missing FX pairs
- Validate data format for all downloads
- Create `data/consolidated_dataset/` structure
- Test Lightning.ai connection and GPU availability
