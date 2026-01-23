## Plan: Build Forex Time-Series Foundation Model

Transform your existing multi-task forex trading system into a pre-trainable foundation model architecture that can be fine-tuned for specific trading tasks across 46 currency pairs (315M timesteps).

---

### Step 1: Create Self-Supervised Pre-Training Objectives

**Goal:** Train `SharedEncoder` to learn general time-series representations before any task-specific fine-tuning.

**Current State:**
- `SharedEncoder` ([models/agent_hybrid.py#L12-L143](models/agent_hybrid.py#L12-L143)) outputs a **160-dimensional** context vector (128 BiLSTM + 32 CNN)
- Architecture: CNN (local patterns) → BiLSTM (sequences) → Attention (aggregation)
- All current training is task-specific from scratch—encoder learns jointly with heads

**Required Changes:**

1. **Add `forward_features()` method** to `SharedEncoder` that returns full sequence features `[B, T, 160]` **before** attention aggregation—needed for per-timestep masked prediction:
   ```python
   def forward_features(self, x):
       lstm_out, _ = self.lstm(x)  # [B, T, 128]
       cnn_features = F.relu(self.cnn(x.permute(0,2,1))).permute(0,2,1)  # [B, T, 32]
       return torch.cat([lstm_out, cnn_features], dim=-1)  # [B, T, 160]
   ```

2. **Implement Masked Timestep Prediction** (primary pre-training objective):
   - Mask 15-25% of **patches** (10-20 timestep chunks, not individual points)—preserves temporal structure better than point masking
   - Decoder head reconstructs original values: $\mathcal{L}_{\text{mask}} = \frac{1}{|M|} \sum_{i \in M} \|x_i - \hat{x}_i\|_2^2$
   - Lower mask ratio than NLP (15% vs 15-30%) because financial time series have higher temporal continuity
   - **Research basis:** PatchTST (Nie et al., 2022), MOMENT (Goswami et al., 2024)

3. **Implement Contrastive Learning** (auxiliary pre-training objective):
   - Positive pairs: Same window with different augmentations, OR different pairs at same timestamp (correlated markets)
   - Negative pairs: Windows from different time periods or different regimes
   - InfoNCE loss: $\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k)/\tau)}$
   - Temperature τ = 0.07-0.1 (standard for visual/temporal contrast)
   - **Research basis:** TS-TCC (Eldele et al., 2021), TS2Vec (Yue et al., 2022)

4. **Add Next-Step Forecasting Head** (validates encoder captures predictive structure):
   - Single linear layer: `nn.Linear(160, num_features)` predicting next timestep's features
   - Lightweight sanity check that encoder isn't just memorizing

**Rationale:**
- Masked prediction forces encoder to learn **contextual representations** (what's missing given surroundings)
- Contrastive learning ensures **similar market conditions map to similar embeddings** (regime awareness)
- Pre-training on all 46 pairs simultaneously before task-specific heads creates **universal forex representations**

**Trade-offs:**
- Patch masking (recommended) vs point masking: Patches preserve candle patterns but reduce mask diversity
- Contrastive scope: In-batch negatives (simpler) vs memory bank (more negatives) vs cross-pair hard negatives

---

### Step 2: Build Cross-Pair Data Infrastructure

**Goal:** Enable training on all 46 pairs simultaneously with proper sampling and pair identity.

**Current State:**
- `IterableFXDataset` ([data/iterable_dataset.py#L60-L73](data/iterable_dataset.py#L60-L73)) is **strictly single-pair**
- Feature dimension: ~40-60 features per timestep (OHLCV + technicals + sentiment)
- Each pair trained independently—no cross-pair learning

**Data Inventory:**
| Asset Class | Pairs | Time Range | Est. Samples |
|-------------|-------|------------|--------------|
| FX Majors | 7 | 2010-2025 | ~55M |
| FX Crosses | 22 | 2010-2025 | ~175M |
| FX Emerging | 6 | Variable | ~48M |
| Crypto | 10 | 2019-2025 | ~31.5M |
| Gold | 1 | 2010-2025 | ~8M |
| **Total** | **46** | - | **~315M** |

**Required Changes:**

1. **Create `CrossPairDataset`** in [data/prepare_foundation_dataset.py](data/prepare_foundation_dataset.py):
   ```python
   class CrossPairDataset(IterableDataset):
       def __init__(self, pairs: list[str], sampling='sqrt'):
           self.pair_datasets = {p: IterableFXDataset(p, ...) for p in pairs}
           # sqrt sampling: weight ∝ √(data_size) to prevent major pair domination
           
       def __iter__(self):
           for seq, targets in weighted_round_robin(self.pair_datasets):
               yield seq, targets, pair_idx  # pair_idx for embeddings
   ```

2. **Implement Sampling Strategies:**
   - **Proportional:** Sample based on data size (major pairs dominate)
   - **Uniform:** Equal samples per pair (undersamples majors, oversamples crypto)
   - **√-proportional (recommended):** Balance between coverage and volume—prevents EUR/USD (8M samples) from drowning out SOL-USD (500K)

3. **Add Learnable Pair Embeddings** to `SharedEncoder`:
   - **Option A (Concatenation):** `pair_emb = nn.Embedding(46, 16)` → input becomes `[B, T, features+16]`
   - **Option B (Addition):** `pair_emb = nn.Embedding(46, 160)` → add to encoder output
   - **Option C (FiLM conditioning, recommended):** `γ, β = MLP(pair_emb)` → `output = γ * encoder(x) + β`
   
   FiLM allows pair-specific **modulation** without changing encoder architecture or feature dimensions.

4. **Handle Variable Data Lengths:**
   - FX majors: 15 years (~8M timesteps)
   - Crypto: 6 years (~3M timesteps)
   - Use **epoch = pass through smallest dataset**, oversample smaller pairs within epoch

**Rationale:**
- Cross-pair training creates **transfer learning** opportunities (EUR/USD patterns help GBP/USD)
- Pair embeddings allow model to learn **pair-specific adjustments** while sharing encoder
- √-proportional sampling is empirically best for imbalanced multi-domain datasets

**Trade-offs:**
- FiLM conditioning (expressive) vs simple addition (fewer parameters)
- √-sampling (balanced) vs curriculum (start with majors, add others)

---

### Step 3: Implement Two-Phase Training Pipeline

**Goal:** Phase 1 pre-trains encoder on self-supervised tasks; Phase 2 freezes encoder and fine-tunes task heads.

**Current State:**
- Single-phase training in [train/train_multitask.py](train/train_multitask.py)
- No learning rate scheduling (constant LR)
- Checkpoint saves entire model state—no separation of encoder vs heads

**Required Changes:**

1. **Create [run/pretrain_foundation.py](run/pretrain_foundation.py)** (Phase 1):
   ```python
   def pretrain(cfg):
       encoder = SharedEncoder(...)
       mask_head = MaskedPredictionHead(160, num_features)
       contrast_head = ProjectionHead(160, 128)  # Projects to contrastive space
       
       optimizer = AdamW([encoder, mask_head, contrast_head], lr=1e-4, weight_decay=0.01)
       scheduler = CosineAnnealingWithWarmup(warmup_ratio=0.05)
       
       for epoch in range(100):
           for batch in cross_pair_loader:
               loss_mask = masked_prediction_loss(encoder, mask_head, batch)
               loss_contrast = contrastive_loss(encoder, contrast_head, batch)
               loss = loss_mask + 0.1 * loss_contrast  # Mask is primary
               
       # Save ONLY encoder weights
       torch.save({'encoder': encoder.state_dict(), 'epoch': epoch}, 'pretrained_encoder.pt')
   ```

2. **Create [run/finetune_downstream.py](run/finetune_downstream.py)** (Phase 2):
   ```python
   def finetune(cfg, pretrained_path, target_pair):
       # Load pre-trained encoder
       model = DignityModel(cfg)
       model.price_encoder.load_state_dict(torch.load(pretrained_path)['encoder'])
       
       # Freeze encoder initially
       freeze_encoder(model.price_encoder)
       
       # Train only task heads with lower LR
       optimizer = AdamW(model.task_heads.parameters(), lr=5e-5)
       
       # Optional: Gradual unfreezing after warmup
       for epoch in range(20):
           if epoch > 5:
               unfreeze_layer(model.price_encoder.attention)
           if epoch > 10:
               unfreeze_layer(model.price_encoder.lstm)
   ```

3. **Implement Learning Rate Scheduling** (critical for foundation models):
   ```python
   # Pre-training: Warmup 5% of steps, then cosine decay to 0
   # Fine-tuning: Warmup 10% of steps, cosine decay to 1e-6
   scheduler = get_cosine_schedule_with_warmup(
       optimizer, 
       num_warmup_steps=int(0.05 * total_steps),
       num_training_steps=total_steps
   )
   ```

4. **Add Encoder Freezing Utilities** to [models/agent_hybrid.py](models/agent_hybrid.py):
   ```python
   def freeze_encoder(encoder, freeze=True):
       for param in encoder.parameters():
           param.requires_grad = not freeze
   
   def gradual_unfreeze(encoder, epoch, schedule={'attention': 5, 'lstm': 10, 'cnn': 15}):
       for layer_name, unfreeze_epoch in schedule.items():
           if epoch >= unfreeze_epoch:
               getattr(encoder, layer_name).requires_grad_(True)
   ```

5. **Pre-Training Quality Validation** (before fine-tuning):
   - **Linear probe:** Freeze encoder, train single linear layer for direction classification
   - Target: >55% accuracy indicates useful representations (random = 33% for 3-class)
   - **t-SNE visualization:** Embeddings should cluster by regime, not by pair

**Rationale:**
- Two-phase training prevents **catastrophic forgetting** of general representations
- Cosine LR schedule is empirically best for transformers/attention models
- Gradual unfreezing allows task-specific adaptation without destroying pre-trained features
- Linear probe validates pre-training before investing in full fine-tuning

**Trade-offs:**
- Full freeze (safer) vs gradual unfreeze (potentially better performance)
- Joint pre-training (mask + contrast) vs sequential (mask first, then contrast)

---

### Step 4: Enable Temporal Data Augmentation

**Goal:** Increase effective dataset size and encoder robustness through principled augmentation.

**Current State:**
- **No augmentation implemented** anywhere in codebase
- Implicit augmentation only: sliding window overlap, multiple pairs

**Required Changes:**

1. **Add Augmentation Module** in [data/augmentation.py](data/augmentation.py):

   **Time Warping (±10% speed):**
   ```python
   def time_warp(x, sigma=0.1):
       """Stretch/compress time via cubic spline interpolation."""
       # Safe for direction classification if warp < 15%
       # Changes candle durations but preserves relative patterns
   ```
   
   **Magnitude Scaling (±20%):**
   ```python
   def magnitude_scale(x, sigma=0.2):
       """Scale all values by random factor ~N(1, σ²)."""
       # Simulates different volatility regimes
       # Safe for log returns (preserves relative relationships)
   ```
   
   **Gaussian Noise:**
   ```python
   def add_noise(x, noise_level=0.00005):
       """Add noise calibrated to half-pip precision."""
       # Forex: 1 pip = 0.0001 for majors
       # Noise should be < 0.5 pips to stay within bid-ask spread
   ```
   
   **Window Jittering:**
   ```python
   def jitter_window(x, max_offset=5):
       """Random start offset within original sequence."""
       # Requires dataset-level implementation (access to longer series)
   ```
   
   **MixUp for Time Series:**
   ```python
   def mixup(x1, y1, x2, y2, alpha=0.2):
       """Linear interpolation between samples."""
       lam = np.random.beta(alpha, alpha)
       # Best for regression; classification needs soft labels
   ```

2. **Integrate into DataLoader** (online augmentation):
   ```python
   class AugmentedDataset(IterableDataset):
       def __init__(self, base_dataset, aug_prob=0.5):
           self.augmentations = [time_warp, magnitude_scale, add_noise]
           
       def __iter__(self):
           for x, y in self.base_dataset:
               if random.random() < self.aug_prob:
                   aug = random.choice(self.augmentations)
                   x = aug(x)
               yield x, y
   ```

3. **Contrastive-Specific Augmentations** (for pre-training):
   - **Strong augmentations** for contrastive positives: time_warp + noise + magnitude_scale
   - **Weak augmentations** for masked prediction: noise only (preserve reconstruction targets)

**Rationale:**
- Augmentation is **critical for foundation models**—prevents overfitting to specific patterns
- Time warping teaches **scale-invariant** pattern recognition
- Magnitude scaling improves **regime transfer** (model sees low-vol and high-vol versions)
- MixUp provides **soft regularization** for regression heads

**Noise Calibration (Forex-Specific):**
| Pair Type | 1 Pip | Recommended Noise |
|-----------|-------|-------------------|
| Majors (EUR/USD) | 0.0001 | 0.00005 |
| JPY pairs | 0.01 | 0.005 |
| Crypto | Varies | 0.1% of price |

**Trade-offs:**
- Online augmentation (infinite variety, slower) vs offline (faster, more storage)
- Aggressive augmentation (more robust) vs conservative (preserves label validity)

---

### Step 5: Integrate Uncertainty-Based Task Weighting

**Goal:** Replace static loss weights with learnable parameters that automatically balance 11 tasks.

**Current State:**
- Static weights in [train/train_multitask.py](train/train_multitask.py): all tasks weighted 1.0
- `UncertaintyWeightedLoss` exists in [train/loss_weighting.py](train/loss_weighting.py) but **not integrated**
- 11 tasks: 5 classification (CE loss) + 6 regression (MSE loss)

**Mathematical Foundation (Kendall et al., 2018):**

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{K} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

- Each task gets learnable parameter $\log \sigma_i$ (initialized to 0 → $\sigma = 1$)
- Gradient: $\frac{\partial \mathcal{L}}{\partial \sigma_i} = \frac{\sigma_i^2 - \mathcal{L}_i}{\sigma_i^3}$
- **Equilibrium:** $\sigma_i^2 \approx \mathcal{L}_i$ (uncertainty matches task difficulty)

**Required Changes:**

1. **Connect Existing Implementation** to training loop:
   ```python
   # In train/train_multitask.py
   from train.loss_weighting import UncertaintyWeightedLoss
   
   uncertainty_loss = UncertaintyWeightedLoss(
       task_names=['direction', 'volatility', 'return', ...],  # All 11 tasks
       initial_log_var=0.0  # σ = 1 for all tasks initially
   )
   
   # In training step:
   raw_losses = compute_all_task_losses(outputs, targets)
   weighted_losses, sigmas = uncertainty_loss(raw_losses)
   total_loss = sum(weighted_losses.values())
   ```

2. **Add Uncertainty Parameters to Optimizer:**
   ```python
   optimizer = AdamW([
       {'params': model.parameters(), 'lr': 1e-4},
       {'params': uncertainty_loss.parameters(), 'lr': 1e-3}  # Faster LR for σ
   ])
   ```

3. **Log Uncertainty Evolution** (debugging/interpretability):
   ```python
   # Track σ values over training
   wandb.log({f'sigma_{task}': sigma for task, sigma in sigmas.items()})
   # Expected: regression tasks → higher σ (larger loss scale)
   #           classification tasks → lower σ (harder gradients)
   ```

**Task Grouping Strategy:**
| Task | Type | Expected σ | Rationale |
|------|------|------------|-----------|
| direction | CE | Low (~0.5) | Primary task, tight gradients |
| return | MSE | High (~2-5) | Different scale than classification |
| volatility | CE | Low (~0.5) | Binary, well-defined |
| topk_returns | MSE | High (~3-10) | Multiple outputs, higher variance |

**Rationale:**
- Manual weight tuning doesn't scale to 11 tasks—uncertainty weighting **auto-balances**
- Tasks with higher loss (harder) automatically get **down-weighted** to prevent domination
- Regression vs classification scale differences handled automatically
- Provides **interpretability**: σ values reveal which tasks model finds difficult

**Trade-offs:**
- All tasks independent (current) vs grouped (share σ across classification tasks)
- Fixed σ during fine-tuning (stable) vs continued learning (adaptive)

---

### Step 6: Create Evaluation Framework

**Goal:** Rigorously benchmark foundation model against baselines and measure transfer efficiency.

**Current State:**
- Basic metrics in [eval/run_evaluation.py](eval/run_evaluation.py): accuracy, RMSE, confusion matrix
- TimesFM comparison via subprocess ([eval/timesfm_wrapper.py](eval/timesfm_wrapper.py))
- No transfer learning or sample efficiency evaluation

**Required Changes:**

1. **Create [eval/foundation_benchmarks.py](eval/foundation_benchmarks.py):**

   **Zero-Shot Evaluation:**
   ```python
   def eval_zero_shot(pretrained_encoder, held_out_pairs=['SOL-USD', 'USD/TRY', ...]):
       """Test on pairs never seen during pre-training."""
       # Train only linear probe head, freeze encoder
       # Baseline: Train from scratch on same pairs
       metrics = {
           'zero_shot_acc': ...,
           'scratch_acc': ...,
           'transfer_gap': zero_shot_acc - scratch_acc
       }
   ```
   
   **Sample Efficiency Curves:**
   ```python
   def eval_sample_efficiency(pretrained_encoder, target_pair, n_samples=[100, 1000, 10000, 'full']):
       """How much data needed to match from-scratch performance?"""
       results = {}
       for n in n_samples:
           pretrained_perf = finetune_and_eval(pretrained_encoder, data[:n])
           scratch_perf = train_from_scratch(data[:n])
           results[n] = {'pretrained': pretrained_perf, 'scratch': scratch_perf}
       
       # Key metric: samples needed for pretrained to match scratch@full
       return results, compute_sample_efficiency_ratio(results)
   ```
   
   **Cross-Domain Transfer:**
   ```python
   def eval_cross_domain(fx_pretrained_encoder, crypto_test_pairs):
       """Pre-train on FX, fine-tune on crypto."""
       # Measures domain generalization
       # Harder test: very different volatility profiles
   ```

2. **Head-to-Head TimesFM Comparison:**
   ```python
   def compare_to_timesfm(model, test_data, horizon=30):
       """Fair comparison: same test set, same horizon."""
       dignity_preds = model.forecast(test_data, horizon)
       tfm_preds = timesfm_wrapper.forecast_naive(horizon, test_data)
       
       metrics = {
           'dignity_rmse': rmse(dignity_preds, actuals),
           'timesfm_rmse': rmse(tfm_preds, actuals),
           'ensemble_rmse': rmse(0.5*dignity_preds + 0.5*tfm_preds, actuals)
       }
   ```

3. **Pre-Training Quality Metrics:**
   ```python
   def eval_pretrain_quality(encoder):
       """Validate representations before fine-tuning."""
       return {
           'linear_probe_acc': linear_probe_direction(encoder),  # Target: >55%
           'regime_clustering': silhouette_score(embeddings, regimes),  # Target: >0.3
           'pair_separation': inter_pair_distance(embeddings),  # Lower = more shared
       }
   ```

**Benchmark Targets:**
| Metric | Random | From-Scratch | Pre-Trained Target |
|--------|--------|--------------|-------------------|
| Direction Acc (3-class) | 33% | ~58% | >62% |
| Zero-shot Acc | 33% | N/A | >50% |
| Sample Efficiency | 1x | 1x | 3-5x (same perf with 20-33% data) |
| TimesFM RMSE Gap | - | Baseline | ≤TimesFM |

**Rationale:**
- Zero-shot tests **generalization** (the core promise of foundation models)
- Sample efficiency measures **practical value** (less data needed for new pairs)
- TimesFM comparison provides **external baseline** (Google's pre-trained model)
- Regime clustering validates **semantic quality** of learned representations

**Trade-offs:**
- Held-out pairs (strict test) vs leave-one-out cross-validation (more data points)
- Single checkpoint eval vs ensemble of checkpoints

---

### Further Considerations

1. **Context Length Strategy:**
   - Current: t_in=120 (2 hours of minute bars)
   - Options: 240 (4 hours), 480 (8 hours), multi-scale pre-training
   - **Recommendation:** Pre-train with variable lengths [60, 120, 240] to learn scale-invariant patterns; use [utils/attention_optimization.py](utils/attention_optimization.py) chunking for 480+
   - Trade-off: Longer context captures regime transitions but 4x memory cost

2. **Curriculum Learning:**
   - Option A: Start with easy tasks (next-step prediction) → hard tasks (regime detection)
   - Option B: All 11 tasks from epoch 1, let uncertainty weighting balance
   - **Recommendation:** Start with masked prediction only (epochs 1-20), add contrastive (epochs 20-50), then fine-tune with uncertainty weighting
   - Rationale: Masked prediction is most stable; contrastive needs good representations to define positives/negatives

3. **Data Expansion:**
   - Current: 46 pairs, 315M samples (forex/crypto only)
   - Expansion options: Stocks, ETFs, indices, commodities (potentially 1000+ instruments)
   - **Recommendation:** Establish forex-only baseline first; expansion introduces domain shift complexity
   - Trade-off: More data = better generalization, but harder to debug and longer training

4. **Compute Budget:**
   - Pre-training estimate: 315M × 100 epochs × ~10ms/sample ≈ **3,500 GPU-hours**
   - With 8× A100: ~18 days; with consumer GPU: ~6 months
   - **Recommendation:** Start with 10% data subset (31.5M samples) for architecture validation, then scale
   - Consider: PyTorch Lightning + DeepSpeed for efficient distributed training
