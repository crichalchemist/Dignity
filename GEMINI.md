# Dignity Core — Quant Finance System (for Gemini)

This document provides the primary instructional context for the Dignity Core project: a privacy-preserving deep learning framework for modeling transactional behavior, with a fully operational quantitative finance cascade.

## Context

Dignity Core is a privacy-preserving transaction modeling framework with a fixed CNN1D→StackedLSTM→AdditiveAttention backbone and four task heads (risk, forecast, policy, cascade). The cascade mode implements a hierarchical 4-head pipeline: RegimeHead → RiskHead → AlphaHead → PolicyHead. Each head conditions the next, while the backbone remains shared.

**Guided Learning:** End-to-end training of the 4-stage cascade can cause gradient vanishing. The solution applies auxiliary supervision losses at each head during training. This improved Annualized Rate of Return (ARR) from 17.97% to 25.28% on quant backtests. `cascade_loss()` in `models/dignity.py` implements this.

**Asset-class-aware thresholds:** Signal parameters calibrate per instrument class via `AssetConfig` in `core/signals.py`. DC thresholds differ by 10x across forex, crypto, equity, and commodity.

**DC intrinsic time:** `SignalProcessor.dc_state_machine()` implements a full directional-change state machine with overshoot tracking and bars-since-event counting.

## Architecture Overview

```
Data Sources (synthetic, CSV, MetaApi)
  └─ data/source/
       │
       ▼
core/privacy.py  [PrivacyManager — hashing, quantization, DP noise]
       │
       ▼
core/signals.py  [SignalProcessor — 32 features, AssetConfig per instrument]
       │
       ▼
data/pipeline.py  [TransactionPipeline — signal computation, scaling, windowing]
       │
       ▼
models/backbone/hybrid.py  [DignityBackbone — CNN1D → StackedLSTM → AdditiveAttention]
       │  context [B, 256]
       ▼
models/head/regime.py  [RegimeHead]  → regime_probs [B, 4]  (calm/trending/volatile/crisis)
       │  cat(context, regime_probs) [B, 260]
       ▼
models/head/risk.py  [RiskHead]  → (var_estimate [B,1], position_limit [B,1])
       │  cat(context, regime_probs) [B, 260]
       ▼
models/head/alpha.py  [AlphaHead]  → alpha_score [B, 1] in [-1, 1]
       │  cat(context, alpha_score, var_estimate) [B, 258]
       ▼
models/head/policy.py  [PolicyHead]  → (action_logits [B, n_actions], value [B, 1])
       │
       ▼
data/source/metaapi.py  [MetaApiExecutor — live trade execution]
```

Single-head tasks (risk, forecast, policy) use only the backbone + one head. The cascade mode chains all four heads with Guided Learning loss.

## Critical Files

### Unchanged (do not touch)
- `models/backbone/hybrid.py`, `cnn1d.py`, `lstm.py`, `attention.py` — backbone frozen

### Core Files
| File | Purpose |
|------|---------|
| `core/config.py` | `DignityConfig` with `ModelConfig`, `DataConfig`, `TrainConfig`, `ExecutionConfig` |
| `core/signals.py` | `SignalProcessor` — 32 features, `AssetConfig`, DC state machine |
| `core/privacy.py` | `PrivacyManager` — hashing, quantization, DP noise, sanitization |
| `models/dignity.py` | `Dignity` model — task routing, `forward_cascade()`, `cascade_loss()` |
| `models/head/regime.py` | `RegimeHead` — softmax regime classification |
| `models/head/risk.py` | `RiskHead` — dual sigmoid (VaR + position limit) |
| `models/head/alpha.py` | `AlphaHead` — tanh-bounded alpha score |
| `models/head/policy.py` | `PolicyHead` — actor-critic with `sample_action()` |
| `data/pipeline.py` | `TransactionPipeline` — fit/transform/sequence windowing |
| `data/source/synthetic.py` | `SyntheticGenerator` — OHLCV and transaction data |
| `data/source/metaapi.py` | `MetaApiSource` (streaming) + `MetaApiExecutor` (execution) |
| `train/engine.py` | Training loops, `train_cascade_epoch`, cosine scheduler |
| `train/cli.py` | `dignity-train` entry point, cascade/single-head branching |
| `export/to_onnx.py` | ONNX export with cascade wrapper, verification, benchmarking |
| `config/train_quant_paper.yaml` | Paper trading config (safe, default) |
| `config/train_quant.yaml` | Live execution config (gated) |

## Code Conventions

- **Functional style preferred**: Pure functions, immutable data, composition over inheritance. `nn.Module` used where PyTorch requires it.
- **Ruff for linting/formatting**: Config in `pyproject.toml`. Line length 100, double quotes, isort-compatible imports.
- **Type hints everywhere**: PEP 484 style, primary function documentation.
- **Minimal docstrings**: Explain "why" not "what."
- **`.editorconfig`**: 4-space indent, UTF-8, LF line endings, no final newline.
- **`model.train(False)` not `model.eval()`**: Security hook flags Python's built-in `eval()`.

## Testing

- **324 tests** across 6 files: `test_core.py`, `test_data.py`, `test_models.py`, `test_train.py`, `test_export.py`, `test_backtest.py`
- Fixtures in `tests/conftest.py`: `device`, `sample_sequence` (`[4, 100, 32]`), `sample_labels` (`[4]`), auto-seeding (seed 42)
- Markers: `unit`, `integration`, `slow`, `fast`, `real_api`, `regression`, `performance`
- Run all: `pytest tests/ -v`

## Verification

```bash
# 1. Signal tests pass
pytest tests/test_core.py -v -k "signal or asset or dc"

# 2. Pipeline produces 32 features
python -c "
from data.pipeline import TransactionPipeline
p = TransactionPipeline(seq_len=100)
print('feature count:', len(p.features))  # 31 (regime added by pipeline)
"

# 3. Cascade forward produces correct shapes
python -c "
import torch
from models.dignity import Dignity
m = Dignity(task='cascade', input_size=32, hidden_size=256)
x = torch.randn(4, 100, 32)
out = m(x)
for k, v in out.items():
    if hasattr(v, 'shape'):
        print(k, v.shape)
    else:
        print(k, [t.shape for t in v])
"

# 4. Full test suite
pytest tests/ -v

# 5. Paper trading smoke test
dignity-train --config config/train_quant_paper.yaml

# 6. ONNX export
python -m export.to_onnx --checkpoint checkpoints/quant/best.pt \
  --output quant.onnx --config config/train_quant_paper.yaml --benchmark
```
