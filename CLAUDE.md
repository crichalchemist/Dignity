# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Dignity Core is a privacy-preserving deep learning framework for modeling transactional behavior patterns. Refactored from the Sequence research prototype into ~8,000 lines of focused, deployable code. Python 3.10+, PyTorch 2.1+.

## Commands

```bash
# Install
pip install -e .              # editable install
pip install -e ".[dev]"       # with dev dependencies (pytest, ruff)

# Test
pytest tests/ -v              # all 324 tests (6 test files)
pytest tests/test_core.py -v  # core utilities only
pytest tests/test_data.py -v  # data pipeline only
pytest tests/test_models.py -v # model components only
pytest tests/ -m unit         # only unit-marked tests
pytest tests/ -m "not slow"   # skip slow tests
pytest tests/ --cov=. --cov-report=html  # with coverage

# Lint & Format
ruff check .                  # lint
ruff check --fix .            # auto-fix
ruff format .                 # format (double quotes, 100 char lines)

# Train
dignity-train --config config/train_risk.yaml      # risk scoring
dignity-train --config config/train_forecast.yaml   # forecasting
dignity-train --config config/colab.yaml            # Colab-optimized

# Quant cascade training — canonical configs
dignity-train --config config/train_quant_paper.yaml  # paper trading (default, safe)
dignity-train --config config/train_quant.yaml        # live execution (gated — see below)

# Export
python -m export.to_onnx --checkpoint checkpoints/model.pt --output model.onnx --benchmark
```

## Architecture

The system is a composable pipeline — data flows strictly left-to-right:

```
Data Sources → Privacy (core/privacy.py) → Signals (core/signals.py)
  → Pipeline (data/pipeline.py) → Model (models/) → ONNX (export/)
```

### Key design decisions

- **Backbone is a fixed composition**: `DignityBackbone` in `models/backbone/hybrid.py` always composes CNN1D → StackedLSTM → AdditiveAttention. The individual components are separate modules but the assembly order is not configurable — it's intentional, not a limitation.
- **Task heads are swappable**: The `Dignity` model in `models/dignity.py` selects a head based on `task` config (`risk`/`forecast`/`policy`/`cascade`). Adding a new task means adding a new head module and a branch in `Dignity.__init__`.
- **Privacy is a preprocessing concern**: Privacy operations (hashing, anonymization, DP noise) happen before model training, not inside the model. `core/privacy.py` operates on raw data via the `PrivacyManager` class; the model never sees un-sanitized inputs.
- **Signals are computed features**: `core/signals.py` provides `SignalProcessor` with 32 OHLCV-derived features (RSI, MACD, Bollinger, ATR, stochastic, ADX, OBV, VWAP, volatility, momentum, DC state machine, regime). These are computed by `TransactionPipeline` before sequence windowing.
- **Configuration is YAML-declarative**: `DignityConfig` in `core/config.py` nests `ModelConfig`, `DataConfig`, `TrainConfig`, and `ExecutionConfig` dataclasses. All six YAML presets in `config/` are valid starting points.
- **Cascade uses Guided Learning**: The `cascade` task chains Regime → Risk → Alpha → Policy heads. `cascade_loss()` applies auxiliary supervision at each head to prevent vanishing gradients.

### Tensor shapes

Standard input: `[batch, seq_len, features]` = `[B, 100, 32]` by default.
- Backbone outputs: `(context_vector [B, hidden], attention_weights [B, seq_len])` (unidirectional LSTM)
- RiskHead: `(var_estimate [B, 1], position_limit [B, 1])` — dual sigmoid
- ForecastHead: `[B, pred_len, num_features]`
- PolicyHead: `(action_logits [B, n_actions], value [B, 1])`
- Cascade: dict with `regime_probs`, `var_estimate`, `position_limit`, `alpha_score`, `action_logits`, `value`, `attention_weights`

## Code Conventions

- **Functional style preferred**: Pure functions, immutable data, composition over inheritance. The backbone modules use `nn.Module` (PyTorch requires it) but the rest of the codebase avoids classes where possible.
- **Ruff for linting/formatting**: Config in `pyproject.toml`. Line length 100, double quotes, isort-compatible imports. Run `ruff check .` and `ruff format .`.
- **Type hints everywhere**: PEP 484 style. Type hints are the primary function documentation.
- **Minimal docstrings**: Explain "why" not "what". If the type signature is clear, a docstring may not be needed.
- **.editorconfig**: 4-space indent, UTF-8, LF line endings, no final newline. YAML/JSON use 2-space indent.
- **`model.train(False)` not `model.eval()`**: A security hook flags Python's built-in `eval()`. Use `model.train(False)` or the `_set_inference_mode()` helper in `models/dignity.py` to switch to inference mode.

## Testing

- `pytest.ini` sets `pythonpath = . run` — imports work from project root.
- Fixtures in `tests/conftest.py`: `device`, `sample_sequence` ([4,100,32]), `sample_labels` ([4]), and auto-seeding (torch+numpy seed 42).
- Markers: `unit`, `integration`, `slow`, `fast`, `real_api`, `regression`, `performance`. Timeouts are per-marker in conftest, not global.
- Tests are organized by layer: `test_core.py` (signals, privacy, config), `test_data.py` (pipeline, loader, synthetic), `test_models.py` (backbone, heads, full model), `test_train.py` (training loops), `test_export.py` (ONNX), `test_backtest.py` (backtesting).

## Privacy Model

This is a core differentiator, not an afterthought. The `PrivacyManager` class in `core/privacy.py` implements:
- SHA-256 identifier hashing with configurable salt (`hash_identifier()`, `anonymize_addresses()`)
- k-anonymity via amount quantization (`quantize_amounts()`)
- Epsilon-differential privacy via Laplace noise injection (`add_noise()`)
- k-threshold rare event suppression (`suppress_rare_events()`)

The `sanitize_dataset()` classmethod chains quantization, noise, and anonymization. Epsilon and k parameters are the main privacy-utility trade-off knobs.

## Production Path Gating

All development training uses `config/train_quant_paper.yaml` (paper trading, `risk_sdk_enabled: false`).
`config/train_quant.yaml` (live execution, `paper_trading: false`) is gated behind:

1. **Section 4** — backtest gate: ARR ≥ 15%, Sharpe ≥ 1.0, drawdown ≤ 20%, win rate ≥ 52%
2. **Section 5** — 30-day paper soak: daily drawdown ≤ 5% on every day, gate trigger rate within ±5% of backtest

Do not switch to the live config until both gates pass. See `docs/plans/2026-04-13-dignity-production-path.md`.

## Monthly Re-training Cadence

Once live, re-train monthly on the last 6 months of live + paper bar data:

```bash
# Collect latest bars via MetaApi into data/
dignity-train --config config/train_quant_paper.yaml  # validate on paper first

# If paper soak gate re-passes (scripts/go_live.sh delegates to backtest.go_live_check):
dignity-train --config config/train_quant.yaml
```

Trigger conditions (whichever comes first):
- Calendar month boundary
- 7-day Sharpe (per `scripts/monitor.sh`) drops below 0.5 for 3 consecutive days
- Regime distribution shifts by > 15% from the training distribution

Keep the last 3 checkpoints in `checkpoints/quant/` before overwriting.
