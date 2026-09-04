# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This repo lives on a **macOS/Linux dual-boot machine** and is worked on from both sides, so
absolute paths left behind by tooling often point at the macOS install and are dead under Linux
(e.g. `.remember/MIGRATED-TO.txt` → `/Users/controlroom/...`). Treat a macOS path here as "valid
on the other boot", not as corruption. The macOS volumes are APFS and are **not readable from
Linux** — no APFS driver is installed.

`.venv/` is a **Linux** virtualenv built from `/usr/bin/python3.12` (Python 3.12.3; torch from
the CPU wheel index, currently 2.14.0+cpu). Activate it before anything else:

```bash
source .venv/bin/activate
```

If it is missing or points at a macOS path (the other boot rebuilt it), recreate it:

```bash
/usr/bin/python3.12 -m venv --without-pip .venv && source .venv/bin/activate
python -m ensurepip --upgrade
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e . --no-deps
```

`torch>=2.5` is required (`torch.onnx.export(..., dynamo=False)` in `export/to_onnx.py`);
conda-forge's current pytorch (2.6) also satisfies it if you prefer a conda env.

Verify before claiming any test/train result actually ran.

## Commands

```bash
# Tests (96 collected: test_core 21, test_data 23, test_models 11, test_privacy 34,
# test_operator 4, test_docs 1, test_train 2 — one skips without CUDA)
pytest tests/ -v
pytest tests/test_models.py -v                       # one file
pytest tests/test_models.py::TestDignityModel::test_risk_model -v   # one test
pytest tests/ -m "not slow" -v                       # marker filter

# Lint / format — ruff only, line length 88, enforced by CI and pre-commit
ruff check .
ruff format .
pre-commit run --all-files

# The CI coverage gate — measures only core/privacy.py (pyproject [tool.coverage.run] include)
pytest tests/ --cov --cov-fail-under=100

# Train
dignity-train --config config/train_risk.yaml        # or: python -m train.cli --config ...
python -m train.cli --config config/colab.yaml
# notebooks/train_risk.ipynb is the same run as a notebook (Colab badge inside). ruff lints
# and formats .ipynb files too, so it must stay `ruff check`/`ruff format` clean.

# ONNX export (run the module directly; see "Known breakage" below)
python -m export.to_onnx --checkpoint checkpoints/dignity_risk_best.pt --output dignity_risk.onnx --benchmark
```

`pytest.ini` sets `pythonpath = . run`. Imports across the package are **absolute from the repo
root** (`from core.signals import SignalProcessor`), so anything invoked outside pytest must run
with the repo root on `sys.path` — `train/cli.py` handles this with a `sys.path.insert`, which is
why `E402` is ignored for `data/`, `train/`, `export/`-adjacent trees in `pyproject.toml`.

## Architecture

One data path, one backbone, three interchangeable heads. The pieces only make sense together:

```
data/source/{synthetic,crypto}.py   →  raw DataFrame (volume, price, fee_rate, tx_count[, label])
core/privacy.py                     →  privacy stage: runs once per fit/transform, BEFORE signals
data/pipeline.py TransactionPipeline→  compute_signals → RobustScaler → sliding windows
                                       (process_blocks: per-sequence split for synthetic)
data/loader.py                      →  TransactionDataset / create_dataloader
models/dignity.py Dignity           →  DignityBackbone + one head
train/engine.py                     →  train_epoch / validate_epoch / save_checkpoint
export/to_onnx.py                   →  export + verify + benchmark
```

**`Dignity(task=...)` is the only model entry point.** It composes `DignityBackbone`
(CNN1D → StackedLSTM → AdditiveAttention, in `models/backbone/hybrid.py`) with `RiskHead`,
`ForecastHead`, or `PolicyHead`. Backbone `forward` returns `(context [B,H], attn_weights [B,T])`;
`Dignity.forward` returns `(predictions, attn_weights)`. **Every caller must unpack the tuple** —
this is the single most common source of shape bugs, and `train/engine.py` compensates with a
`predictions.dim() > y.dim()` squeeze.

Adding a task = new head module in `models/head/` + a branch in `Dignity.__init__` + a loss branch
in `train/cli.py`. Nothing else changes.

`core/signals.SignalProcessor` is an **all-static utility class**. `core/privacy.PrivacyManager`
is **instance-based**: it is bound to a `PrivacyBudget` (ε ledger), an optional HMAC key, and
an injectable `rng`. `TransactionPipeline` builds one per instance from `DignityConfig.privacy`.
Every public privacy claim is backed by a named test — see `docs/THREAT-MODEL.md`. If you add
a mechanism, it needs a test in `tests/test_privacy.py` or CI's 100% gate on that file fails.

## Behavior that will surprise you

- **Configured features are silently dropped.** `TransactionPipeline.fit` keeps only
  `[f for f in self.features if f in df.columns]` and stores it as `self.available_features`.
  `compute_signals` only ever derives `volume_volatility`, `volatility`, `momentum`, and
  `directional_change`. So `base.yaml`'s 9-feature list resolves to **7** against synthetic data
  (`price_change` and `regime` are never produced), and `train/cli.py` builds the model with
  `input_size=len(pipeline.available_features)` — **`model.input_size` in YAML is ignored**.
  If a model's input width is unexpected, check `available_features` first.
- **Synthetic data is 1000 independent sequences glued end to end**, all normal blocks
  first, then all anomalous. `train/cli.py` therefore uses `TransactionPipeline.process_blocks`
  (block = `seq_len + 20` rows): shuffle and split at the block level, fit the scaler on train
  blocks only, and compute signals and windows inside each block. Do not go back to
  `process()` + a positional slice — that trained on 100% normal and validated on 100%
  anomalous rows (val loss ≈ 42 in the January checkpoints).
- **AMP and the risk loss.** `RiskHead` ends in a sigmoid and the risk criterion is `BCELoss`,
  which PyTorch refuses inside an autocast region. `train_epoch` therefore computes the loss
  outside autocast in float32. This only bites on CUDA (CPU autocast is a no-op here), so a CPU
  run proves nothing about it; `tests/test_train.py` has a GPU-only test. If you ever switch the
  head to logits, switch the loss to `BCEWithLogitsLoss` in the same change and re-check ONNX
  consumers that expect probabilities.
- **Timestamps are milliseconds everywhere.** `CryptoSource._normalize_timestamp` auto-detects
  seconds (`< 2e10`), ms (`2e10–3e13`), ns (`> 3e13`), and datetime strings, normalizing all to ms
  to match CCXT's native format. Any new data source must emit ms or a join against crypto data
  will silently match zero rows. Do not reintroduce `// 10**9`.

## Known breakage (do not "discover" these again — fix or leave, but don't be confused)

- `setup.py` declares `dignity-export = export.to_onnx:main`, but `export/to_onnx.py` has **no
  `main()`** — only an `if __name__ == "__main__":` block. The console script fails; `python -m
  export.to_onnx` works.
- `pytest.ini` documents that markers (`fast`, `timeout_300`) map to timeouts "via conftest.py
  pytest-timeout hook". **No such hook exists** in `tests/conftest.py` and `pytest-timeout` is not
  in `requirements.txt`. The markers are declared but inert.
- **`docs/` and `README.md` still contain API examples that do not match the code.** Trust the
  source. Verified drift: `docs/ARCHITECTURE.md` shows `from core.signals import
  compute_volatility` (does not exist — use `SignalProcessor.volatility`); `docs/README.md` shows
  `export_dignity_to_onnx` (actual: `export_to_onnx`) and `python -m train.cli --config ...
  --epochs 10` (`--epochs` is not a CLI flag; epochs come from YAML); `README.md` shows
  `generate_dataset(seq_length=...)` (actual kwarg is `seq_len`). Privacy examples were rewritten
  on this branch and are covered by `tests/test_docs.py`. When you touch a documented symbol, fix
  the doc in the same change.

## Conventions

- `ruff format` at line length 88 is the formatter; `ruff check` (`E,W,F,I,N,UP,B,C4,SIM`) is
  the linter. Both run in CI's `lint` job and in pre-commit. Width is settled; `pyproject.toml`
  and `.editorconfig` agree.
- Type hints use modern union syntax (`np.ndarray | None`) — `target-version = "py310"`.
- Config is dataclass-backed: `DignityConfig{model,data,train}` in `core/config.py`, loaded via
  `DignityConfig.from_yaml`. Add a field to the dataclass **and** to `config/base.yaml`; unknown
  YAML keys raise `TypeError` at load.
- A `privacy:` block in YAML is optional and validated at load (`PrivacyConfig`). No block =
  no privacy stage. `bounds` are public parameters chosen a priori — never derive them from data.
- Tests are organized as classes (`TestSignalProcessor`, `TestDignityModel`, …); node ids are
  `file::Class::test_name`. They use root-level fixtures in `tests/conftest.py` (`device`,
  `sample_sequence` `[4,100,9]`, `sample_labels`) and an autouse seed reset. Reuse them rather
  than re-seeding locally.
- `.env` holds API keys (COGNEE, FRED, TWELVEDATA) and is gitignored. `.internal/` is gitignored
  local design notes — read it for background, never cite it as public documentation.
