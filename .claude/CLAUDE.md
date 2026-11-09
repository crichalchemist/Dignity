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
pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps
```

`torch>=2.5` is required (`torch.onnx.export(..., dynamo=False)` in `export/to_onnx.py`);
conda-forge's current pytorch (2.6) also satisfies it if you prefer a conda env.

**On the macOS boot** the Linux `.venv/` is unusable. Use the existing conda env instead,
prefixing every command with `conda run -n dignity-model` (torch 2.10, Python 3.11). Verified
2026-11-08: 100 passed, 1 skipped (CUDA), and `pre-commit run --all-files` is clean.
`requirements-dev.txt` must also be installed into that env:
`conda run -n dignity-model pip install -r requirements-dev.txt`.

Verify before claiming any test/train result actually ran.

On the macOS checkout, `checkpoints/risk/` holds untracked **January** checkpoints
`dignity_risk_{best,epochN}.pt` that cannot be regenerated; they use the old checkpoint
format that export no longer supports (DGNTY-3, DGNTY-35). No shipped config writes there,
`tests/test_core.py` enforces it, and anything you train goes to `runs/`.

## Commands

```bash
# Tests (235 collected: test_core 41, test_data 68, test_models 14, test_privacy 34,
# test_operator 4, test_docs 4, test_train 25, test_checkpoint 3, test_export 7,
# test_calibration 3, test_labels 7, test_gates 11, test_ci 2, test_eval 12 — one skips without CUDA)
# every test is limited to 300 s (pytest.ini; pytest-timeout)
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

python -m eval.gates --seed 1000   # data gate; tuning seed only, refuses 2026

# Train
# checkpoints land in ./runs/<task>/ (gitignored); never in checkpoints/
dignity-train --config config/train_risk.yaml        # or: python -m train.cli --config ...
python -m train.cli --config config/colab.yaml
# notebooks/train_risk.ipynb is the same run as a notebook (Colab badge inside). ruff lints
# and formats .ipynb files too, so it must stay `ruff check`/`ruff format` clean.

# ONNX export
dignity-export --checkpoint runs/risk/best.pt --output dignity_risk.onnx --benchmark  # or python -m export.to_onnx
```

`pytest.ini` sets `pythonpath = . run`. Imports across the package are **absolute from the repo
root** (`from core.signals import SignalProcessor`), so anything invoked outside pytest must run
with the repo root on `sys.path` — `train/cli.py` handles this with a `sys.path.insert`, which is
why `E402` is ignored for `data/`, `train/`, `export/`-adjacent trees in `pyproject.toml`.

## Architecture

One data path, one backbone, three interchangeable heads. The pieces only make sense together:

```
data/source/synthetic.py            →  event-record frame: raw columns + block_id, is_anomalous,
                                       anomaly_type, anomaly_id, block_label
data/source/crypto.py               →  raw DataFrame (volume, price[, fee_rate, tx_count]), no labels
core/privacy.py                     →  privacy stage: runs once per fit/transform, BEFORE signals
data/pipeline.py TransactionPipeline→  compute_signals → RobustScaler → sliding windows
                                       (process_blocks: block split 70/15/15 → SplitData)
data/labels.py                      →  risk/forecast labels from the event record
data/loader.py                      →  TransactionDataset (dict batches) / create_dataloader
models/dignity.py Dignity           →  DignityBackbone + one head
train/engine.py                     →  train_epoch / predict
train/loop.py fit                  →  shared loop (cli, notebook, eval.run); best.pt on val PR-AUC
train/checkpoint.py                 →  self-describing checkpoints (FORMAT_VERSION 2) / build_model_from_meta
train/losses.py                     →  risk_loss
export/to_onnx.py                   →  export + verify + benchmark
```

**`Dignity(task=...)` is the only model entry point.** It composes `DignityBackbone`
(CNN1D → StackedLSTM → AdditiveAttention, in `models/backbone/hybrid.py`) with `RiskHead`,
`ForecastHead`, or `PolicyHead`. Backbone `forward` returns `(context [B,H], attn_weights [B,T])`;
`Dignity.forward` returns `(predictions, attn_weights)`. **Every caller must unpack the tuple** —
this is the single most common source of shape bugs, and `train/losses.py::risk_loss` reshapes predictions
and labels to `[B]`.

Adding a task = new head module in `models/head/` + a branch in `Dignity.__init__` + a loss in
`train/losses.py` and its branch in `train/loop.py::fit`. Nothing else changes.

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
  (`price_change` and `regime` are never produced), and `train/loop.py::fit` sizes the model from
  `SplitData.feature_names` (the pipeline's `available_features`) — **`model.input_size`
  in YAML is ignored**.
  If a model's input width is unexpected, check `available_features` first.
  The checkpoint stores `available_features` in order (`meta.feature_names`).
- **Synthetic data is an event record of blocks, interleaved.** `SyntheticGenerator` writes one
  row per step with `block_id`, `is_anomalous`, `anomaly_type`, `anomaly_id` and `block_label`;
  normal and anomalous blocks are drawn independently, so they interleave. Labels come from the
  per-row event record, never from a single row. Risk = overlap rule: a window is positive when its
  anomalous fraction exceeds `data.label_overlap`, never "label of the last row". `create_sequences`
  rejects 2-D targets; forecast targets come from `data.labels.forecast_labels`.
  `train/loop.py::prepare_data` uses `TransactionPipeline.process_blocks`: shuffle and split blocks 70/15/15,
  fit the scaler on train blocks only, and compute signals and windows inside each block. Do not
  go back to `process()` + a positional slice — that trained on 100% normal and validated on 100%
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

- **Only `task: risk` trains end to end.** This contradicts "Adding a task ... Nothing else
  changes" above. The engine also needs work:
  - `train/cli.py` raises `NotImplementedError` for `forecast` and `policy` until M4.
  - `train/loop.py::fit` never passes head kwargs (`pred_len`, `n_actions`) from config.
- **`docs/` and `README.md` drift from the code easily.** `tests/test_docs.py` guards removed
  claims and links to missing files, not stale examples. Trust the source, and when you touch a
  documented symbol, fix the doc in the same change.

## Conventions

- `ruff format` at line length 88 is the formatter; `ruff check` (`E,W,F,I,N,UP,B,C4,SIM`) is
  the linter. Both run in CI's `lint` job and in pre-commit. Width is settled; `pyproject.toml`
  and `.editorconfig` agree.
- Type hints use modern union syntax (`np.ndarray | None`) — `target-version = "py310"`.
- Config is dataclass-backed: `DignityConfig{model,data,train}` in `core/config.py`, loaded via
  `DignityConfig.from_yaml`. Add a field to the dataclass **and** to `config/base.yaml`; unknown
  keys, top-level or nested, raise `TypeError` at load. `data.source` accepts only `synthetic`.
- A `privacy:` block in YAML is optional and validated at load (`PrivacyConfig`). No block =
  no privacy stage. `bounds` are public parameters chosen a priori — never derive them from data.
- Tests are organized as classes (`TestSignalProcessor`, `TestDignityModel`, …); node ids are
  `file::Class::test_name`. They use root-level fixtures in `tests/conftest.py` (`device`,
  `sample_sequence` `[4,100,9]`, `sample_labels`) and an autouse seed reset. Reuse them rather
  than re-seeding locally.
- Branches: both `main` and `dev` are protected (required checks `lint`, `test (3.10)`,
  `test (3.12)`; no deletion, no force-push); `dev` is the integration branch. Branch feature
  work from `dev` and open PRs against `dev`; `dev` → `main` is the release step. CI runs on
  pushes to both and on every PR. PR titles carry `DGNTY-n`; merged branches are deleted
  automatically.
- `.env` holds API keys (COGNEE, FRED, TWELVEDATA) and is gitignored. `.internal/` is gitignored
  local design notes — read it for background, never cite it as public documentation.
