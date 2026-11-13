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
2026-11-11: 1347 passed, 1 skipped (CUDA), and `pre-commit run --all-files` is clean.
`requirements-dev.txt` must also be installed into that env:
`conda run -n dignity-model pip install -r requirements-dev.txt`.

Verify before claiming any test/train result actually ran.

On the macOS checkout, `checkpoints/risk/` holds untracked **January** checkpoints
`dignity_risk_{best,epochN}.pt` that cannot be regenerated; they use the old checkpoint
format that export no longer supports (DGNTY-3, DGNTY-35). No shipped config writes there,
`tests/test_core.py` enforces it, and anything you train goes to `runs/`.

## Commands

```bash
# Tests (1582 collected: test_core 87, test_data 169, test_models 23, test_privacy 42,
# test_operator 4, test_docs 769, test_train 106, test_checkpoint 4, test_export 9,
# test_calibration 3, test_labels 7, test_gates 74, test_ci 4, test_eval 120,
# test_colab_eval 13, test_scripts 19, test_release 129 — one skips without CUDA)
# every test is limited to 300 s (pytest.ini; pytest-timeout), except the README quickstart
# test, which carries its own 900 s limit (three subprocesses, each capped at 300 s)
pytest tests/ -v
pytest tests/test_models.py -v                       # one file
pytest tests/test_docs.py -v   # doc numbers, symbols, paths and the README quickstart (M6)
pytest tests/test_models.py::TestDignityModel::test_risk_model -v   # one test
pytest tests/ -m "not slow" -v                       # marker filter
pytest tests/test_docs.py -m "not slow" -v         # fast doc run: skips only the README quickstart test (a name filter on Quickstart is case-insensitive and also deselects four README tests)

# Lint / format — ruff only, line length 88, enforced by CI and pre-commit
ruff check .
ruff format .
pre-commit run --all-files

# The CI coverage gate — measures only core/privacy.py (pyproject [tool.coverage.run] include)
pytest tests/ --cov --cov-fail-under=100

python -m eval.gates --out results/v1.0/data_gate.json   # ten-seed data gate on the tuning seeds 1000-1009 (refuses everything else); the committed result passes every check at hard build-up strength 0.9 (presets 0.9/0.95/1.0, 3600 blocks each), the smallest value of `STRENGTH_GRID` (0.6-0.9) that passed the headroom check on the eight selection seeds and then the ten-seed gate; the headroom ratio and wins use the risk-aware forecast probe; search records in results/strength-search/; --seeds, --n-blocks and --strength are for rehearsals
python scripts/refresh_doc_numbers.py README.md docs/RESULTS.md   # after any results change
python scripts/verify_release.py   # v1.0 success criteria 2-5 on results/v1.0 (exit 1 = a criterion failed or a training raise was recorded, 2 = unusable results, including results not on the evaluation seed); results/v1.0 holds the first evaluation, which the restated criteria call unusable until it is replaced
python scripts/verify_release.py --results DIR --seed tuning   # rehearsal on one tuning-seed run (seed read from DIR/risk.json, 1000-1009); prints a banner, not the evaluation
python -m eval.run --smoke --seed tuning  # CI-sized, all three tasks (--tasks defaults to risk,forecast,policy); outputs under runs/smoke/
python -m eval.run --smoke --seed tuning --data-seed 1003  # any tuning seed (1000-1009), own run directory eval_tuning_<N>
python -m eval.run --smoke --seed tuning --tasks risk,forecast  # --tasks forecast also fits risk (baseline) but writes only forecast.json; unknown names exit 2
python -m eval.run --smoke --seed tuning --privacy-sweep  # also trains risk per epsilon on the config/train_risk.yaml privacy template (own models, so works without --tasks risk); writes privacy.json + privacy_curve.png
python -m eval.run --config config/synthetic_hard.yaml --seed eval --tasks risk,forecast,policy --privacy-sweep  # full run on a Colab GPU; see docs/RESULTS.md

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

Longer-form architecture: docs/ARCHITECTURE.md (data contract, heads, pipeline)

One data path, one backbone, three interchangeable heads. The pieces only make sense together:

```
data/source/synthetic.py            →  event-record frame: raw columns + block_id, is_anomalous,
                                       anomaly_type, anomaly_id, block_label, build_up
data/source/crypto.py               →  raw DataFrame (volume, price[, fee_rate, tx_count]), no labels
core/privacy.py                     →  privacy stage: one release per block split (train/val/test), BEFORE signals
data/pipeline.py TransactionPipeline→  compute_signals → RobustScaler → sliding windows
                                       (process_blocks: block split 45/15/40 → SplitData)
data/labels.py                      →  risk/forecast labels from the event record
data/loader.py                      →  TransactionDataset (dict batches) / create_dataloader
models/dignity.py Dignity           →  DignityBackbone + one head
train/engine.py                     →  train_epoch / predict
train/loop.py fit                  →  shared loop (cli, notebook, eval.run); best.pt on val PR-AUC (policy: val neg_cost under the review cap; forecast: val lift_upper above the eligibility floor)
train/checkpoint.py                 →  self-describing checkpoints (FORMAT_VERSION 2) / build_model_from_meta
train/losses.py                     →  risk_loss, forecast_loss (masked per-horizon BCE),
                                       expected_cost_loss (policy: softmax-weighted cost table)
eval/{metrics,baselines,plots,run,privacy_sweep}.py → risk.json, forecast.json, policy.json (+ privacy.json with --privacy-sweep) + charts; CIs bootstrap blocks, not windows
export/to_onnx.py                   →  export + verify + benchmark
```

**`Dignity(task=...)` is the only model entry point.** It composes `DignityBackbone`
(CNN1D → StackedLSTM → AdditiveAttention, in `models/backbone/hybrid.py`) with `RiskHead`,
`ForecastHead`, or `PolicyHead`. Backbone `forward` returns `(context [B,H], attn_weights [B,T])`;
`Dignity.forward` returns `(predictions, attn_weights)`. **Every caller must unpack the tuple** —
this is the single most common source of shape bugs, and `train/losses.py::risk_loss` reshapes predictions
and labels to `[B]`; `forecast_loss` keeps `[B, H]` and masks windows with an active anomaly.

Adding a task = a head in `models/head/`, a branch in `Dignity.__init__`, a loss in
`train/losses.py`, target/selection/head_kwargs branches in `train/loop.fit`, metrics in
`eval/metrics.py`, a baseline in `eval/baselines.py`, and a branch in `eval/run.py`.

`core/signals.SignalProcessor` is a **stateless utility class** (static functions; `process_sequence` is a classmethod). `core/privacy.PrivacyManager`
is **instance-based**: it is bound to a `PrivacyBudget` (ε ledger), an optional HMAC key, and
an injectable `rng`. `TransactionPipeline.process_blocks` builds one
`PrivacyManager`+`PrivacyBudget` per split (parallel composition); the instance-level manager
serves only `fit`/`transform`/`process`.
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
  row per step with `block_id`, `is_anomalous`, `anomaly_type`, `anomaly_id`, `block_label` and `build_up`;
  normal and anomalous blocks are drawn independently, so they interleave. Labels come from the
  per-row event record, never from a single row. Risk = overlap rule: a window is positive when its
  anomalous fraction exceeds `data.label_overlap`, never "label of the last row". `create_sequences`
  rejects 2-D targets; forecast targets come from `data.labels.forecast_labels`.
  `precursor_strength > 0` overlays build-ups on four anomaly types, from a child random
  stream; `decoy_rate > 0` adds decoys (build-ups nothing follows) from a second one, so it
  never moves a true build-up; strength 0 reproduces the old frame. `build_up` is for tests
  and analysis: `data.pipeline.EVENT_LABEL_COLUMNS` refuses it and the other label columns
  as features.
  `train/loop.py::prepare_data` uses `TransactionPipeline.process_blocks`: shuffle and split blocks 45/15/40,
  fit the scaler on train blocks only, and compute signals and windows inside each block.
  `prepare_data(config, privacy_seed=None)` seeds only the privacy noise (default: `synthetic.seed`);
  the data and the block split always follow `synthetic.seed`.
  With a `privacy:` block each split is its own release on its own budget (generalize classes
  fitted on train), and the pipeline releases once: a later `transform` or `process_blocks` raises
  `BudgetExhausted`. Do not
  go back to `process()` + a positional slice — that trained on 100% normal and validated on 100%
  anomalous rows (val loss ≈ 42 in the January checkpoints).
- **AMP and the risk loss.** `RiskHead` ends in a sigmoid and the risk criterion is binary cross-entropy on probabilities
  (`F.binary_cross_entropy`), which PyTorch refuses inside an autocast region. `train_epoch` therefore computes the loss
  outside autocast in float32. This only bites on CUDA (CPU autocast is a no-op here), so a CPU
  run proves nothing about it; `tests/test_train.py` has a GPU-only test. If you ever switch the
  head to logits, switch the loss to `BCEWithLogitsLoss` in the same change and re-check ONNX
  consumers that expect probabilities.
  `PolicyHead` returns logits; `expected_cost_loss` takes the softmax in float32 outside autocast.
- **The policy warm-starts from a risk checkpoint.** Trained cold, the spec expected-cost loss
  stalls at "review everything". `config/train_policy.yaml` sets `train.init_backbone_from:
  ./runs/risk/best.pt`, so train risk first; `train/loop.load_backbone` copies only the
  `backbone.*` tensors (feature names, scaler center/scale, privacy block and shapes must match, else `ValueError` (under privacy the scaler tolerates 25% of scale: system-mode noise is unseeded; seeded `deterministic_rng` noise would not need it, but the tolerance is not tightened); the checkpoint must also share the run's `data:` block (same split); the two shipped configs share one `privacy:` block, keep them equal) and `fit` reads
  the key only for `task: policy` and `task: forecast`. Policy selects `best.pt` on val `neg_cost` under the review
  cap and ignores `select_metric`, `pos_weight` and the chance guard; resuming across metrics
  raises `ValueError`.
- **Forecast is early warning, and `eval.run` fits risk for it.** Forecast windows with an
  anomaly active at their last row are masked out of the loss, the metrics and lead time
  (`forecast_mask`). `--tasks forecast` and `--tasks policy` also fit risk once per training
  seed (their baseline is today's risk score, and each forecast and policy seed warm-starts its
  backbone from the same seed's risk `best.pt`; a forecast epoch is kept only when its val
  `lift_upper` beats the same seed's risk score, else `fit` raises `ForecastNotEligibleError`
  (`train.forecast_eligibility: false` only in `config/smoke.yaml`), which `eval.run` records
  as `metrics.raised`, prints as a `TRAINING_RAISED` line and exits 1 on), but write `risk.json`
  only when `risk` is in `--tasks`. Each task trains under its own `eval_<seed>/<task>/train_seed_<s>/` directory.
- **Timestamps are milliseconds everywhere.** `CryptoSource._normalize_timestamp` auto-detects
  seconds (`< 2e10`), ms (`2e10–3e13`), ns (`> 3e13`), and datetime strings, normalizing all to ms
  to match CCXT's native format. Any new data source must emit ms or a join against crypto data
  will silently match zero rows. Do not reintroduce `// 10**9`.

## Conventions

- `ruff format` at line length 88 is the formatter; `ruff check` (`E,W,F,I,N,UP,B,C4,SIM`) is
  the linter. Both run in CI's `lint` job and in pre-commit. Width is settled; `pyproject.toml`
  and `.editorconfig` agree.
- Type hints use modern union syntax (`np.ndarray | None`) — `target-version = "py310"`.
- Config is dataclass-backed: `DignityConfig{model,data,train,eval,policy,privacy,device,seed}` in `core/config.py`, loaded via
  `DignityConfig.from_yaml`. Add a field to the dataclass **and** to `config/base.yaml`; unknown
  keys, top-level or nested, raise `TypeError` at load. `data.source` accepts only `synthetic`.
  `config/base.yaml` is the config reference (there is no CONFIGURATION.md): every key carries
  a comment saying what it does, `tests/test_docs.py::TestBaseConfigDoc` enforces a comment of at
  least three words per key and pins the load-bearing ones to the code, and you
  keep those comments current when you add a field.
- A `privacy:` block in YAML is optional and validated at load (`PrivacyConfig`). No block =
  no privacy stage. `bounds` are public parameters chosen a priori — never derive them from data.
- `privacy.deterministic_rng: true` (+ `privacy_seed=`) is the seeded experiment mode; shipped
  configs stay on SystemRandom.
- Framing guard (`tests/test_docs.py`, `framing_hits`): a tripwire over a fixed trading/price-forecasting vocabulary in every public doc and `config/*.yaml` comment, not a proof; allowed forms (`trade-off`, `wash_trading`, `pump_and_dump`, `layering`) are listed beside it and the vocabulary is not extended.
- Doc numbers are machine-checked in **every** public doc (`README.md` and `docs/**/*.md`;
  `tests/test_docs.py`, `check_numbers`). A quoted number is a token directly followed by a marker,
  `0.84 <!--num risk.json metrics.pr_auc.value .2f-->`, grammar
  `<!--num <file>.json <dotted.path> <fmt>-->`, resolved against `results/v1.0/<file>.json`
  (dicts by key, lists by non-negative integer: `pr_auc_by_horizon.4.value`; `fmt` is a Python
  format spec: `.2f`, `.1%`, `d`, `g` for ε (floats such as `10.0`), `s` compares the whole token;
  a JSON null renders as `n/a` for every `fmt`; a path through a null or a scalar is "no such
  path"). Only the number core of the token is compared. The marker grammar (`NUM_RE`,
  `TOKEN_RE`, `split_token`, `resolve`, `strip_code`) is defined once in `scripts/refresh_doc_numbers.py`;
  `tests/test_docs.py` imports it and `check_numbers` there is the checker. `split_token` splits a
  token into prefix, core and suffix, so a refresh rewrites the core and keeps the wrappers. Accepted wrappers: `[0.72 <!--…-->, 0.88 <!--…-->]` and
  `(0.72, 0.88)` for a CI (three markers), `**0.81**`, `_`/`~`, table pipes glued to the number,
  a `/` glued before or after the number (`0 <!--…-->/34 <!--…-->`) and a trailing `. , ; :`. A range such as `0.72–0.88`
  is rejected: write the bracket form. A `<!--num` comment that does not parse is a malformed
  marker and fails. Text between `<!--results-start-->` and `<!--results-end-->` may hold no
  number without a marker; the tags must alternate, start first (swapped or nested tags fail),
  and a backticked purely numeric literal (`` `0.99` ``, `` `-0.5` ``, `` `12%` ``) in a zone is an
  unmarked number. Labels that merely contain digits stay backticked: keys and paths
  (`recall_at_1pct_fpr`, `policy.review_budget`, `results/v1.0/risk.json`); an ε label carries
  its own marker (`metrics.points.N.epsilon`, `g`); a horizon label is a backticked
  `pr_auc_by_horizon.N` (forecast.json has no list of horizon values); "95% CI" lives in prose
  outside the zone. A backticked numeric literal such as the evaluation seed `2026` is an unmarked
  number inside a zone, so it must sit OUTSIDE results zones. Markers and zone tags inside a fenced
  block (``` or ~~~) or an inline code span are documentation: neither checked nor counted. A
  comment that looks like a zone tag or the provisional tag but is not written exactly
  (`<!-- results-start -->`) is reported as a malformed zone tag. Banner: while the four `results/v1.0/*.json`
  files (one run: same seed, git_sha and config_hash, asserted) carry the tuning seed, README and
  `docs/RESULTS.md` each contain the line (a substring check)
  `<!--provisional-->**Provisional: numbers are from the tuning seed; the final evaluation has
  not run.**` (one line); once `risk.json` has the evaluation seed the word Provisional and the
  `<!--provisional-->` comment appear in neither.
- The refresh command covers both `README.md` and `docs/RESULTS.md`; after it, rewrite the sentences the doc tests point at.
- Doc references are checked the same way, on inline code only (fenced blocks are skipped).
  Symbol: `^(core|data|models|train|export|eval)(\.\w+)+$` must import and resolve (longest
  importable module prefix, then `getattr`); under `data`, `data.synthetic`, `train` and `eval`
  a name may instead be a config dataclass field (`data.synthetic.block_len`, `eval.epsilons`).
  Path: a span with `/` or ending `.py .yaml .md .json .png .ipynb`, and no whitespace, `* { } < >`
  or `scheme://`, must exist relative to the repo root after stripping a `::` or `#` suffix and a
  leading `./`; an absolute path or one that climbs out of the repo (`/etc/hosts`, `../x`) is
  reported as missing even when it exists; only `runs/` and `checkpoints/` (gitignored) may be absent. Node: a non-test
  `path::name` needs a `.py` repo-relative path and a top-level def or class `name`; a method is
  `path::Class.method` or `path::Class::method`, and both parts are looked up by AST.
  `tests/…::Class::test` citations are checked separately and a bare `Test*` name must carry its
  file. A fenced block opened as `bash quickstart` (exactly) is run by the suite under `tmp_path`
  on CPU: every command must start with `python` and may only write `runs/…`, `dignity_risk.onnx` and the
  config's `eval.out_dir` and `img_dir`, which the test redirects under `tmp_path`
  (`config/smoke.yaml` is an input the test rewrites, not a write target). The checkpoint, the ONNX
  file and the results JSON files must exist afterwards, and `stray_writes` compares git status
  and the size and mtime of every file under `results/`, `docs/img/`, `runs/` and `checkpoints/`
  and of any root `*.onnx` before and after. The quickstart test carries its own 900 s limit.
  When you touch a documented symbol, fix the doc in the same change; the doc tests enforce it.
- Tests are organized as classes (`TestSignalProcessor`, `TestDignityModel`, …); node ids are
  `file::Class::test_name`. They use root-level fixtures in `tests/conftest.py` (`device`,
  `sample_sequence` `[4,100,9]`, `sample_labels`) and an autouse seed reset. Reuse them rather
  than re-seeding locally.
- Branches: both `main` and `dev` are protected (required checks `lint`, `test (3.10)`,
  `test (3.12)`; no deletion, no force-push); `dev` is the integration branch. Branch feature
  work from `dev` and open PRs against `dev`; `dev` → `main` is the release step. CI runs on
  pushes to both and on every PR; the `smoke` job (installed console scripts, all three tasks)
  is also a required check (on `dev` and `main` since 2026-11-09). PR titles carry `DGNTY-n`; merged branches are deleted
  automatically.
- `.env` holds API keys (COGNEE, FRED, TWELVEDATA) and is gitignored. `.internal/` is gitignored
  local design notes — read it for background, never cite it as public documentation.
