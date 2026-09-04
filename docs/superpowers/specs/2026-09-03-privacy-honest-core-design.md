# Privacy Honest Core — Design Spec

**Date:** 2026-09-03
**Status:** Approved in brainstorming; awaiting implementation plan
**Branch:** `privacy-honest-core`
**Scope:** `core/privacy.py`, `core/config.py`, `data/pipeline.py`, `train/cli.py`, `config/*.yaml`, `tests/`, `docs/`, `README.md`, `.github/`, `.pre-commit-config.yaml`, `pyproject.toml`, `.editorconfig`, `.gitignore`, `.claude/CLAUDE.md`

## 0. Summary

Dignity brands itself a privacy-preserving framework, but `core/privacy.py` has no
callers outside tests and several of its guarantees are false as written. This work
makes every public privacy claim true or removes it, wires the primitives into the
data pipeline as a real stage, adds an operator-side threat model with enforcing
tests, and puts the whole thing under CI so the claims stay true on every push.

Two privacy layers, kept separate and named:

- **Subject layer** — protects individuals in the training data: input-level ε-DP
  (Laplace, bounded, budget-ledgered) and k-anonymity (quantile generalization).
- **Operator layer** — protects the person running the model from an outside observer:
  local-only inference, self-contained artifact, no telemetry — each enforced by a test.

## 1. Context

Three documents in the repo describe three different projects. `.internal/trainingplan.md`
models pseudonymous merchant operators (instrument, geohint, user-agent hash).
`docs/training-curriculum.md` is a forex foundation-model plan from the reverted April
direction. `docs/PRIVACY.md` documents a functional API (`hash_identifiers(df, ...)`,
`anonymize_amounts`) and a "Secure Aggregation" capability, none of which exist.

The code implements a fourth thing: a compact CNN-LSTM-attention sequence model with
three task heads and a privacy module wired to nothing. `TransactionPipeline`,
`train/cli.py`, `train/engine.py` and `export/to_onnx.py` never touch `PrivacyManager`.
Data flows source → signals → scaler → model unsanitized.

The repo is public. Unbacked privacy claims in a public repo are the problem.

## 2. Decisions taken

| Question | Decision |
|---|---|
| Whose privacy, from whom | Both layers: data subject vs. artifact holder; operator vs. outside observer |
| Rigor bar | Open-source library others adopt **and** portfolio piece: every claim defensible, scope cut hard |
| Approach | A — honest core first, then wire it in. DP-SGD (Opacus) is an explicit later phase, gated on this work |
| k-anonymity | Make it true (quantile bins + merge under k), not drop the claim |
| `suppress_rare_events` | Delete — zero callers, zero tests |
| `sanitize_dataset` | Delete — returned un-noised data beside noised; composition moves to the pipeline stage |
| "Built for deniability" | Keep, defined in `THREAT-MODEL.md` as three enforced properties |
| Formatter width | 88 — matches the tree; `pyproject.toml` and `.editorconfig` change to agree |
| GPU | Out of scope. Card is AMD Polaris (gfx803); ROCm support is unofficial. Separate spike if wanted |

## 3. Claims contract

After this work the public surface (README, `docs/`, docstrings) may claim exactly the
following and nothing more.

| Claim kept | Precise wording | Stopped saying |
|---|---|---|
| Differential privacy | **Input-level** ε-DP: Laplace noise on selected numeric features before training, ε **accounted per pipeline instance** by a ledger that refuses to overspend. Floating-point implementation. | "differentially private model"; any DP claim about trained weights |
| Identifier protection | **Keyed pseudonymization** (HMAC-SHA256, key required at call time). Same key → linkable; different key → unlinkable. | "anonymization" |
| Amount protection | **k-anonymity for the generalized column as released** via quantile binning with under-k bins merged. | k claims about features derived downstream |
| Operator privacy | **Local-only inference**: no network I/O in inference/export path; ONNX self-contained; no telemetry. Enforced by tests. | "deniability" as a bare slogan |
| Removed | — | "Secure Aggregation"; functional `hash_identifiers(df, ...)` API |

**Rule:** a claim appears in docs only if a test in `tests/` fails when the property breaks.
No test, no claim. `tests/test_docs.py` enforces the "removed" column mechanically.

## 4. Purge

Mechanical, first, needs none of the new code.

- **Delete** `docs/training-curriculum.md`.
- **Delete** `tests/__init__.py::get_api_url` (dead `localhost:8000` helper). Leave the file as an empty package marker.
- **Delete** `PrivacyManager.suppress_rare_events` and `PrivacyManager.sanitize_dataset`.
- **Rewrite** `docs/PRIVACY.md` around the contract table. Drop Secure Aggregation.
- **Fix** `docs/ARCHITECTURE.md` privacy snippet to the real API.
- **Fix** README privacy examples (`PrivacyManager(hash_salt=...)`, `quantize_amounts(precision=...)`), the "Privacy Operations" section, and "Design Philosophy" wording.
- **Leave** README non-privacy drift (`generate_dataset(seq_length=)`, badge counts other than the tests badge) — already logged in CLAUDE.md for a separate pass.
- **Leave** `.internal/trainingplan.md` — gitignored working notes, not a public claim.

## 5. Privacy primitives — `core/privacy.py`

Every signature changes, so names change too. In a privacy module the names are claims.

```python
class BudgetExhausted(ValueError): ...

class PrivacyBudget:
    def __init__(self, epsilon_total: float) -> None
    def spend(self, epsilon: float) -> None          # raises BudgetExhausted if spent + epsilon > total
    @property spent -> float
    @property remaining -> float

class PrivacyManager:
    def __init__(self, budget: PrivacyBudget, key: bytes | str | None = None, rng=None) -> None
    #   key: optional at construction; if given, str is UTF-8 encoded; < 16 bytes → ValueError
    #   rng: any object with .random() -> float in [0, 1). Default: secrets.SystemRandom()

    def pseudonymize(self, identifier: str) -> str
    #   HMAC-SHA256(key, identifier.encode()).hexdigest(). key is None → ValueError.
    def pseudonymize_many(self, identifiers: list[str]) -> list[str]

    def add_laplace_noise(self, values: np.ndarray, *, epsilon: float, bounds: tuple[float, float]) -> np.ndarray
    #   epsilon <= 0 → ValueError; bounds[0] >= bounds[1] → ValueError
    #   clip to bounds; sensitivity = hi - lo; scale = sensitivity / epsilon
    #   budget.spend(epsilon) BEFORE sampling (fail before any release)
    #   noise via inverse CDF: u = rng.random() - 0.5; x = -scale * sign(u) * ln(1 - 2|u|)
    #   returns a new array; input untouched

    @staticmethod
    def generalize_amounts(values: np.ndarray, *, bins: int = 10, k: int = 5) -> np.ndarray
    #   len(values) < k → ValueError
    #   edges = quantile(values, linspace(0, 1, bins + 1)); collapse duplicate edges
    #   assign bins; while any bin has < k members, merge it into its smaller neighbour
    #   return each record's bin midpoint; len(output) == len(values)
```

`core/__init__.py` exports `PrivacyManager`, `PrivacyBudget`, `BudgetExhausted`.

### Rationale

- **HMAC with a key** removes the unsalted default, the `f"{salt}{identifier}"` split
  ambiguity (`"ab"+"cd"` == `"a"+"bcd"`), and the fast-hash brute-force surface for
  low-entropy identifiers. Key optional at construction so the pipeline can build a
  manager for noise/generalize without one; required at `pseudonymize` call time so it
  can never silently run unkeyed.
- **`bounds` instead of `sensitivity`.** Per-value noise is ε-DP only for a bounded
  value. Taking bounds and deriving sensitivity as the range removes the footgun where
  `sensitivity=1.0` silently voided the guarantee for any feature outside [0, 1].
  Semantics: each released value is ε-local-DP for that feature.
- **Ledger** implements sequential composition. ε adds across calls; exceeding
  `epsilon_total` raises rather than degrading. Pure ε, no δ — what we can prove.
- **CSPRNG-backed noise.** `np.random.laplace` is PCG64, and `conftest.py` seeds it to 42.
  A recoverable noise stream defeats DP. Default RNG is `secrets.SystemRandom`;
  injectable so tests use `random.Random(seed)` without weakening production.
- **Real k.** Quantile edges start balanced; merging (not suppressing) under-k bins keeps
  every record and guarantees every equivalence class has ≥ k members when n ≥ k.

### Named limitation

Floating-point Laplace is vulnerable to Mironov (2012). The snapping mechanism is out of
scope. Stated in `THREAT-MODEL.md`; the DP claim is worded "floating-point Laplace."

## 6. Pipeline wiring and config

### Order

```
source → [privacy stage] → compute_signals → select features → scaler → windows
```

The stage runs on **raw columns, before `compute_signals`**. By the DP post-processing
theorem, every signal derived from a noised column inherits its ε. Reversing the order
would leak raw values through un-noised derived features.

### `TransactionPipeline`

- `__init__(..., privacy: PrivacyConfig | None = None, privacy_rng=None)`. When `privacy`
  is present, constructs one `PrivacyBudget(epsilon_total)` and one `PrivacyManager` for
  the instance's lifetime, exposed as `self.privacy_manager`. `privacy_rng` is forwarded
  to `PrivacyManager(rng=...)` — a test seam only; production leaves it `None`.
- `_apply_privacy(df) -> df`. For each configured column applies exactly one mechanism —
  `laplace` or `generalize` — in place. No columns added or removed; `input_size` unaffected.
- **The stage runs exactly once per public call.** `fit`, `transform`, and `fit_transform`
  each call `_apply_privacy` once, then delegate to private `_fit_on(df)` / `_transform_on(df)`
  that assume privacy is already applied. In particular `fit_transform` is
  `p = _apply_privacy(df); _fit_on(p); return _transform_on(p)` — one spend, and the scaler
  is fit on the same noise draw it transforms. The current `fit(df).transform(df)` chaining
  would double-spend and fit on a different draw; it goes.
- Budget spans the instance. Each public call is a new release and spends again.
  `BudgetExhausted` propagates.

### `train/cli.py`

After `pipeline.process(...)`, if privacy is configured, print
`Privacy: ε spent {spent:.3f} of {total:.3f}`.

### `PrivacyConfig` in `core/config.py`

```yaml
privacy:
  key_env: DIGNITY_PRIVACY_KEY       # env var NAME. The key never appears in YAML.
  epsilon_total: 1.0
  k: 5
  features:
    volume:   {mechanism: laplace, epsilon: 0.5, bounds: [0, 10000]}
    price:    {mechanism: laplace, epsilon: 0.5, bounds: [0, 1000]}
    fee_rate: {mechanism: generalize, bins: 10}
```

- **Absent block = no privacy stage.** Presence is the switch; there is no `enabled` flag.
  README states this in one sentence.
- `key_env` optional. If set, `TransactionPipeline` reads `os.environ[key_env]` and passes
  it as `key`; unset env var → `ValueError` at pipeline construction.
- `bounds` are public parameters set a priori. Deriving them from data leaks extremes.
- Validation at `DignityConfig.from_yaml` → `ValueError`: unknown mechanism; `epsilon <= 0`;
  `laplace` without `bounds`; `bounds[0] >= bounds[1]`; `generalize` with `epsilon` or
  `bounds`; `k < 2`; `bins < 2`; sum of feature ε > `epsilon_total`.
- `DignityConfig.to_yaml` round-trips the block.
- `config/base.yaml`: block present, commented out, with the example above.
- `config/train_risk.yaml`: block **on**, with synthetic-appropriate bounds
  (`volume: [0, 1000]`, `price: [0, 500]`, `fee_rate: generalize bins 10`), so the default
  training path exercises the stage.
- `config/train_forecast.yaml`, `config/colab.yaml`: block absent (unchanged).

### Stated limits

1. **Pseudonymization is a primitive, not a stage.** No current source has identifier
   columns. Wiring waits for a source that does.
2. **k covers the generalized column, not features derived from it.** Rolling windows
   over generalized values can re-split equivalence classes. Post-processing preserves
   DP; it does not preserve k. Users needing k on model input choose columns that are
   not signal sources.

## 7. Operator layer — `docs/THREAT-MODEL.md`

One page. Two adversaries. Deniability defined here or nowhere.

**Adversary A — holds training artifacts** (weights, ONNX, preprocessed features).
Subject layer defends: every raw value released into training is ε-LDP or k-anonymous.
Not defended: an adversary holding raw source data; model-level memorization beyond
what input noise suppresses (no DP-SGD).

**Adversary B — observes the operator's machine or network.** Operator layer defends,
and *deniability* means exactly: (1) inference and export path perform no network I/O;
(2) ONNX artifact is self-contained — no external data, no embedded URLs or absolute
paths; (3) nothing in the package phones home. Not defended: local disk access
(artifacts unencrypted at rest); traffic analysis of what the operator does with
predictions.

**Limitations:** floating-point Laplace; public `bounds`; k not preserved under
post-processing; no DP-SGD; validation on synthetic data only so far.

## 8. Testing

Tests are named for the outcome they protect. Determinism comes from the injectable `rng`:
a constant `0.5` yields exactly zero noise (inverse CDF at u = 0), making clipping and
ordering testable bit-for-bit; `random.Random(42)` makes distributional tests reproducible.

### Removed

`tests/test_core.py::TestPrivacyManager` (4 tests). Two call the API without a key; one
(`test_add_noise`, `atol=50`) asserts noise is *small* — a passing test that privacy is weak.

### `tests/test_privacy.py` (new)

| Unit | Tests |
|---|---|
| `PrivacyBudget` | `refuses_to_overspend`, `allows_spending_exactly_to_total`, `composes_sequentially` |
| `pseudonymize` | `same_key_is_deterministic`, `different_keys_are_unlinkable`, `rejects_short_key`, `requires_key_at_call_time`, `distinct_key_id_pairs_never_collide_on_concatenation` |
| `add_laplace_noise` | `clips_to_bounds_before_noising`, `noise_scale_is_sensitivity_over_epsilon` (10k draws; mean abs noise within 5% of Δ/ε), `spends_budget`, `rejects_nonpositive_epsilon`, `rejects_inverted_bounds`, `default_rng_is_system_random` |
| `generalize_amounts` | `every_equivalence_class_has_at_least_k`, `holds_under_heavy_ties`, `preserves_record_count`, `rejects_n_below_k` |

### `tests/test_operator.py` (new)

- `inference_path_imports_no_network_modules` — `ast`-parse every `.py` under `core/`,
  `data/`, `models/`, `export/`, `train/`; assert no import of `socket`, `http`, `urllib`,
  `ssl`, `requests`, `aiohttp`, `ccxt`, `websocket`. Static; no `sys.modules` inspection.
- `onnx_export_is_self_contained` — export a tiny `Dignity` in-test; `onnx.load`; assert no
  tensor has `EXTERNAL` data location; no `metadata_props`/`doc_string` contains `://` or
  an absolute path.
- `predict_succeeds_with_sockets_disabled` — monkeypatch `socket.socket` to raise; run
  `Dignity.predict` on a random tensor.

### `tests/test_docs.py` (new)

- `docs_do_not_reintroduce_removed_claims` — grep `README.md` and `docs/**/*.md` for
  `Secure Aggregation`, `hash_identifiers(`, `anonymize_addresses`, `sanitize_dataset`,
  `suppress_rare_events`, `differentially private model`. Any hit fails.

### `tests/test_data.py` (additions to `TestTransactionPipeline`)

- `privacy_stage_runs_before_signals` — zero-noise rng, bounds tighter than the data;
  derived volatility reflects clipped values.
- `fit_transform_spends_budget_exactly_once` — after `process(fit=True)`, `budget.spent`
  equals the sum of configured ε, not twice it.
- `absent_privacy_block_is_a_noop` — output identical to the no-privacy pipeline.
- `budget_exhausted_propagates_from_transform`.
- `config_rejects_bad_privacy_block` — parametrized over the validation cases in §6.

### Count

78 test functions; **89 collected** once parametrized cases expand — this work's own tests;
the merged tree with `master`'s production-path suites collects 384 (test_privacy 34,
test_core 21, test_data 18, test_models 11, test_operator 4, test_docs 1). Every new test
maps to a sentence in §3, §6, or §7.

## 9. CI/CD

Nothing exists today. `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: { branches: [master] }
  pull_request: {}
permissions: { contents: read }
concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ["3.10", "3.12"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "${{ matrix.python-version }}", cache: pip }
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt && pip install -e . --no-deps
      - run: pytest tests/ --cov --cov-fail-under=100
```

### Gates and why

- **`--cov --cov-fail-under=100`** (coverage `include = ["core/privacy.py"]` in pyproject) — the honest-core principle as a build
  failure. Not repo-wide; the one module whose every line is a claim.
- **Torch from the CPU index first** — default Linux wheel pulls CUDA (2GB+). CPU wheel is
  ~200MB. `-e . --no-deps` so `setup.py` doesn't re-resolve torch.
- **Python 3.10 and 3.12** — declared floor and current stable with solid torch wheels.
- **`ruff format --check`** forces the width decision → 88 (§2). `pyproject.toml`
  `line-length = 88`; `.editorconfig` `max_line_length = 88`. Zero reformat diff.

### Pre-commit — `.pre-commit-config.yaml`

Ruff lint and ruff format hooks only (`astral-sh/ruff-pre-commit`, pinned). No pytest —
torch makes it a 30-second commit, and a bypassed hook is worse than none. `pre-commit`
added to `setup.py` `dev` extras. README contributing section gains `pre-commit install`.

### Riding along

- README `tests-31 passing` static badge → real Actions workflow badge.
- `.gitignore` gains `.coverage` and `htmlcov/`.

### Your action after first green run

Branch protection on `master` requiring `lint` and `test` — a GitHub repo setting, not
code. Without it CI reports; with it CI enforces.

## 10. Documentation changes

| File | Change |
|---|---|
| `docs/PRIVACY.md` | Rewrite around §3; real API only |
| `docs/THREAT-MODEL.md` | New; §7 |
| `docs/ARCHITECTURE.md` | Fix privacy snippet |
| `docs/training-curriculum.md` | Delete |
| `README.md` | Privacy sections to match §3; one sentence "no `privacy:` block = no privacy stage"; badge; contributing gains pre-commit |
| `.claude/CLAUDE.md` | Update: formatter width resolved to 88; `PrivacyManager` is now instance-based; `PrivacyConfig` exists; privacy stage order; remove "all-static utility classes" line for `PrivacyManager`; `SignalProcessor` stays static |

## 11. Out of scope

- DP-SGD / Opacus (phase 2, gated on this work).
- Snapping mechanism for floating-point Laplace.
- Pseudonymization pipeline stage (no identifier-bearing source exists).
- GPU training on the AMD Polaris card (separate spike; ROCm gfx803 is unofficial).
- README non-privacy drift; `requirements.txt` vs `setup.py` unification; `torch.cuda.amp`
  → `torch.amp` migration; `train/cli.py --resume` no-op; `dignity-export` missing `main()`.
- Reconciling local `master` (ahead 1 / behind 6) with `origin/master` — deliberate revert,
  user's call.

## 12. Success criteria

All verifiable by command; none by assertion.

- [ ] `pytest tests/ -v` green (384 collected on the merged tree; 89 from this work), on Python 3.10 and 3.12.
- [ ] `pytest tests/ --cov --cov-fail-under=100` passes (only `core/privacy.py` is measured).
- [ ] `ruff check .` and `ruff format --check .` clean at width 88.
- [ ] CI workflow green on both matrix legs on the PR.
- [ ] `grep -rn "suppress_rare_events\|sanitize_dataset\|Secure Aggregation\|hash_identifiers(" README.md docs/` returns nothing.
- [ ] `PrivacyManager` has no callers with `sensitivity=`; every `add_laplace_noise` call passes `bounds`.
- [ ] `dignity-train --config config/train_risk.yaml` prints an ε-spent line before epoch 1.
- [ ] Every claim in `README.md` and `docs/PRIVACY.md` has a named test in §8.
