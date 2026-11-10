# Privacy Operations

`core/privacy.py` implements two things, and claims exactly two things. Each
sentence in **bold** below is backed by a named test under `tests/`:
`tests/test_privacy.py` for the primitives, `tests/test_data.py::TestPrivacyStage`
and `tests/test_data.py::TestSequenceAwareSplit` for the pipeline stage,
`tests/test_core.py::TestPrivacyConfig` for config, and
`tests/test_operator.py::TestOperatorLayer` for the operator layer.
For who these protect and from whom, read [THREAT-MODEL.md](THREAT-MODEL.md).

## What is claimed

| Guarantee | Mechanism | Wording we use | Wording we do not use |
|---|---|---|---|
| Identifier protection | HMAC-SHA256 with a required key | *keyed pseudonymization* | "anonymized" |
| Value protection | Clipped Laplace noise, ε ledgered | *input-level ε-differential privacy* | any DP claim about the trained model |
| Value protection | Quantile bins merged to ≥ k | *k-anonymity for the generalized column of the train release* | k claims about derived features, val or test |

## Setup

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

# One budget per release. Epsilons add across calls; overspending raises.
budget = PrivacyBudget(epsilon_total=1.0)

# The key is only needed for pseudonymization. Load it from the environment;
# a literal here is for illustration only.
pm = PrivacyManager(budget, key=b"replace-with-16+-random-bytes")
```

## Keyed pseudonymization

**Same key, same identifier → same pseudonym. Different key → unrelated pseudonym.**
The key is required at call time; there is no unkeyed fallback. Keys under 16
bytes are rejected.

```python
pseudonym = pm.pseudonymize("0x1234abcd5678ef90")  # 64 hex chars
many = pm.pseudonymize_many(["addr_a", "addr_b", "addr_a"])  # many[0] == many[2]
```

This is reversible pseudonymization, not identity removal: anyone holding the
key can link records. That is the intended property — it lets you join across
your own datasets while making the pseudonyms useless to anyone without the key.

## Bounded Laplace noise (input-level ε-DP)

**Values are clipped to `bounds` before noising, so sensitivity is `hi - lo` and
cannot be understated.** **ε is spent from the budget before any sample is drawn.**
**By default, noise comes from `secrets.SystemRandom`, not a seedable PRNG.**
(`tests/test_privacy.py::TestAddLaplaceNoise::test_default_rng_is_system_random`,
`tests/test_data.py::TestPrivacyStage::test_system_randomness_is_the_default_mode`.)
A seed alone never changes this: the training path always supplies one, and only
`deterministic_rng` turns it into a seeded stream
(`tests/test_data.py::TestPrivacyStage::test_system_mode_ignores_a_supplied_seed`),
and both shipped training configs leave it off
(`tests/test_core.py::TestPrivacyConfig::test_shipped_configs_stay_on_system_randomness`).

A column may first pass through a fixed public function (`transform: log1p`,
laplace only); clipping, sensitivity and `bounds` are then in log1p units, and the
released column stays in log1p units (the noise is never inverted). A fitted scaler
is not allowed here; it would derive the transform from the data.
(`tests/test_data.py::TestPrivacyStage::test_log1p_transform_is_applied_before_clipping_and_noise`.)

**Experiment mode.** `privacy.deterministic_rng: true` plus a seed gives each split
its own seeded `random.Random`, so the privacy release reproduces exactly. This is seeded
experiment mode, not DP-grade randomness: reproducibility and DP-grade randomness
are mutually exclusive, so use it for experiments and the published curve, never
for a release. The CLI prints the active mode, and a real release records it
(`tests/test_data.py::TestPrivacyStage::test_a_real_release_records_system_mode_and_says_it_is_not_reproducible`,
`tests/test_data.py::TestPrivacyStage::test_a_release_through_the_rng_seam_records_injected_mode`). Tests:
`tests/test_data.py::TestPrivacyStage::test_deterministic_mode_reproduces_the_release`,
`tests/test_data.py::TestPrivacyStage::test_deterministic_splits_draw_distinct_noise`,
`tests/test_data.py::TestPrivacyStage::test_deterministic_mode_without_seed_is_refused`,
`tests/test_data.py::TestPrivacyStage::test_describe_privacy_names_the_mode_and_warns_when_unseeded`.

```python
amounts = np.array([123.4, 789.0, 456.7, 5000.0])  # 5000 will be clipped to 1000
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))
budget.spent  # 0.5
budget.remaining  # 0.5
pm.add_laplace_noise(
    amounts, epsilon=0.6, bounds=(0.0, 1000.0)
)  # raises BudgetExhausted
```

Each released value is ε-differentially private for that feature (local DP).
Spending ε₁ on one column and ε₂ on another composes to ε₁ + ε₂; the ledger
enforces that the sum never exceeds `epsilon_total`.

The unit is one value in one row. A subject contributing m rows is protected at
m·ε (group privacy); see `docs/THREAT-MODEL.md`, Known limitations.

`bounds` are **public parameters**. Do not derive them from the data — that leaks
the extremes. Choose them from domain knowledge before you look.

## k-anonymous generalization

**Every output value is shared by at least k records.** Quantile edges start the
bins balanced; any bin with fewer than k members is merged into its smaller
neighbour until none remain. Each record becomes the midpoint of its class's
min and max.

```python
volumes = np.random.default_rng(0).lognormal(3, 1, size=500)
generalized = PrivacyManager.generalize_amounts(volumes, bins=10, k=5)
```

Use `PrivacyManager.fit_generalization` to fit classes on one release and
`Generalization.apply` to map another release onto them;
**k-anonymity holds for the train release only** (the release the classes were
fitted on). Val and test
values are mapped onto the train classes with no per-split check. Tests:
`tests/test_privacy.py::TestGeneralization::test_train_classes_hold_at_least_k_fit_records`,
`tests/test_data.py::TestSequenceAwareSplit::test_generalize_edges_and_scaler_fit_on_train_release_only`.

This holds for the generalized column of the train release. Signals computed from it
downstream (rolling volatility, momentum) are **not** covered by the k claim.

## The privacy stage in the pipeline

`TransactionPipeline` applies these mechanisms to raw columns **before**
computing signals, so derived features inherit the DP guarantee by
post-processing. The stage runs exactly once per `fit`, `transform`, or
`fit_transform`, and once per split in `process_blocks`, which makes three
releases under **parallel composition** over disjoint blocks (see THREAT-MODEL,
*Per-split releases and parallel composition*; tests
`tests/test_data.py::TestSequenceAwareSplit::test_each_split_spends_its_own_full_budget`,
`tests/test_data.py::TestSequenceAwareSplit::test_every_block_lands_in_exactly_one_split`).
A pipeline releases once on every path: a second `fit`, `transform`,
`fit_transform` or `process_blocks` raises `BudgetExhausted` (test
`tests/test_data.py::TestSequenceAwareSplit::test_a_pipeline_never_releases_twice_on_any_pair_of_paths`).
It is driven by a `privacy:` block in the YAML config:

```yaml
privacy:
  key_env: DIGNITY_PRIVACY_KEY       # env var NAME; only used if you call pseudonymize() yourself
  epsilon_total: 1.0
  k: 5
  deterministic_rng: false           # default; true = seeded experiment mode, NOT DP-grade randomness
  features:
    volume:   {mechanism: laplace, epsilon: 0.5, bounds: [0, 1000]}
    price:    {mechanism: laplace, epsilon: 0.5, bounds: [0, 500]}
    fee_rate: {mechanism: generalize, bins: 10}
    tx_count: {mechanism: generalize, bins: 10}
```

**No `privacy:` block means no privacy stage runs.** Misconfiguration — an unknown
mechanism, ε ≤ 0, missing or inverted bounds, k < 2, or feature epsilons summing
past `epsilon_total` — fails at config load, not mid-training.

### What the shipped config does

`config/train_risk.yaml` (and `config/train_policy.yaml`, which carries the same
block so the policy warm start from a risk checkpoint is accepted) consumes
`volume`, `fee_rate`, `tx_count` and `volume_volatility`, and releases them under:

- `volume`: Laplace at the whole `epsilon_total: 1.0`, after the public `log1p`
  transform, with `bounds` equal to `public_log1p_volume_bounds()` of the file's
  `data.synthetic` rounded outward to two decimals. The bounds come from declared
  generator parameters, never from data.
- `fee_rate`, `tx_count`: k-anonymous generalization (`bins: 10`, `k: 5`). They are
  **not** under the Laplace mechanism and are not governed by epsilon.
- `price` is neither released nor consumed, and neither are its three signals
  (`volatility`, `price_change`, `momentum`, `directional_change`), so
  `flash_crash`, a price-only typology, is invisible to the shipped model.

The per-value epsilon is 1.0 (the eps = 1 point of the evaluation grid); the
per-subject cost of a block is `block_len` x 1.0, because a block's rows are
correlated and each row is a release (THREAT-MODEL). The utility cost is
reported with the privacy sweep, not here. A pipeline
instance releases **once**, whatever the epsilons: `process_blocks` (what
`train/loop.py::prepare_data` calls) splits train/val/test by block and releases
each split on its own budget (parallel composition over disjoint blocks). Any
later `fit`, `transform`, `fit_transform` or `process_blocks` raises
`BudgetExhausted`, whichever path released first. A flag enforces this, so it
holds for a generalize-only block and for feature epsilons that sum to less than
`epsilon_total`. A `transform` on an unfitted pipeline raises `RuntimeError`
before releasing, so the release is kept for a later `fit`. This is deliberate
(THREAT-MODEL, Known limitations). Tests:
`tests/test_data.py::TestPrivacyStage::test_budget_exhausted_propagates_from_transform`,
`tests/test_data.py::TestSequenceAwareSplit::test_a_pipeline_never_releases_twice_on_any_pair_of_paths`,
`tests/test_data.py::TestSequenceAwareSplit::test_no_further_release_after_process_blocks`,
`tests/test_data.py::TestPrivacyStage::test_unfitted_transform_fails_before_it_releases_anything`.
Fitting the generalize classes needs at least `k` rows in the train release; val
and test values map onto the train classes at any size, with no k claim for them.

`dignity-train` prints `Privacy: mode=<mode>; ε spent per split train=X val=X test=X, each of E (parallel composition over disjoint blocks)`
(each to three decimals, with a mode-specific suffix in system and deterministic mode; `tests/test_data.py::TestPrivacyStage::test_describe_privacy_names_the_mode_and_warns_when_unseeded`) after preprocessing so the ledger is visible, not
theoretical.

## What is not here

- No differential privacy on the trained weights (no DP-SGD). The model itself
  carries no DP guarantee.
- No protocol for combining values across participants without revealing them
  individually, no federated learning.
- No snapping mechanism; see THREAT-MODEL.md on floating-point Laplace.

## Reference

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014) —
  Laplace mechanism (§3.3), sequential composition (§3.5), post-processing (Prop. 2.1).
- Sweeney, *k-Anonymity: A Model for Protecting Privacy* (2002).
- Mironov, *On Significance of the Least Significant Bits for Differential Privacy* (CCS 2012).
