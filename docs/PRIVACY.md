# Privacy Operations

`core/privacy.py` implements two things, and claims exactly two things. Each
sentence in **bold** below is backed by a named test in `tests/test_privacy.py`.
For who these protect and from whom, read [THREAT-MODEL.md](THREAT-MODEL.md).

## What is claimed

| Guarantee | Mechanism | Wording we use | Wording we do not use |
|---|---|---|---|
| Identifier protection | HMAC-SHA256 with a required key | *keyed pseudonymization* | "anonymized" |
| Value protection | Clipped Laplace noise, ε ledgered | *input-level ε-differential privacy* | any DP claim about the trained model |
| Value protection | Quantile bins merged to ≥ k | *k-anonymity for the column as released* | k claims about derived features |

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
**Noise comes from `secrets.SystemRandom`, not a seedable PRNG.**

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

This holds for the generalized column as released. Signals computed from it
downstream (rolling volatility, momentum) are **not** covered by the k claim.

## The privacy stage in the pipeline

`TransactionPipeline` applies these mechanisms to raw columns **before**
computing signals, so derived features inherit the DP guarantee by
post-processing. The stage runs exactly once per `fit`, `transform`, or
`fit_transform`. It is driven by a `privacy:` block in the YAML config:

```yaml
privacy:
  key_env: DIGNITY_PRIVACY_KEY       # env var NAME; only used if you call pseudonymize() yourself
  epsilon_total: 1.0
  k: 5
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

`config/train_risk.yaml` is written to exercise the stage, not to train a useful
model. With `epsilon_total: 1.0` split 0.5/0.5, the Laplace scale is
width/ε = 2000 for `volume` and 1000 for `price` — far above the synthetic
signal, so released values carry little information about the source. Because
the feature epsilons sum to `epsilon_total`, a pipeline instance can release
**once**: `process_blocks` (what `train/cli.py` calls) spends the whole budget in
a single release and only then splits train/val, and any later `transform`
raises `BudgetExhausted`. `generalize` needs at least `k` rows per release, so a fitted pipeline
cannot score batches smaller than `k` with the stage on.

`dignity-train` prints `Privacy: ε spent X of Y` after preprocessing so the
ledger is visible, not theoretical.

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
