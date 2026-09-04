# Threat Model

Dignity makes two privacy claims for two different people. This page says exactly
what each claim covers, what it does not, and what "deniability" means here. Every
property below is enforced by a named test; if the test does not exist, the
property is not claimed.

## Adversary A — holds the training artifacts

Anyone with the trained weights, the ONNX export, or the preprocessed feature
arrays. This is the **subject layer**: it protects individuals whose transactions
appear in the training data.

**Defended.** Every raw value released into training passes through one of:

- *Bounded Laplace noise* — ε-differential privacy at the level of each released
  value. Values are clipped to public `bounds`, sensitivity is the width of those
  bounds, and every call spends ε from a per-pipeline ledger that refuses to
  exceed `epsilon_total`. Tests: `TestAddLaplaceNoise`, `TestPrivacyBudget`,
  `TestPrivacyStage::test_fit_transform_spends_budget_exactly_once`.
- *k-anonymous generalization* — every value in the generalized column is shared
  by at least k records. Test: `TestGeneralizeAmounts::test_every_equivalence_class_has_at_least_k`.

Signals derived from a noised column (volatility, momentum, directional change)
inherit its ε by the post-processing theorem, because the privacy stage runs
before signal computation. Test: `TestPrivacyStage::test_privacy_stage_runs_before_signals`.

**Not defended.**

- An adversary who already holds the raw source data. No input mechanism helps
  once the input is theirs.
- Memorization inside the trained weights beyond what input noise suppresses.
  Training is ordinary SGD; the trained model carries no DP guarantee of its own.
- k-anonymity of features *derived* from a generalized column. Rolling windows
  can re-split equivalence classes. The k claim is for the column as released.

## Adversary B — observes the operator's machine or network

Someone watching the person who runs the model. This is the **operator layer**.

**Defended — and this is the whole definition of "deniability" in this project:**

1. The inference and export path performs no network I/O.
   Test: `TestOperatorLayer::test_inference_path_imports_no_network_modules`.
2. The ONNX artifact is self-contained: no external data files, no embedded URLs,
   no embedded local paths. Test: `TestOperatorLayer::test_onnx_export_is_self_contained`.
3. Prediction works with sockets disabled; nothing phones home.
   Test: `TestOperatorLayer::test_predict_succeeds_with_sockets_disabled`.

**Not defended.**

- An adversary with access to the operator's disk. Artifacts are not encrypted
  at rest.
- Traffic analysis of whatever the operator does *with* the predictions.

## Known limitations

- **Floating-point Laplace.** The noise is sampled by inverse CDF in IEEE
  floating point, which is vulnerable to the attack in Mironov, *On Significance
  of the Least Significant Bits for Differential Privacy* (CCS 2012). The
  snapping mechanism that fixes this is not implemented.
- **`bounds` are public.** Choosing them from the data would leak its extremes,
  so they must be set a priori in config. They are visible to any adversary.
- **Pseudonymization is a primitive, not a pipeline stage.** No current data
  source carries identifier columns, so `PrivacyManager.pseudonymize` is exposed
  and tested but not wired into `TransactionPipeline`.
- **Validation is on synthetic data.** No real transaction data has passed
  through this pipeline. The mechanisms are properties of the algorithm and hold
  regardless; empirical re-identification risk on real subjects is untested.
