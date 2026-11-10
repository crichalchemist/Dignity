# Threat Model

Dignity makes two privacy claims for two different people. This page says exactly
what each claim covers, what it does not, and what "deniability" means here. Every
property below is enforced by a named test; if the test does not exist, the
property is not claimed.

## Adversary A — holds the training artifacts

Anyone with the trained weights, the ONNX export, or the preprocessed feature
arrays. This is the **subject layer**: it protects individuals whose transactions
appear in the training data.

**Defended.** Every raw column listed under `privacy.features` passes through one of:

- *Bounded Laplace noise* — ε-differential privacy at the level of each released
  value. Values are clipped to public `bounds`, sensitivity is the width of those
  bounds, and every call spends ε from a ledger (a ledger per split in
  `process_blocks`, and one per pipeline on the fit/transform path) that refuses
  to exceed `epsilon_total`. Tests: `tests/test_privacy.py::TestAddLaplaceNoise`,
  `tests/test_privacy.py::TestPrivacyBudget`,
  `tests/test_data.py::TestPrivacyStage::test_fit_transform_spends_budget_exactly_once`.
  A column may first pass through a fixed public function (`transform: log1p`);
  clipping, sensitivity and `bounds` are then in transformed units, and the released
  column stays in those units. Test:
  `tests/test_data.py::TestPrivacyStage::test_log1p_transform_is_applied_before_clipping_and_noise`.
- *k-anonymous generalization* — every value in the generalized column of the
  train release is shared by at least k train records. Tests:
  `tests/test_privacy.py::TestGeneralizeAmounts::test_every_equivalence_class_has_at_least_k`,
  `tests/test_privacy.py::TestGeneralization::test_train_classes_hold_at_least_k_fit_records`.

Signals derived from a noised column (volatility, momentum, directional change)
inherit its ε by the post-processing theorem, because the privacy stage runs
before signal computation. Test: `tests/test_data.py::TestPrivacyStage::test_privacy_stage_runs_before_signals`.

**Not defended.**

- Raw model inputs that are not listed under `privacy.features`, and labels. The
  stage covers only what the config names; `train/loop.py::prepare_data` releases labels as-is.
  The shipped `config/train_risk.yaml` lists every raw source column its model
  consumes (`volume`, `fee_rate`, `tx_count`; `price` is not consumed). The test maps each
  derived signal back to its source column through `data.pipeline.SIGNAL_SOURCES`,
  so `price` would count as consumed whenever a price signal were. Test:
  `tests/test_core.py::TestPrivacyConfig::test_shipped_risk_config_covers_every_raw_model_input`.
- An adversary who already holds the raw source data. No input mechanism helps
  once the input is theirs.
- Memorization inside the trained weights beyond what input noise suppresses.
  Training is ordinary SGD; the trained model carries no DP guarantee of its own.
- k-anonymity of features *derived* from a generalized column. Rolling windows
  can re-split equivalence classes. The k claim is for the generalized column of the
  train release, not for derived features and not for val or test.

### Per-split releases and parallel composition

`TransactionPipeline.process_blocks` makes three privacy releases, one per block split (train, val, test). Each has its own `PrivacyManager` and `PrivacyBudget` and spends the full `epsilon_total`. The splits are disjoint sets of blocks (test: `tests/test_data.py::TestSequenceAwareSplit::test_every_block_lands_in_exactly_one_split`), and in the synthetic path every subject's rows lie in one block. No released value therefore belongs to two releases, and by parallel composition the three releases together are ε-differentially private per released Laplace value, not 3ε. Per subject the bound stays m·ε by group privacy, with m = `block_len`. The RobustScaler and the `generalize` class boundaries and representatives are fitted on the train release only and then applied to val and test.

k-anonymity holds for the train release only: every class has at least k train records. The fitted boundaries and representatives are computed from raw train values, so they are part of what the train release discloses. Val and test values are mapped onto those train classes with no per-split check, so a class can hold fewer than k val or test records, and no k claim is made for val or test. Tests: `tests/test_data.py::TestSequenceAwareSplit::test_each_split_spends_its_own_full_budget`, `tests/test_data.py::TestSequenceAwareSplit::test_generalize_edges_and_scaler_fit_on_train_release_only`, `tests/test_privacy.py::TestGeneralization::test_train_classes_hold_at_least_k_fit_records`.

## Adversary B — observes the operator's machine or network

Someone watching the person who runs the model. This is the **operator layer**.

**Defended — and this is the whole definition of "deniability" in this project:**

1. The inference and export path performs no network I/O.
   Test: `tests/test_operator.py::TestOperatorLayer::test_inference_path_imports_no_network_modules`.
2. The ONNX artifact is self-contained: no external data files, no embedded URLs,
   no embedded local paths. Test: `tests/test_operator.py::TestOperatorLayer::test_onnx_export_is_self_contained`.
3. Prediction works with sockets disabled; nothing phones home.
   Test: `tests/test_operator.py::TestOperatorLayer::test_predict_succeeds_with_sockets_disabled`.

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
- **The unit of privacy is one released value** (one row of one column). A subject
  who contributes m rows is protected at m·ε by group privacy, not ε. In the
  shipped synthetic path every subject contributes one block of
  `synthetic.block_len` rows (160 by default). User-level
  accounting is not implemented; the CLI's "ε spent" line is the per-value ledger.
- **A pipeline releases once, on every path.** A `TransactionPipeline` with a `privacy:` block makes exactly one release of data: either one `fit` or `fit_transform` call (one ledger on the instance), or one `process_blocks` call (three releases, one ledger per split). Any further `fit`, `transform`, `fit_transform` or `process_blocks` raises `BudgetExhausted`. A flag on the instance enforces this, not the ledger, so it holds for a generalize-only block (which spends no ε) and for a Laplace block whose feature epsilons sum to less than `epsilon_total`. A `transform` on an unfitted pipeline raises `RuntimeError` before releasing, so it does not count as the release. v1.0 has no inference-time release path. New data would be a new, disjoint release with its own budget (parallel composition again), noised at collection with the same `privacy:` block, so the model would see inputs from the distribution it was trained on. That path is not implemented. Tests: `tests/test_data.py::TestPrivacyStage::test_budget_exhausted_propagates_from_transform`, `tests/test_data.py::TestSequenceAwareSplit::test_a_pipeline_never_releases_twice_on_any_pair_of_paths` (generalize and Laplace; every first path against every second path), `tests/test_data.py::TestSequenceAwareSplit::test_no_further_release_after_process_blocks`, `tests/test_data.py::TestPrivacyStage::test_unfitted_transform_fails_before_it_releases_anything`.
- **Pseudonymization is a primitive, not a pipeline stage.** No current data
  source carries identifier columns, so `PrivacyManager.pseudonymize` is exposed
  and tested but not wired into `TransactionPipeline`.
- **Validation is on synthetic data.** No real transaction data has passed
  through this pipeline. The mechanisms are properties of the algorithm and hold
  regardless; empirical re-identification risk on real subjects is untested.
