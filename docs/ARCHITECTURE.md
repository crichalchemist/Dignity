# Architecture

Dignity has one data path, one backbone and three heads. All three tasks are about
risk: **risk** detects an anomalous window, **forecast** warns early that an anomaly
will start within the horizon, and **policy** triages a window to `allow`, `review`
or `block` under a review budget.

```
SyntheticGenerator.generate()        event record: one row per step, blocks contiguous
        |
TransactionPipeline.process_blocks   split by block -> privacy release per split
        |                            -> signals per block -> scaler (train only) -> windows
        v
SplitData(train, val, test)          WindowSet: X [N, T, F] plus every label
        |
Dignity(task=...)                    DignityBackbone -> context [B, H], attn [B, T]
        |                            + RiskHead | ForecastHead | PolicyHead
        v
train.loop.fit  ->  best.pt  ->  export.to_onnx
eval.run (calls fit once per training seed)  ->  results, charts
```

Each section names the symbols and tests that carry its claims. For metric values
see [RESULTS.md](RESULTS.md); this document quotes none. For the privacy stage see
[PRIVACY.md](PRIVACY.md) and [THREAT-MODEL.md](THREAT-MODEL.md).

## Data contract

`data.source` is `synthetic` only in v1.0: `DataConfig` rejects anything else at load
(`tests/test_core.py::TestDignityConfig::test_unsupported_data_source_is_rejected_not_silently_synthetic`).
`CryptoSource` in `data/source/crypto.py` is present but unsupported until v1.1 (DGNTY-41).

`data/source/synthetic.py::SyntheticGenerator.generate` returns an event record with
exactly these columns. Every block is an independent normal process
(`data/source/dynamics.py`); with probability `data.synthetic.contamination` it
receives typed anomalies at random onsets.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `timestamp` | int64 | Milliseconds since the epoch, one row per hour. Milliseconds are normative for every source. |
| `volume` | float64 | Transaction volume of the step. |
| `price` | float64 | Price level of the step. |
| `fee_rate` | float64 | Fee level of the step. |
| `tx_count` | int64 | Number of transactions in the step. |
| `block_id` | int64 | Which contiguous block the row belongs to; the unit of the train/val/test split. |
| `is_anomalous` | bool | True on every row inside an injected anomaly. |
| `anomaly_type` | str (object under pandas 2) | The anomaly's type on anomalous rows, `none` elsewhere. |
| `anomaly_id` | int64 | Counter of injected anomalies, `-1` on normal rows. Every anomaly gets its own id, so back-to-back anomalies stay distinct (`tests/test_labels.py::TestWindowLabels::test_back_to_back_anomalies_count_as_a_new_onset`). |
| `block_label` | int8 | 1 when the block holds an anomaly, flipped with probability `data.synthetic.label_noise`. Only the block-level baseline trains on it. |
| `build_up` | int8 | 0 on ordinary rows, 1 on the rows of a build-up that precedes an anomaly, 2 on a decoy. For tests and analysis only: `data.pipeline.EVENT_LABEL_COLUMNS` refuses it, and every other label column, as a feature. |

`fee_rate` and `tx_count` are not calibrated against public data. They are only kept
plausible: `fee_rate` has a spread, and `tx_count` tracks volume within a block and is
overdispersed (`tests/test_data.py::TestNormalProcessPlausibility`).

`data.source.anomalies.ANOMALY_TYPES` holds six types. Each injector edits only
`volume`, `price`, `fee_rate` and `tx_count`, and the drawn severity scales the effect.
The second column lists the columns a type changes at full severity and the direction
each moves over the span; `tests/test_data.py::TestAnomalyInjectors` runs the injectors,
and so does
`tests/test_docs.py::TestArchitectureDoc::test_each_anomaly_row_names_the_columns_its_injector_changes`,
which checks this table against them.

| Type | Changes | Mechanism |
| --- | --- | --- |
| `wash_trading` | `volume` up, `tx_count` up, `price` damped | Volume and transaction count inflate while the `price` column varies less than normal. |
| `pump_and_dump` | `volume` up, `price` rises then falls | A volume ramp with a price rise over the first part of the span, then a sharper fall. |
| `fee_spike` | `fee_rate` up, `tx_count` up | `fee_rate` is multiplied by up to 10 and `tx_count` by up to 2; price and volume stay normal. |
| `flash_crash` | `volume` up, `price` drops then recovers partly | One sharp price drop on the first row, where volume peaks; volume stays raised for the rest of the span while the price partly recovers. |
| `layering` | `volume` up | Every fourth row gets an added volume spike. |
| `fee_evasion` | `fee_rate` down, `volume` up | `fee_rate` is cut towards zero while volume rises. |

`benign_spike_rate` adds a look-alike burst to normal blocks without any label
(`data/source/anomalies.py::inject_benign_spike`). `anomalies_per_block`,
`duration_range`, `severity_range` and `type_weights` shape the draw; see the comments
in [config/base.yaml](../config/base.yaml).

### Build-ups and decoys

With `data.synthetic.precursor_strength` above 0 the generator adds a build-up on rows
that stay labelled normal.

- `wash_trading`, `pump_and_dump`, `fee_spike` and `fee_evasion` get a linear build-up over
  the rows before the onset. Its length is drawn from `data.synthetic.precursor_len_range`
  and its intensity at row i of an n-row ramp is `data.synthetic.precursor_strength` times
  the anomaly's severity times (i+1)/n: `data.source.anomalies.apply_build_up` scales the
  type's own effect down by that weight.
  Accumulation lifts volume and the quoted level drifts up; a backlog lifts counts and fee
  rates; structuring probes the fee level. A ramp's volume, fee and count effects stay
  below the anomaly's own, but a pump ramp's drift can lift the level by more than the
  pump itself does, and it moves the quoted level for the rest of its block. No return
  outside a ramp changes.
- `flash_crash` and `layering` have no build-up, so about a third of first onsets stay
  unpredictable by construction.
- Build-up rows stay labelled normal: `is_anomalous` is false, `anomaly_type` is `none`,
  and `block_label` is unchanged.
- A true build-up that the block start or the previous anomaly clips to fewer rows than the
  range minimum is skipped, so a short ramp cannot give the outcome away. A ramp can still
  start right where the previous anomaly ended (recurrence context).
- A decoy is a build-up that leads to no anomaly: with probability
  `data.synthetic.decoy_rate` a block holds at most one, of a type drawn evenly among the
  four, with the same length law and intensity as a true one. It may not overlap or touch
  an anomaly or a true build-up (rows one before to one after hold neither), so it never
  ends where an anomaly begins; after ten tries without room the block has none. Its rows
  are marked `build_up` 2 and stay labelled normal. Two limits follow from these rules.
  An anomaly of a type without a build-up can still begin a few rows after a decoy by
  chance. And a decoy never starts on the row right after an anomaly, so a build-up that
  starts there is a true one; the risk score sees the same earlier anomaly.
- Strength 0 reproduces the earlier frame exactly, because the ramps use their own random
  stream, and no label moves at any strength. Decoys use a stream of their own, so
  `data.synthetic.decoy_rate` never moves a true build-up.

### Labels

All labels come from the per-row event record, never from a single row
(`data/labels.py`). A window covers `data.seq_len` rows; its last row is `t`.

- **Risk.** `data/labels.py::risk_labels`: the window is positive when its anomalous
  fraction is strictly greater than `data.label_overlap`. At 0.0 one anomalous row is
  enough (`tests/test_labels.py::TestWindowLabels::test_overlap_must_exceed_the_configured_fraction`).
- **Forecast.** `data/labels.py::forecast_labels` returns `y [N, H]` and a `mask [N]`.
  `y[:, h-1]` is 1 when an anomaly onset (the first row of an `anomaly_id`) lies in
  `(t, t+h]`, so the labels are cumulative over h
  (`tests/test_labels.py::TestWindowLabels::test_early_warning_labels_are_cumulative_over_the_horizon`).
  `mask` is False where an anomaly is already active at `t`; those windows are left out
  of the loss, the metrics and the lead times, because their score reacts to the
  anomaly instead of warning of it.
- **Policy.** The true state is the risk label (anomaly or normal). The cost table
  `C[action][state]` is `PolicyConfig.costs` and the review cap is
  `policy.review_budget`, the share of windows analysts may review. The shipped values:

| Action | Cost if the window is an anomaly | Cost if it is normal |
| --- | --- | --- |
| `allow` | 10.0 | 0.0 |
| `review` | 1.0 | 1.0 |
| `block` | 0.5 | 3.0 |

### Presets, gate and calibration

`config/synthetic_{easy,medium,hard}.yaml` are the three difficulty presets; each
holds 3600 blocks, so that every tuning seed has at least 200 scorable test onsets.
`eval/gates.py::data_gate` generates each preset once per seed on the ten tuning
seeds, splits it by block and scores two baselines on the val and the test split:
`window_logistic` on windows and `block_tree` on blocks. The AUROCs, the recall
spread, the mean lift and the build-up share it judges are means over seeds, and the
forecast PR-AUC ratio is a ratio of two means; the onset count and each seed's lift
floor bind every seed on its own; each per-horizon win count must reach at least
seven eighths of the seeds, rounded up; the detection gain is pooled over all
seeds' onsets. It passes when all six checks hold: each baseline's hard-preset
AUROC, the mean of the val and test scores over all seeds, lies in
`eval.gates.AUROC_BAND`; the same means fall strictly from easy to medium to hard for
both baselines, and the build-up strength falls with them; the `window_logistic`
recall at 10% false-positive rate, per onset-position bucket and averaged over seeds
on the val split, spreads by at most `eval.gates.FLATNESS_MAX_SPREAD`, 0.15 (the
test split's spread is reported, never judged; the limit was 0.1 until the first
build-up strength search showed a val spread of about 0.11 with build-ups and 0.14
without them, while the larger test split showed about 0.09, so the old limit sat
inside the difference between the two splits); every seed's hard-preset test split
holds at least `eval.gates.MIN_TEST_ONSETS` scorable onsets (`test_onsets`); and the
forecast headroom check holds. That check fits the logistic probe
`eval/baselines.py::window_logistic_forecast` on the hard preset's windows, built from
the same `data.features` the model reads, and asks whether it forecasts anomalies
better than the stronger of today's risk score and the oracle (the true risk label)
at horizons 1, 5 and 10, whether it beats a random score at warning of onsets, whether
its lift over chance is real among windows no earlier anomaly has touched, and whether
the share of build-up windows that end in an onset sits inside a band. The PR-AUC
ratio and the per-horizon wins use the probe that is also told, as one extra input
column, whether an anomaly lies earlier in the window (`risk_aware=True`), because
the plain probe reads that far worse than the oracle it is compared with; the ratio
for the plain probe is reported beside it as `pr_auc_ratio_plain`. The lift, the
warnings and the build-up share use the plain probe. With no
build-ups in the data (strength 0) the check fails by
design (`tests/test_gates.py::TestForecastHeadroom`). The gate
is a tuning tool: `eval/gates.py::main` refuses a seed outside the tuning seeds, a
repeated seed and a strength off `eval.gates.STRENGTH_GRID` with exit code 2, before
it generates anything (`tests/test_gates.py::TestDataGate`). The values the shipped
presets reach are in
[results/v1.0/data_gate_tuning.json](../results/v1.0/data_gate_tuning.json).

`data/calibration/reference_stats.json` holds a band for each statistic in
`data.stats.STAT_NAMES`; `scripts/calibrate.py` builds it from public hourly bars,
and the generator's normal process must land inside the bands
(`tests/test_data.py::TestGeneratorCalibration::test_normal_process_lands_inside_calibration_bands`).

## Pipeline

`data/pipeline.py::TransactionPipeline.process_blocks` turns the event record into a
`SplitData`. The order of operations is fixed:

1. **Block shuffle and split.** Blocks are shuffled with the generator seed. Val and
   test each get `round(n * fraction)` blocks of `data.split` (at least one) and train
   gets the rest, so no block, and no window, straddles two splits (`tests/test_data.py::TestSequenceAwareSplit::test_every_block_lands_in_exactly_one_split`,
   `tests/test_data.py::TestSequenceAwareSplit::test_windows_never_span_two_blocks`).
   It needs at least three blocks and every block at least `seq_len + horizon` rows.
2. **Privacy release per split.** With a `privacy:` block, each split is its own
   release on its own `PrivacyBudget` (parallel composition over disjoint blocks); the
   train release fits the generalization classes that val and test then apply.
   `SplitData.epsilon_spent` reports the spend per split. A pipeline releases once; a
   second `process_blocks`, `fit` or `transform` raises `BudgetExhausted`
   (`tests/test_data.py::TestSequenceAwareSplit::test_each_split_spends_its_own_full_budget`,
   `tests/test_data.py::TestSequenceAwareSplit::test_no_further_release_after_process_blocks`,
   `tests/test_data.py::TestSequenceAwareSplit::test_a_pipeline_never_releases_twice_on_any_pair_of_paths`).
   Without a block no stage runs.
3. **Signals per block.** `TransactionPipeline.compute_signals` derives the four
   signals below inside each block, from the released columns, so nothing crosses a
   block joint (`tests/test_data.py::TestSequenceAwareSplit::test_rolling_signals_restart_at_every_block`).
4. **Scaler fit on train only.** `_fit_scaler` fits a `RobustScaler` on the train
   blocks' features; val and test are only transformed
   (`tests/test_data.py::TestSequenceAwareSplit::test_validation_outliers_are_not_absorbed_by_the_scaler`).
5. **Windows.** `_windows` slides `data.seq_len` rows inside each block. A window
   starts at row 0 up to `block_len - seq_len - horizon` (`data/labels.py::window_starts`),
   so the `model.horizon` rows after every window's last row are inside the block and
   the forecast label is always defined.

`SplitData` holds `train`, `val` and `test` plus `feature_names`, `scaler_center`,
`scaler_scale`, `epsilon_spent` and `privacy_mode`. Each `WindowSet` carries `X`
`[N, T, F]` float32 and the labels `y_risk`, `y_forecast`, `forecast_mask`, plus the
bookkeeping `block_index`, `onset_offset`, `end_row` and `anomaly_type` (the dominant
type in the window, or `none`) that the evaluation needs. `train/loop.py::prepare_data`
builds the model's `SplitData` from a config; the CLI, the notebook, `eval.run` and the
privacy sweep call it. The data gate (`eval/gates.py::_score_preset`) and the window
baseline of `eval.run` build a second, raw-column split by calling `process_blocks`
directly: its features are the generator's raw columns, it has no privacy stage, and
it uses the same seed and split fractions.

## Signals

`core/signals.py::SignalProcessor` holds no state: its static functions take arrays
(`process_sequence` is a classmethod that combines them).
`data.pipeline.SIGNAL_SOURCES` lists the four signals the pipeline derives and the
raw column each reads. They are applied per block, and the first rows of every block
are the warm-up.

| Signal | Source column | Definition |
| --- | --- | --- |
| `volume_volatility` | `volume` | `core.signals.SignalProcessor.volatility`: the population standard deviation of the 20 rows before row i. Row 0 is 0; until 20 rows exist the window is shorter. It never reads row i or later (DGNTY-32, `tests/test_core.py::TestSignalProcessor::test_volatility_never_reads_the_future`). |
| `volatility` | `price` | The same function on the price level. |
| `momentum` | `price` | `core.signals.SignalProcessor.price_momentum`: `(x[i] - x[i-10]) / x[i-10]` for `i >= 10`, and 0 before. |
| `directional_change` | `price` | `core.signals.SignalProcessor.directional_change`: the relative change from the previous row; 1 above `+0.01`, -1 below `-0.01`, else 0. Row 0 is 0. |

**Configured features that are not produced are dropped.** `TransactionPipeline`
keeps only the names in `data.features` that are columns after `compute_signals`,
in the configured order, and stores them as `available_features`. `price_change`,
`regime` and `entropy` are in the `DataConfig` default list (`config/base.yaml` names
the first two) but are never derived, so they vanish.
`train/loop.py::fit` sizes the model from `SplitData.feature_names`, so
`model.input_size` in YAML is ignored; if a model's input width surprises you, check
`available_features` first. `SignalProcessor` also defines `entropy`,
`regime_detection` and `process_sequence`; the pipeline does not call them.

## Backbone

`models/backbone/hybrid.py::DignityBackbone` maps `x [B, T, F]` to
`(context [B, H], attn_weights [B, T])` with `H = model.hidden_size`, for an odd
`model.cnn_kernel_size` (every shipped config). An even kernel adds one step per
convolution, so `attn_weights` is `[B, T+2]`:

1. `CNN1D`: two convolution layers of kernel `model.cnn_kernel_size` over time,
   `[B, T, F] -> [B, T, H]` for an odd kernel.
2. `StackedLSTM`: `model.n_layers` unidirectional layers, `[B, T, H] -> [B, T, H]`.
3. `AdditiveAttention`: weights over the T steps and their weighted sum, the context.

`model.dropout` applies inside the CNN, between LSTM layers (only when
`model.n_layers > 1`), after the LSTM and inside the head.
`tests/test_models.py::TestBackboneComponents::test_dignity_backbone` pins the shapes.

## Heads

`models/dignity.py::Dignity` is the only model entry point: `Dignity(task=...)` joins
the backbone with one head, and `Dignity.forward` returns
`(predictions, attn_weights)` for every task. Every caller must unpack the tuple.

| Task | Head | `predictions` | Training loss |
| --- | --- | --- | --- |
| `risk` | `models/head/risk.py::RiskHead` | Sigmoid probabilities `[B, 1]` | `train/losses.py::risk_loss`: binary cross-entropy with `train.pos_weight` (default: train negatives over positives) |
| `forecast` | `models/head/forecast.py::ForecastHead` | Onset probabilities `[B, horizon]`, `P(onset within h rows)` for h = 1..H, never decreasing in h by construction | `train/losses.py::forecast_loss`: masked per-horizon binary cross-entropy with `train.pos_weight` (default: negatives over positives among the unmasked train labels) |
| `policy` | `models/head/policy.py::PolicyHead` | Logits `[B, 3]` over `models.head.policy.ACTIONS` = `allow`, `review`, `block` | `train/losses.py::expected_cost_loss`: softmax-weighted cost table |

Pinned by `tests/test_models.py::TestTaskHeads::test_forecast_probabilities_never_decrease_with_horizon`
and `tests/test_models.py::TestTaskHeads::test_policy_head_returns_raw_logits_not_probabilities`.

`ForecastHead` uses the cumulative-hazard form `1 - exp(-cumsum(softplus(z)))`, which
is what makes the horizons non-decreasing. `PolicyHead` refuses an action count other
than `len(ACTIONS)`, because the cost table prices exactly those three.

The decision rule is outside the model. `models/head/policy.py::decide_actions` takes
the argmax over `ACTIONS`, then enforces the cap: only the `review_capacity` most
review-likely windows keep `review`; the others go to their next-best action,
`allow` or `block`. `block` never counts against the review budget
(`tests/test_models.py::TestDecisionRule::test_reviews_over_analyst_capacity_go_to_their_next_best_action`).

## Training and evaluation

`train/loop.py::fit(config, data, seed, run_dir, resume_from=None)` is the one
training loop; `dignity-train` (`train/cli.py::main`), the notebook and `eval.run`
all call it. It returns a `train.loop.FitResult`: `best_path`, `history`,
`best_metric` and `meta`.

- **Optimizer.** AdamW with a cosine schedule over `train.epochs`; gradient clipping
  at `train.gradient_clip`. `train.use_amp` only takes effect on CUDA.
- **AMP and the risk loss.** The risk head ends in a sigmoid and the loss is
  binary cross-entropy on probabilities, which PyTorch refuses inside autocast, so `train/engine.py::train_epoch`
  computes the loss outside autocast in float32. A CPU run proves nothing about this;
  `tests/test_train.py::TestTrainEpoch::test_amp_risk_training_runs_on_gpu` covers it
  and skips without CUDA.
- **Selection.** `best.pt` is the epoch with the highest validation value of
  `train.select_metric` (`pr_auc`, `auroc` or `recall_at_1pct_fpr`), on a strict
  improvement (`tests/test_train.py::TestFit::test_best_checkpoint_is_the_epoch_with_highest_val_pr_auc`).
  For forecast the validation metric is the mean over the horizons that hold both
  classes. Policy ignores `train.select_metric`, `train.pos_weight` and the chance
  guard: it selects on validation `neg_cost`, the negative mean cost of the capped
  decisions.
- **Chance guard.** At `train.guard_epochs`, if validation AUROC is below
  `train.chance_guard_auroc`, `fit` warns with `train.loop.ChanceGuardWarning`
  (`tests/test_train.py::TestFit::test_chance_guard_warns_once_when_val_auroc_stays_below_floor`).
  Forecast reads the last horizon; policy has no AUROC. No warning is raised when
  `train.chance_guard_auroc` is null, when the run ends before `train.guard_epochs`, or
  (forecast) when the last horizon's validation column holds one class.
- **Early stopping.** After `train.patience` epochs without improvement `fit` stops;
  `None` disables it
  (`tests/test_train.py::TestFit::test_early_stopping_halts_after_patience_epochs_without_improvement`).
  `last.pt` is written every `train.save_interval` epochs and on the final or stopping
  epoch. `--resume` restores the model, optimizer and grad-scaler state; resuming across
  selection metrics raises `ValueError`.
- **Warm start.** The shipped policy config sets `train.init_backbone_from` to a risk
  `best.pt`; the key is read only for `task: policy` and is skipped on `--resume`.
  `train/loop.py::load_backbone` copies only the `backbone.*` tensors and raises
  `ValueError` when the features, the scaler, the privacy block, the data block, the
  set of backbone tensors or any tensor shape differs. Without a `privacy:` block any
  difference in the scaler raises; under one the scaler may differ by up to 25% of the
  run's scale, because system-mode noise is unseeded
  (`tests/test_train.py::TestLoadBackbone::test_warm_start_copies_the_risk_backbone_and_leaves_the_head_alone`;
  the refusals are `tests/test_train.py::TestLoadBackbone`).

`python -m eval.run` (`eval/run.py::main`) trains through `fit`, scores the test split
and writes the results. The test split is for reporting; checkpoint selection, the
forecast alarm threshold and the baseline policy's thresholds all use validation.

| Flag | Meaning |
| --- | --- |
| `--config PATH` or `--smoke` | A full run of a config, or the CI-sized `config/smoke.yaml`. Exactly one is required. |
| `--seed tuning` or `--seed eval` | Which `eval` seed drives the generator. `--smoke` never runs on the evaluation seed. |
| `--data-seed N` | Needs `--seed tuning`; N is any tuning seed (1000 to 1009) and replaces the generator seed, so a rehearsal can run on a seed other than the config's. The three tasks train under their own run directory `eval_tuning_<N>`; the privacy sweep directories are not seed-qualified, so give each data seed its own `train.checkpoint_dir` when sweeping. |
| `--tasks risk,forecast,policy` | The default. Unknown names exit 2. Forecast and policy also fit risk, because their baseline is today's risk score and each policy seed warm-starts from the same seed's risk model, but the risk result file is written only when `risk` is in `--tasks`. |
| `--privacy-sweep` | Also trains risk with this run's config at each `eval.epsilons` value, taking the privacy block and the features of `config/train_risk.yaml`, with its Laplace epsilons rescaled to that total, public bounds and seeded noise (`privacy.deterministic_rng`, not DP-grade). Trains a non-private reference per replicate, then writes the privacy results and the curve (`docs/img/privacy_curve.png` by default). |

Each result file `eval.run` writes (risk, forecast, policy and, with
`--privacy-sweep`, privacy) has the keys `config_hash`, `seed`, `git_sha`, `metrics` and
`baselines`, in `eval.out_dir`; charts go to `eval.img_dir`. The headline model is the
first of `eval.n_train_seeds` training seeds. The risk, forecast and policy files carry
`metrics.seed_spread` over every training seed, the headline included; the privacy file
carries `replicate_spread` per epsilon and `seed_spread` for its reference.
Confidence intervals bootstrap whole blocks, never single windows, because windows of
one block overlap (`eval/metrics.py::bootstrap_ci`, DGNTY-21).

| Task | Metrics (`eval/metrics.py`) | Baselines (`eval/baselines.py`) |
| --- | --- | --- |
| Risk | `risk_metrics`: AUROC, PR-AUC, recall at 1% false-positive rate, Brier score, prevalence, recall per anomaly type at the test split's 1%-false-positive point; also scored per block, a block's score being its highest window score | `majority`; `window_logistic` on the raw columns' window statistics; `block_tree` on per-block statistics, trained on the noisy `block_label` |
| Forecast | `forecast_metrics`: PR-AUC per horizon on unmasked windows; `lead_times`: rows between the first alarm and each onset; `paired_forecast_gain`: seed-matched average-precision gain over the risk-score baseline, block bootstrap; `detection_gain`: share of onsets warned versus a random score, block bootstrap | `risk_score_forecast`: today's risk score used at every horizon |
| Policy | `policy_metrics`: expected cost, review rate, block rate, anomaly recall, false-block rate | `threshold_policy`: two thresholds on the risk score, tuned on validation under the review budget |

`metrics.model_gate` in the risk file (`eval/metrics.py::model_gate`) holds a window gate,
which compares the model's PR-AUC with `majority` and `window_logistic`, and a block
gate, which compares the block-level PR-AUC with `block_tree`. A gate passes when the
model's PR-AUC and its confidence interval's lower bound both exceed the best
baseline's PR-AUC, and `model_gate` passes when both gates do.

## Checkpoint and export

`train/checkpoint.py` writes self-describing checkpoints: a plain dict readable with
`torch.load(weights_only=True)` holding `model_state`, `optimizer_state`,
`grad_scaler_state`, `meta`, `epoch`, `metrics` and `best_metric`. `meta` is stored as
a plain dict of the fields of `train.checkpoint.CheckpointMeta`: `format_version`,
`task`, `input_size`, `feature_names`, `head_kwargs`, `scaler_center`, `scaler_scale`
and `config` (the whole run config); `load_checkpoint` returns it as a `CheckpointMeta`
(`tests/test_checkpoint.py::TestCheckpoint::test_checkpoint_alone_rebuilds_an_identical_model`).
`train.checkpoint.FORMAT_VERSION` is 2. `build_model_from_meta` rebuilds the model
from `meta` alone.

Any checkpoint with another `format_version`, including every checkpoint written
before v1.0, and any file `torch.load(weights_only=True)` rejects as an unsafe pickle,
raises
`train.checkpoint.UnsupportedCheckpointError` (DGNTY-35); retrain with `dignity-train`.

`export/to_onnx.py` exports and verifies:

- `export_to_onnx` writes opset 13 with input `input` `[batch, sequence, features]`
  and outputs `predictions` and `attention_weights`; the batch axis of every tensor and
  the sequence axis of the input and `attention_weights` are dynamic.
- `verify_onnx_export` runs the ONNX model and compares predictions and attention
  weights with PyTorch.
- `export.to_onnx.main` (`dignity-export`) traces at `(1, data.seq_len, input_size)`
  from the checkpoint. It returns 0 on success, 1 when verification
  fails and 3 for an unsupported checkpoint, with the message on stderr
  (`tests/test_export.py::TestExportCli::test_failed_verification_exits_nonzero`,
  `tests/test_export.py::TestExportCli::test_pre_v1_checkpoint_exits_3_with_a_clear_message`).

The ONNX model consumes scaled windows of the pipeline's released inputs, in the
pipeline's order: privacy release, then signals, then
`(x - meta.scaler_center) / meta.scaler_scale` over the columns in
`meta.feature_names` order. A checkpoint trained without a `privacy:` block therefore
needs only the signals and the scaler. A checkpoint trained under a `privacy:` block
(both shipped privacy configs) was fitted to released inputs: with `transform: log1p`,
`volume` is in log1p units, clipped to the block's `bounds` and then noised, and the
scaler's centre and scale are in those units. The `generalize` columns were binned with
classes fitted on the train split, and the checkpoint does not store them. So that
release cannot be reproduced from the checkpoint, and v1.0 ships no way to prepare new
private inputs for an exported model. **v1.0 has no inference-time privacy release**
(DGNTY-27): the privacy stage runs only in the pipeline's training-time paths
(`process_blocks`, `fit`, `transform`, `fit_transform` and `process`), and nothing in
the export path adds noise to or generalizes a live input. The risk and forecast
models emit probabilities; the policy model emits logits, so a consumer applies a
softmax and then `models.head.policy.decide_actions`.
