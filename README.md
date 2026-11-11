# Dignity Core: Sequence Models for Transaction Risk

<!--provisional-->**Provisional: numbers are from the tuning seed; the final evaluation has not run.**

[![CI](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml/badge.svg)](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10_to_3.12-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Detect, warn early, decide: three heads on one CNN-LSTM-attention backbone score transaction risk on synthetic data, each against the baseline it has to beat.

## What it does

- **Detect**: is an anomaly active in this window? The `risk` head, scored by PR-AUC against `window_logistic` and `block_tree`.
- **Warn early**: will an anomaly start within the next 1 to `model.horizon` rows? The forecast head, scored by PR-AUC per horizon and by lead time against `risk_score_forecast`.
- **Decide**: allow, review or block each window under a 5% review budget (`policy.review_budget`). The policy head, scored by expected cost against `threshold_policy`.

It scores risk and gives no advice on what to transact.

## Results

**Verdict.** Risk clears both baselines; forecast and triage do not clear theirs; the privacy curve is at chance at every grid value except the two largest.

**Outcome.** On risk, the window gate passes against `window_logistic` and the block gate passes against `block_tree`. On forecast, the model's PR-AUC interval clears `risk_score_forecast` at no horizon, its point estimate is below the baseline's at every horizon, the model's PR-AUC interval contains the prevalence at every horizon, the baseline's PR-AUC interval lies wholly above the prevalence at every horizon, the baseline's lower bound is above the model's point estimate at most horizons, and the model warns ahead of no onset. On policy, the triage head's expected cost is higher than the threshold policy's, so the triage head does not beat the threshold policy.

The numbers of the four `eval.run` results are tuning-seed scores. They are for the `hard` preset (`config/synthetic_hard.yaml`). The headline training seed is the first of three training seeds. Intervals are 95% bootstrap intervals over test blocks. `pr_auc_by_horizon.N` scores an onset within the next N plus one rows; chance is the prevalence. Commands, baselines and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

<!--results-start-->
| Metric | Model | Baseline | File |
|---|---|---|---|
| `auroc` | 0.84 <!--num risk.json metrics.auroc.value .2f--> | `window_logistic` 0.68 <!--num risk.json baselines.window_logistic.auroc.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc` | 0.81 <!--num risk.json metrics.pr_auc.value .2f--> [0.72 <!--num risk.json metrics.pr_auc.ci_low .2f-->, 0.88 <!--num risk.json metrics.pr_auc.ci_high .2f-->] | `window_logistic` 0.55 <!--num risk.json baselines.window_logistic.pr_auc.value .2f--> | `results/v1.0/risk.json` |
| `recall_at_1pct_fpr` | 0.59 <!--num risk.json metrics.recall_at_1pct_fpr.value .2f--> | `window_logistic` 0.08 <!--num risk.json baselines.window_logistic.recall_at_1pct_fpr.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc` per block | 0.87 <!--num risk.json metrics.block_level.pr_auc.value .2f--> | `block_tree` 0.45 <!--num risk.json baselines.block_tree.pr_auc.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc_by_horizon.4` (five rows ahead; chance 0.0223 <!--num forecast.json metrics.prevalence_by_horizon.4 .4f-->) | 0.0336 <!--num forecast.json metrics.pr_auc_by_horizon.4.value .4f--> [0.0202 <!--num forecast.json metrics.pr_auc_by_horizon.4.ci_low .4f-->, 0.0635 <!--num forecast.json metrics.pr_auc_by_horizon.4.ci_high .4f-->] | `risk_score_forecast` 0.0566 <!--num forecast.json baselines.risk_score_forecast.pr_auc_by_horizon.4.value .4f--> | `results/v1.0/forecast.json` |
| `lead_time.median` | n/a <!--num forecast.json metrics.lead_time.median .1f--> | `risk_score_forecast` 10.0 <!--num forecast.json baselines.risk_score_forecast.lead_time.median .1f--> | `results/v1.0/forecast.json` |
| `lead_time.n_missed` | 34 <!--num forecast.json metrics.lead_time.n_missed d--> | `risk_score_forecast` 32 <!--num forecast.json baselines.risk_score_forecast.lead_time.n_missed d--> | `results/v1.0/forecast.json` |
| `lead_time.n_onsets` | 34 <!--num forecast.json metrics.lead_time.n_onsets d--> | `risk_score_forecast` 34 <!--num forecast.json baselines.risk_score_forecast.lead_time.n_onsets d--> | `results/v1.0/forecast.json` |
| `expected_cost` | 1.20 <!--num policy.json metrics.expected_cost.value .2f--> [0.83 <!--num policy.json metrics.expected_cost.ci_low .2f-->, 1.59 <!--num policy.json metrics.expected_cost.ci_high .2f-->] | `threshold_policy` 1.07 <!--num policy.json baselines.threshold_policy.expected_cost.value .2f--> [0.72 <!--num policy.json baselines.threshold_policy.expected_cost.ci_low .2f-->, 1.46 <!--num policy.json baselines.threshold_policy.expected_cost.ci_high .2f-->] | `results/v1.0/policy.json` |
| `pr_auc` without privacy (non-private, privacy feature set, not the headline model) | 0.86 <!--num privacy.json metrics.reference.risk.pr_auc.value .2f--> [0.78 <!--num privacy.json metrics.reference.risk.pr_auc.ci_low .2f-->, 0.93 <!--num privacy.json metrics.reference.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.54 <!--num privacy.json baselines.window_logistic.reference.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
| `pr_auc` at ε 0.1 <!--num privacy.json metrics.points.0.epsilon g--> | 0.33 <!--num privacy.json metrics.points.0.risk.pr_auc.value .2f--> [0.22 <!--num privacy.json metrics.points.0.risk.pr_auc.ci_low .2f-->, 0.44 <!--num privacy.json metrics.points.0.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.26 <!--num privacy.json baselines.window_logistic.points.0.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
| `pr_auc` at ε 100 <!--num privacy.json metrics.points.6.epsilon g--> | 0.69 <!--num privacy.json metrics.points.6.risk.pr_auc.value .2f--> [0.57 <!--num privacy.json metrics.points.6.risk.pr_auc.ci_low .2f-->, 0.79 <!--num privacy.json metrics.points.6.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.36 <!--num privacy.json baselines.window_logistic.points.6.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
<!--results-end-->

![Risk precision-recall curves against the baselines](docs/img/risk_pr_curve.png)
![Forecast PR-AUC per horizon against baseline and prevalence](docs/img/forecast_pr_auc_by_horizon.png)
![Privacy-versus-utility curve of risk PR-AUC at each epsilon](docs/img/privacy_curve.png)

The privacy baseline is `window_logistic` on the released windows. The reference row is a non-private model on the privacy feature set, not the headline model; the prevalence, the chance line, is 0.27 <!--num privacy.json metrics.reference.risk.prevalence.value .2f-->. The shipped config is not a point on this curve: its ε of 1.0 is a grid value, but the curve runs on the hard preset with that preset's training settings.

## Quickstart

Python 3.10 to 3.12 (CI runs Linux with 3.10 and 3.12; `numpy<2` caps the range). On Intel macOS PyPI has no torch 2.5 or newer wheel, so install PyTorch from conda-forge first and skip the `pip install torch` line, as the project's macOS checkout does. From a fresh clone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
```

Then:

```bash quickstart
python -m train.cli --config config/smoke.yaml
python -m export.to_onnx --checkpoint runs/smoke/train/best.pt --output dignity_risk.onnx
python -m eval.run --config config/smoke.yaml --seed tuning
```

They train a risk model into `runs/smoke/train`, export and verify `dignity_risk.onnx`, and write results to `runs/smoke/results` and charts to `runs/smoke/img`. The full evaluation, and any config that keeps the default `eval.out_dir` and `eval.img_dir`, writes `results/v1.0` and `docs/img` (see the [Colab runbook](docs/RESULTS.md#colab-runbook)). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How this was built

Agent workflow rules live in `.claude/CLAUDE.md`. An adversarial review found that the original benchmark was trivially easy: a depth-3 tree on per-block statistics separated every block (DGNTY-10). The generator was rebuilt with difficulty presets; DGNTY-10 closes with the final evaluation. Each finding became an issue (DGNTY ids are rows of [docs/HOW-IT-WAS-BUILT.md](docs/HOW-IT-WAS-BUILT.md); the tracker is private), and each fix went to a PR and a CI run; that page records the status of every one, and DGNTY-8, DGNTY-27 and DGNTY-40 are still open (see its Status column).

## Limitations

- Synthetic data only: `data.source` accepts only `synthetic`, and there is no result on real data.
- `fee_rate` and `tx_count` are uncalibrated.
- The triage costs (`policy.costs`) are a stated assumption in the config.
- Privacy: On the tuning seed, the shipped allocation leaves the risk model's PR-AUC not distinguishable from chance on the hard preset at every ε on the grid up to 10, and above chance at the two largest grid values, 30 and 100 ([curve](docs/PRIVACY.md#privacy-versus-utility-curve)), although inputs can be made differentially private (ε is the per-released-value Laplace budget on `volume`; `fee_rate` and `tx_count` are released by k-anonymous generalization, which ε does not govern; see the [threat model](docs/THREAT-MODEL.md)). The intervals are one noise draw and one training seed, the two largest grid values are not offered as private settings, and k-anonymity holds for the train release only. There is no inference-time privacy release path (DGNTY-27).

## Roadmap

Planned for v1.1, none of it in v1.0:

- Planned: fix the real-data source `CryptoSource` (DGNTY-7, DGNTY-13, DGNTY-14, DGNTY-15).
- Planned: a dataset adapter layer that dispatches on `data.source` (DGNTY-41).
- Planned: a labelled manipulation-event dataset, the public pump-and-dump events (La Morgia et al., 2020), as the first real-data result (DGNTY-42).
- Planned: per-block aggregates from a public chain dataset (DGNTY-44).
- Planned, not yet tracked: per-entity amount features for structuring typologies.
