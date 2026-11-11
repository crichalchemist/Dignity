# Dignity Core: Sequence Models for Transaction Risk

<!--provisional-->**Provisional: numbers are from the tuning seed; the final evaluation has not run.**

[![CI](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml/badge.svg)](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Detect, warn early, decide: a CNN-LSTM-attention backbone with three interchangeable heads scores transaction risk on synthetic data, and each head is reported next to the baseline it has to beat.

## What it does

- **Detect**: is an anomaly active in this window? The `risk` head, scored by PR-AUC against `window_logistic` and `block_tree`.
- **Warn early**: will an anomaly start within the next 1 to `model.horizon` rows? The forecast head, scored by PR-AUC per horizon and by lead time against `risk_score_forecast`.
- **Decide**: allow, review or block each window under a 5% review budget (`policy.review_budget`). The policy head, scored by expected cost against `threshold_policy`.

Inputs can be made differentially private (ε is the per-released-value Laplace budget on `volume`; `fee_rate` and `tx_count` are released by k-anonymous generalization, which ε does not govern; `price` is not consumed; the shipped config is not a point on the curve; see the [threat model](docs/THREAT-MODEL.md)), but the shipped allocation leaves the risk model's PR-AUC not distinguishable from chance on the hard preset at every ε on the grid up to 10, and above chance at the two largest grid values, 30 and 100. It scores transaction risk and gives no advice on what to transact.

## Results

**Outcome.** These are the outcomes for the headline training seed. On risk, the window gate passes against `window_logistic` and the block gate passes against `block_tree`. On forecast, the model's PR-AUC interval clears `risk_score_forecast` at no horizon, its point estimate is below the baseline's at every horizon, the model's PR-AUC interval contains the prevalence at every horizon (the prevalence is the chance line of PR-AUC), the baseline's PR-AUC interval lies wholly above the prevalence at every horizon, the baseline's lower bound is above the model's point estimate at most horizons (more than half of them, not all), and the model warns ahead of no onset. On policy, the triage head's expected cost is higher than the threshold policy's, so the triage head does not beat the threshold policy. On the privacy curve, the risk model's PR-AUC is not distinguishable from chance at every grid value except the two largest, where it is above chance.

The numbers of the four `eval.run` results are tuning-seed scores. Intervals are 95% bootstrap intervals over test blocks; commands, baselines and caveats are in [docs/RESULTS.md](docs/RESULTS.md).

<!--results-start-->
| Metric | Model | Best baseline | File |
|---|---|---|---|
| `auroc` | 0.84 <!--num risk.json metrics.auroc.value .2f--> | `window_logistic` 0.68 <!--num risk.json baselines.window_logistic.auroc.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc` | 0.81 <!--num risk.json metrics.pr_auc.value .2f--> [0.72 <!--num risk.json metrics.pr_auc.ci_low .2f-->, 0.88 <!--num risk.json metrics.pr_auc.ci_high .2f-->] | `window_logistic` 0.55 <!--num risk.json baselines.window_logistic.pr_auc.value .2f--> | `results/v1.0/risk.json` |
| `recall_at_1pct_fpr` | 0.59 <!--num risk.json metrics.recall_at_1pct_fpr.value .2f--> | `window_logistic` 0.08 <!--num risk.json baselines.window_logistic.recall_at_1pct_fpr.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc` per block | 0.87 <!--num risk.json metrics.block_level.pr_auc.value .2f--> | `block_tree` 0.45 <!--num risk.json baselines.block_tree.pr_auc.value .2f--> | `results/v1.0/risk.json` |
| `pr_auc_by_horizon.4` | 0.0336 <!--num forecast.json metrics.pr_auc_by_horizon.4.value .4f--> | `risk_score_forecast` 0.0566 <!--num forecast.json baselines.risk_score_forecast.pr_auc_by_horizon.4.value .4f--> | `results/v1.0/forecast.json` |
| `lead_time.median` | n/a <!--num forecast.json metrics.lead_time.median .1f--> | `risk_score_forecast` 10.0 <!--num forecast.json baselines.risk_score_forecast.lead_time.median .1f--> | `results/v1.0/forecast.json` |
| `lead_time.n_missed` | 34 <!--num forecast.json metrics.lead_time.n_missed d--> | `risk_score_forecast` 32 <!--num forecast.json baselines.risk_score_forecast.lead_time.n_missed d--> | `results/v1.0/forecast.json` |
| `lead_time.n_onsets` | 34 <!--num forecast.json metrics.lead_time.n_onsets d--> | `risk_score_forecast` 34 <!--num forecast.json baselines.risk_score_forecast.lead_time.n_onsets d--> | `results/v1.0/forecast.json` |
| `expected_cost` | 1.20 <!--num policy.json metrics.expected_cost.value .2f--> [0.83 <!--num policy.json metrics.expected_cost.ci_low .2f-->, 1.59 <!--num policy.json metrics.expected_cost.ci_high .2f-->] | `threshold_policy` 1.07 <!--num policy.json baselines.threshold_policy.expected_cost.value .2f--> [0.72 <!--num policy.json baselines.threshold_policy.expected_cost.ci_low .2f-->, 1.46 <!--num policy.json baselines.threshold_policy.expected_cost.ci_high .2f-->] | `results/v1.0/policy.json` |
| `pr_auc` without privacy | 0.86 <!--num privacy.json metrics.reference.risk.pr_auc.value .2f--> [0.78 <!--num privacy.json metrics.reference.risk.pr_auc.ci_low .2f-->, 0.93 <!--num privacy.json metrics.reference.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.54 <!--num privacy.json baselines.window_logistic.reference.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
| `prevalence` | 0.27 <!--num privacy.json metrics.reference.risk.prevalence.value .2f--> | n/a | `results/v1.0/privacy.json` |
| `pr_auc` at ε 0.1 <!--num privacy.json metrics.points.0.epsilon g--> | 0.33 <!--num privacy.json metrics.points.0.risk.pr_auc.value .2f--> [0.22 <!--num privacy.json metrics.points.0.risk.pr_auc.ci_low .2f-->, 0.44 <!--num privacy.json metrics.points.0.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.26 <!--num privacy.json baselines.window_logistic.points.0.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
| `pr_auc` at ε 100 <!--num privacy.json metrics.points.6.epsilon g--> | 0.69 <!--num privacy.json metrics.points.6.risk.pr_auc.value .2f--> [0.57 <!--num privacy.json metrics.points.6.risk.pr_auc.ci_low .2f-->, 0.79 <!--num privacy.json metrics.points.6.risk.pr_auc.ci_high .2f-->] | `window_logistic` 0.36 <!--num privacy.json baselines.window_logistic.points.6.pr_auc.value .2f--> | `results/v1.0/privacy.json` |
<!--results-end-->

![Precision-recall curves of the risk model, window_logistic and majority on the test split](docs/img/risk_pr_curve.png)
![PR-AUC per forecast horizon for the model and risk_score_forecast with the prevalence of each horizon](docs/img/forecast_pr_auc_by_horizon.png)
![Privacy-versus-utility curve: risk PR-AUC at each epsilon with the non-private reference and the chance line](docs/img/privacy_curve.png)

The privacy baseline is `window_logistic` on the released windows. The shipped config is not a point on this curve: its ε of 1.0 is a grid value, but the curve runs on the hard preset with that preset's training settings.

## Quickstart

Python 3.10 or newer, from a fresh clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
```

The smoke preset runs on CPU and offline, and `tests/test_docs.py` executes this block:

```bash quickstart
python -m train.cli --config config/smoke.yaml
python -m export.to_onnx --checkpoint runs/smoke/train/best.pt --output dignity_risk.onnx
python -m eval.run --config config/smoke.yaml --seed tuning
```

The first command trains a risk model into `runs/smoke/train`. The second exports it to ONNX and verifies it against PyTorch, writing `dignity_risk.onnx` in the working directory. The third writes the result files to `runs/smoke/results` and the charts to `runs/smoke/img`. The smoke config keeps its outputs under `runs/smoke`; the full evaluation, and any config that leaves `eval.out_dir` and `eval.img_dir` at their defaults, writes `results/v1.0` and `docs/img` (see the [Colab runbook](docs/RESULTS.md#colab-runbook)). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data contract and heads.

## How this was built

Agent workflow rules live in `.claude/CLAUDE.md`. An adversarial review found that the original benchmark was trivially easy: a depth-3 tree on per-block statistics separated every block (DGNTY-10). The generator was rebuilt with difficulty presets, and every model now sits next to a baseline. Each finding became an issue, and each fix went to a PR and a CI run; [docs/HOW-IT-WAS-BUILT.md](docs/HOW-IT-WAS-BUILT.md) records the status of every one.

## Limitations

- Synthetic data only: `data.source` accepts only `synthetic`, and there is no result on real data.
- `fee_rate` and `tx_count` are uncalibrated.
- The triage costs (`policy.costs`) are a stated assumption in the config.
- Privacy: On the tuning seed, the shipped allocation leaves the risk model's PR-AUC not distinguishable from chance on the hard preset at every ε on the grid up to 10, and above chance at the two largest grid values, 30 and 100 ([curve](docs/PRIVACY.md#privacy-versus-utility-curve)). The scope of each release is in the [threat model](docs/THREAT-MODEL.md), and there is no inference-time privacy release path (DGNTY-27).

## Roadmap

Planned for v1.1, none of it in v1.0:

- Planned: fix `CryptoSource` (DGNTY-7, DGNTY-13, DGNTY-14, DGNTY-15).
- Planned: dispatch on `data.source` (DGNTY-41).
- Planned: the La Morgia pump-and-dump events as the first real-data result.
- Planned: on-chain calibration (DGNTY-44) and per-entity amount features for structuring typologies.
