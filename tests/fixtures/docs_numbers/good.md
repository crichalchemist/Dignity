# Good fixture

Numbers outside a results zone are not checked, so 12 and 0.99 are fine here.
The 95% CI is prose, and so is this line.

<!--results-start-->0.84 <!--num results.json metrics.pr_auc.value .2f-->
84.1% <!--num results.json metrics.pr_auc.value .1%--> and 0.6 <!--num results.json metrics.pr_auc_by_horizon.1 .1f--> on one line
| `recall_at_1pct_fpr` | `results/v1.0/risk.json` | 0.84 <!--num results.json metrics.pr_auc.value .2f--> [0.80 <!--num results.json metrics.pr_auc.ci_low .2f-->, 0.87 <!--num results.json metrics.pr_auc.ci_high .2f-->] |
| `pr_auc_by_horizon.1` | 0.65 <!--num results.json forecast.pr_auc_by_horizon.1.value .2f--> |
| `policy.review_budget` | n/a <!--num results.json metrics.lead_time .1f--> |
| 1.0 <!--num results.json points.0.epsilon .1f--> | 10 <!--num results.json points.1.epsilon d--> |
This marker is followed by a comment: 0.84 <!--num results.json metrics.pr_auc.value .2f--><!-- note 7 -->
0.7 <!--num results.json forecast.pr_auc_by_horizon.0.value .1f--><!--results-end-->
