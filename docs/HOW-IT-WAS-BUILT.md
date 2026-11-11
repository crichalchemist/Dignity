# How it was built

This page records how Dignity v1.0 was reviewed, what the review found and how each finding was resolved. It is a trail for a reader who wants to check the work, not a results page: it quotes no metric. Results are in [RESULTS](RESULTS.md).

## Agent conventions

Agent-facing build and workflow rules live in `.claude/CLAUDE.md`, the single source, and the Gemini and agents dot-folders carry symlinks to it. Those symlinks are local files and are not tracked, so only the source is named as a path here. The public files at the repository root stay written for human contributors.

The excerpt below is copied verbatim from the Conventions and "Behavior that will surprise you" sections of that file. It shows the kind of rule the agents work under: configuration is dataclass-backed, privacy bounds are public, and a configured feature can be dropped silently.

<!--excerpt .claude/CLAUDE.md-->
```text
- Type hints use modern union syntax (`np.ndarray | None`) — `target-version = "py310"`.
- A `privacy:` block in YAML is optional and validated at load (`PrivacyConfig`). No block =
  no privacy stage. `bounds` are public parameters chosen a priori — never derive them from data.
- `privacy.deterministic_rng: true` (+ `privacy_seed=`) is the seeded experiment mode; shipped
  configs stay on SystemRandom.
- **Configured features are silently dropped.** `TransactionPipeline.fit` keeps only
  `[f for f in self.features if f in df.columns]` and stores it as `self.available_features`.
- **Synthetic data is an event record of blocks, interleaved.** `SyntheticGenerator` writes one
  row per step with `block_id`, `is_anomalous`, `anomaly_type`, `anomaly_id` and `block_label`;
  normal and anomalous blocks are drawn independently, so they interleave. Labels come from the
```
<!--/excerpt-->

## The adversarial review

Five reviewers (code, training, docs, synthetic data, datasets) examined the repository, and a skeptic then re-verified every finding independently. The review page states that 42 raw findings merged into 31 defects, that all 31 were confirmed (none refuted, none left uncertain) and that they were filed as DGNTY-7 to DGNTY-37. Eight recommendation items (DGNTY-38 to DGNTY-45) cover synthetic-data quality and public datasets. DGNTY-1 to DGNTY-6 were known defects filed before the review. The review page gives no skeptic verdict for them or for the recommendation items, so the table says "not recorded" there.

The tracker is Plane, a private tracker that readers of this repository cannot open. This page is therefore self-contained: the table paraphrases each finding in one line (it does not quote the tracker), and the DGNTY-n id is the one that merged PR titles carry.

Two stories from the review are worth telling plainly.

- The original benchmark was trivially easy. The review found that a depth-3 tree on per-block statistics separated every block, so the sequence model was never actually tested (DGNTY-10). The generator was rebuilt with difficulty presets, and the evaluation now pairs every model with a baseline. The finding closes only once the final evaluation has run.
- The review proposed levers for raising privacy utility, and the skeptic caught two of the proposed steps as unsound: widening the volume bounds makes the noise larger, and a fixed signal-to-noise target is unreachable at those bounds. Both were dropped from the recommendation (DGNTY-40).

## Finding to fix

The table has one row per finding of the review's 45 filed items. Status is one of four values: fixed, open (maintainer decision), closes with the final evaluation (M7), deferred (v1.1). A PR link on a row that is not fixed shows work that addressed part of the item, not that it is done. The CI run link is the push run of the PR's merge commit on the integration branch; a row whose PR has no merge commit yet says "not recorded".

Three items stay open by the maintainer's choice: DGNTY-8 (the shipped privacy block against the curve outcome), DGNTY-27 (there is no inference-time privacy release path in v1.0) and DGNTY-40 (the curve shipped, the per-anomaly-type privacy gate did not). DGNTY-46, DGNTY-47 and DGNTY-48 were filed while the work was in progress, they are open, and they are not rows of this table.

| Finding | Area | Skeptic verdict | DGNTY | Status | PR | CI run |
| --- | --- | --- | --- | --- | --- | --- |
| The early-warning task crashed in its training step on a label and output shape mismatch | training | not recorded | DGNTY-1 | fixed | [#31](https://github.com/crichalchemist/Dignity/pull/31) [#33](https://github.com/crichalchemist/Dignity/pull/33) | [run for #31](https://github.com/crichalchemist/Dignity/actions/runs/42612468724) [run for #33](https://github.com/crichalchemist/Dignity/actions/runs/42714693737) |
| The triage task crashed in its training step because its head returned a tuple | training | not recorded | DGNTY-2 | fixed | [#32](https://github.com/crichalchemist/Dignity/pull/32) [#33](https://github.com/crichalchemist/Dignity/pull/33) | [run for #32](https://github.com/crichalchemist/Dignity/actions/runs/42707762444) [run for #33](https://github.com/crichalchemist/Dignity/actions/runs/42714693737) |
| A checkpoint written by the training command failed to load in export (feature count mismatch) | export | not recorded | DGNTY-3 | fixed | [#22](https://github.com/crichalchemist/Dignity/pull/22) [#23](https://github.com/crichalchemist/Dignity/pull/23) | [run for #22](https://github.com/crichalchemist/Dignity/actions/runs/42547586587) [run for #23](https://github.com/crichalchemist/Dignity/actions/runs/42548635331) |
| The export console script pointed at a function that did not exist | export | not recorded | DGNTY-4 | fixed | [#23](https://github.com/crichalchemist/Dignity/pull/23) | [run for #23](https://github.com/crichalchemist/Dignity/actions/runs/42548635331) |
| The training step crashed on batches that carry no labels | training | not recorded | DGNTY-5 | fixed | [#21](https://github.com/crichalchemist/Dignity/pull/21) | [run for #21](https://github.com/crichalchemist/Dignity/actions/runs/42546616622) |
| Config defaults and test timeouts did not do what they claimed (checkpoint folder, inert timeouts) | config | not recorded | DGNTY-6 | fixed | [#15](https://github.com/crichalchemist/Dignity/pull/15) [#17](https://github.com/crichalchemist/Dignity/pull/17) | [run for #15](https://github.com/crichalchemist/Dignity/actions/runs/42538310125) [run for #17](https://github.com/crichalchemist/Dignity/actions/runs/42540734883) |
| The real-data source collapsed frames because it misread a timestamp unit | data | confirmed | DGNTY-7 | deferred (v1.1) | none | not recorded |
| The shipped privacy block drove the risk model to chance and the training command still saved it as best. On the tuning seed the shipped allocation leaves the risk model's PR-AUC not distinguishable from chance on the hard preset at every grid epsilon except the two largest, where it is above chance but below the non-private reference; see [Privacy-versus-utility curve](PRIVACY.md#privacy-versus-utility-curve). | privacy | confirmed | DGNTY-8 | open (maintainer decision) | [#36](https://github.com/crichalchemist/Dignity/pull/36) [#37](https://github.com/crichalchemist/Dignity/pull/37) | [run for #36](https://github.com/crichalchemist/Dignity/actions/runs/42765940079) [run for #37](https://github.com/crichalchemist/Dignity/actions/runs/42815370353) |
| Validation reported only loss and accuracy at a fixed threshold: no ranking metric, baseline or test split | evaluation | confirmed | DGNTY-9 | fixed | [#29](https://github.com/crichalchemist/Dignity/pull/29) [#30](https://github.com/crichalchemist/Dignity/pull/30) | [run for #29](https://github.com/crichalchemist/Dignity/actions/runs/42578189720) [run for #30](https://github.com/crichalchemist/Dignity/actions/runs/42597188123) |
| The synthetic risk task was trivially separable | data | confirmed | DGNTY-10 | closes with the final evaluation (M7) | none | not recorded |
| The synthetic normal process was unrealistic (Gaussian, no clustering, pinned volume, constant fee) | data | confirmed | DGNTY-11 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) [#24](https://github.com/crichalchemist/Dignity/pull/24) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) [run for #24](https://github.com/crichalchemist/Dignity/actions/runs/42552101263) |
| Four guides documented an API that did not exist | docs | confirmed | DGNTY-12 | closes with the final evaluation (M7) | [#41](https://github.com/crichalchemist/Dignity/pull/41) | not recorded |
| The timestamp normaliser misclassified some units and mixed columns | data | confirmed | DGNTY-13 | deferred (v1.1) | none | not recorded |
| The block resampler dropped columns and failed without high and low values | data | confirmed | DGNTY-14 | deferred (v1.1) | none | not recorded |
| Real-data frames produced a different feature count from synthetic ones | data | confirmed | DGNTY-15 | deferred (v1.1) | none | not recorded |
| Synthetic anomalies sat at fixed rows with fixed magnitude and duration | data | confirmed | DGNTY-16 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) [#25](https://github.com/crichalchemist/Dignity/pull/25) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) [run for #25](https://github.com/crichalchemist/Dignity/actions/runs/42553746645) |
| Synthetic fee rate was almost constant, which collapsed generalization and left the scaler with zero spread | data | confirmed | DGNTY-17 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) |
| The block label was broadcast to rows, the anomaly type was discarded and correctness depended on the block length | data | confirmed | DGNTY-18 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) [#25](https://github.com/crichalchemist/Dignity/pull/25) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) [run for #25](https://github.com/crichalchemist/Dignity/actions/runs/42553746645) |
| The synthetic generator seeded the global random state | data | confirmed | DGNTY-19 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) |
| Privacy-on runs were not reproducible under a seed and nothing warned | privacy | confirmed | DGNTY-20 | fixed | [#35](https://github.com/crichalchemist/Dignity/pull/35) | [run for #35](https://github.com/crichalchemist/Dignity/actions/runs/42753203935) |
| Overlapping windows overstated the sample size of the reported splits | evaluation | confirmed | DGNTY-21 | fixed | [#28](https://github.com/crichalchemist/Dignity/pull/28) | [run for #28](https://github.com/crichalchemist/Dignity/actions/runs/42574940294) |
| Resuming training reset the best metric and could overwrite the best checkpoint | training | confirmed | DGNTY-22 | fixed | [#22](https://github.com/crichalchemist/Dignity/pull/22) | [run for #22](https://github.com/crichalchemist/Dignity/actions/runs/42547586587) |
| Several config knobs were silently ignored | config | confirmed | DGNTY-23 | fixed | [#21](https://github.com/crichalchemist/Dignity/pull/21) [#20](https://github.com/crichalchemist/Dignity/pull/20) | [run for #21](https://github.com/crichalchemist/Dignity/actions/runs/42546616622) [run for #20](https://github.com/crichalchemist/Dignity/actions/runs/42544661411) |
| Export discarded its verify result and exited zero on a mismatch | export | confirmed | DGNTY-24 | fixed | [#23](https://github.com/crichalchemist/Dignity/pull/23) | [run for #23](https://github.com/crichalchemist/Dignity/actions/runs/42548635331) |
| The export benchmark used a hard-coded input shape | export | confirmed | DGNTY-25 | fixed | [#23](https://github.com/crichalchemist/Dignity/pull/23) | [run for #23](https://github.com/crichalchemist/Dignity/actions/runs/42548635331) |
| Sequence creation failed on two-dimensional targets | data | confirmed | DGNTY-26 | fixed | [#26](https://github.com/crichalchemist/Dignity/pull/26) | [run for #26](https://github.com/crichalchemist/Dignity/actions/runs/42565042769) |
| A fitted privacy pipeline could not transform further data, so there is no inference path | privacy | confirmed | DGNTY-27 | open (maintainer decision) | [#40](https://github.com/crichalchemist/Dignity/pull/40) [#34](https://github.com/crichalchemist/Dignity/pull/34) | [run for #40](https://github.com/crichalchemist/Dignity/actions/runs/42971898695) [run for #34](https://github.com/crichalchemist/Dignity/actions/runs/42741432391) |
| The raw-input coverage test could not see the price column | tests | confirmed | DGNTY-28 | fixed | [#36](https://github.com/crichalchemist/Dignity/pull/36) | [run for #36](https://github.com/crichalchemist/Dignity/actions/runs/42765940079) |
| The agent guide claimed a doc drift was fixed while most of it was still live | docs | confirmed | DGNTY-29 | fixed | [#16](https://github.com/crichalchemist/Dignity/pull/16) | [run for #16](https://github.com/crichalchemist/Dignity/actions/runs/42539967316) |
| Key paths had no tests (data source, export verify, checkpoint round trip, CLI, CPU training step, generator difficulty) | tests | confirmed | DGNTY-30 | fixed | [#39](https://github.com/crichalchemist/Dignity/pull/39) [#30](https://github.com/crichalchemist/Dignity/pull/30) | [run for #39](https://github.com/crichalchemist/Dignity/actions/runs/42949003267) [run for #30](https://github.com/crichalchemist/Dignity/actions/runs/42597188123) |
| Label detection in the engine misfired on batches of two | training | confirmed | DGNTY-31 | fixed | [#21](https://github.com/crichalchemist/Dignity/pull/21) | [run for #21](https://github.com/crichalchemist/Dignity/actions/runs/42546616622) |
| Volatility warm-up looked ahead | signals | confirmed | DGNTY-32 | fixed | [#19](https://github.com/crichalchemist/Dignity/pull/19) | [run for #19](https://github.com/crichalchemist/Dignity/actions/runs/42543678002) |
| The gradient scaler was recreated every epoch and not checkpointed | training | confirmed | DGNTY-33 | fixed | [#22](https://github.com/crichalchemist/Dignity/pull/22) [#21](https://github.com/crichalchemist/Dignity/pull/21) | [run for #22](https://github.com/crichalchemist/Dignity/actions/runs/42547586587) [run for #21](https://github.com/crichalchemist/Dignity/actions/runs/42546616622) |
| Generalization bin edges were fitted on train and validation together | privacy | confirmed | DGNTY-34 | fixed | [#34](https://github.com/crichalchemist/Dignity/pull/34) | [run for #34](https://github.com/crichalchemist/Dignity/actions/runs/42741432391) |
| Protected old checkpoints were degenerate and must not be exported | export | confirmed | DGNTY-35 | deferred (v1.1) | [#23](https://github.com/crichalchemist/Dignity/pull/23) | [run for #23](https://github.com/crichalchemist/Dignity/actions/runs/42548635331) |
| README and docs hygiene: wrong counts, missing folders, wrong checkpoint path, broken links | docs | confirmed | DGNTY-36 | fixed | [#40](https://github.com/crichalchemist/Dignity/pull/40) [#18](https://github.com/crichalchemist/Dignity/pull/18) | [run for #40](https://github.com/crichalchemist/Dignity/actions/runs/42971898695) [run for #18](https://github.com/crichalchemist/Dignity/actions/runs/42541872548) |
| The privacy doc pointed auditors at the wrong test file | docs | confirmed | DGNTY-37 | fixed | [#34](https://github.com/crichalchemist/Dignity/pull/34) | [run for #34](https://github.com/crichalchemist/Dignity/actions/runs/42741432391) |
| Recommendation: replace the generator core with calibrated, clustered dynamics | data | not recorded | DGNTY-38 | fixed | [#27](https://github.com/crichalchemist/Dignity/pull/27) | [run for #27](https://github.com/crichalchemist/Dignity/actions/runs/42570160464) |
| Recommendation: redesign anomaly types, per-row flags and difficulty presets | data | not recorded | DGNTY-39 | fixed | [#27](https://github.com/crichalchemist/Dignity/pull/27) [#25](https://github.com/crichalchemist/Dignity/pull/25) | [run for #27](https://github.com/crichalchemist/Dignity/actions/runs/42570160464) [run for #25](https://github.com/crichalchemist/Dignity/actions/runs/42553746645) |
| Recommendation: re-tune the privacy stage for utility and gate it per anomaly type | privacy | not recorded | DGNTY-40 | open (maintainer decision) | [#38](https://github.com/crichalchemist/Dignity/pull/38) [#37](https://github.com/crichalchemist/Dignity/pull/37) [#36](https://github.com/crichalchemist/Dignity/pull/36) | [run for #38](https://github.com/crichalchemist/Dignity/actions/runs/42920455265) [run for #37](https://github.com/crichalchemist/Dignity/actions/runs/42815370353) [run for #36](https://github.com/crichalchemist/Dignity/actions/runs/42765940079) |
| Recommendation: fix the real-data source and add a dataset adapter layer | data | not recorded | DGNTY-41 | deferred (v1.1) | [#20](https://github.com/crichalchemist/Dignity/pull/20) | [run for #20](https://github.com/crichalchemist/Dignity/actions/runs/42544661411) |
| Recommendation: a labelled manipulation-event dataset | data | not recorded | DGNTY-42 | deferred (v1.1) | none | not recorded |
| Recommendation: an anti-money-laundering dataset adapter | data | not recorded | DGNTY-43 | deferred (v1.1) | none | not recorded |
| Recommendation: per-block aggregates from a public chain dataset | data | not recorded | DGNTY-44 | deferred (v1.1) | none | not recorded |
| Recommendation: secondary labelled sets and a datasets verdict page | data | not recorded | DGNTY-45 | deferred (v1.1) | none | not recorded |

## Gates

Every change reaches the integration branch through a pull request, and each PR title carries `DGNTY-n`. Both `main` and `dev` are protected: no deletion and no force-push. The required checks are:

- `lint`: `ruff check` and `ruff format --check` at line length 88.
- `test (3.10)` and `test (3.12)`: the full suite, with a 100% coverage gate on `core/privacy.py`, so every privacy mechanism needs a named test.
- `smoke`: the installed console scripts train, export and evaluate all three tasks on a small config, and the working tree must stay clean afterwards.

The same lint gates run locally before a commit through the pre-commit hooks `ruff` and `ruff-format`. The doc checkers in `tests/test_docs.py` also run in CI: they check every quoted number, symbol and path in these pages against the committed results and the code.

## Plane to GitHub

A PR that names a DGNTY item in its title or body gets a comment from the Plane integration that lists the linked work items, with their ids and titles. For example, [#40](https://github.com/crichalchemist/Dignity/pull/40) carries a "Linked to Plane Work Item(s)" comment naming DGNTY-36. The comment is how a PR and its tracker item find each other, and it is why the id convention in PR titles matters.
