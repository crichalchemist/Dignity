"""The release verifier reads the four results files and judges success criteria 2-5.

These tests use synthetic files under tmp_path; only TestFinalResults reads the
committed results.
"""

import copy
import json
import re
from pathlib import Path

import pytest

from core.config import EvalConfig, PolicyConfig
from tests.test_docs import _load_script

release = _load_script("verify_release")
verify = release.verify
main = release.main
ResultsError = release.ResultsError

EPS = EvalConfig().epsilons
SEED = EvalConfig().eval_seed
BUDGET = PolicyConfig().review_budget
NAMES = ("risk", "forecast", "policy", "privacy")
ROOT = Path(__file__).resolve().parent.parent
# The commit tagged v1.0-freeze: the final evaluation was run from it and from nothing later.
FREEZE_SHA = "46f7ba04b7cc584fab98f0bacd66dbec5a2c11fb"
CHARTS = [
    "forecast_lead_time.png",
    "forecast_pr_auc_by_horizon.png",
    "policy_cost.png",
    "privacy_curve.png",
    "risk_per_type_recall.png",
    "risk_pr_curve.png",
]


def _gain(lows):
    """Per-horizon gain entries from their lower bounds (None = no paired interval)."""
    return [
        {"value": None, "ci_low": None, "ci_high": None}
        if low is None
        else {"value": 0.1, "ci_low": low, "ci_high": 0.2}
        for low in lows
    ]


def _points(eps):
    return [
        {"epsilon": e, "risk": {"pr_auc": {"value": 0.30 + 0.01 * i}}}
        for i, e in enumerate(eps)
    ]


def _merge(base, over):
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _passing():
    return {
        "risk": {
            "seed": SEED,
            "metrics": {
                "pr_auc": {"value": 0.52, "ci_low": 0.47, "ci_high": 0.57},
                "block_level": {
                    "pr_auc": {"value": 0.70, "ci_low": 0.62, "ci_high": 0.78}
                },
            },
            "baselines": {
                "majority": {"pr_auc": {"value": 0.27}},
                "window_logistic": {
                    "auroc": {"value": 0.68},
                    "pr_auc": {"value": 0.31},
                },
                "block_tree": {"auroc": {"value": 0.71}, "pr_auc": {"value": 0.55}},
            },
        },
        "forecast": {
            "seed": SEED,
            "metrics": {
                "gain_by_horizon": _gain([0.05] * 10),
                "warning": {
                    "n_onsets": 60,
                    "detection_rate": 0.55,
                    "detection_rate_by_seed": [0.5, 0.55, 0.6],
                    "random_rate": 0.1,
                    "gain": {"value": 0.45, "ci_low": 0.3, "ci_high": 0.6},
                },
                "lead_time": {"median": 4},
                "seed_spread": {"train_seeds": [42, 43, 44]},
            },
        },
        "policy": {
            "seed": SEED,
            "metrics": {
                "expected_cost": {"value": 0.21},
                "review_rate": {"value": 0.03},
            },
            "baselines": {"threshold_policy": {"expected_cost": {"value": 0.27}}},
        },
        "privacy": {
            "seed": SEED,
            "metrics": {
                "points": _points(EPS),
                "reference": {"epsilon": None, "risk": {"pr_auc": {"value": 0.8}}},
            },
        },
    }


def _read(tmp_path, name):
    return json.loads((tmp_path / f"{name}.json").read_text())


def _write(tmp_path, **overrides):
    files = copy.deepcopy(_passing())
    for name, over in overrides.items():
        _merge(files[name], over)
    for name, content in files.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content))


class TestVerifyRelease:
    def test_model_gate_fails_loudly_when_ci_does_not_clear_best_baseline(
        self, tmp_path
    ):
        _write(
            tmp_path,
            risk={"metrics": {"pr_auc": {"value": 0.35, "ci_low": 0.28}}},
        )
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.id == 2 and c2.passed is False
        assert "0.2800" in c2.detail and "0.3100" in c2.detail

    def test_block_tree_is_compared_at_block_level_only(self, tmp_path):
        # window ci_low 0.47 < 0.50 < block ci_low 0.62: judged against the window
        # interval this would fail, judged at block level it passes
        _write(
            tmp_path, risk={"baselines": {"block_tree": {"pr_auc": {"value": 0.50}}}}
        )
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.passed is True
        assert "0.6200" in c2.detail and "0.5000" in c2.detail

        _write(
            tmp_path, risk={"baselines": {"block_tree": {"pr_auc": {"value": 0.65}}}}
        )
        assert verify(tmp_path, SEED, EPS)[0].passed is False

    def test_a_null_interval_bound_fails_the_model_gate(self, tmp_path):
        _write(tmp_path, risk={"metrics": {"pr_auc": {"ci_low": None}}})
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.passed is False and "n/a" in c2.detail

    def test_policy_must_cost_less_than_the_threshold_baseline(self, tmp_path):
        _write(tmp_path, policy={"metrics": {"expected_cost": {"value": 0.30}}})
        c4 = verify(tmp_path, SEED, EPS)[2]
        assert c4.passed is False
        assert "0.3000" in c4.detail and "0.2700" in c4.detail

    def test_a_cheaper_policy_that_overspends_the_review_budget_fails(self, tmp_path):
        over = PolicyConfig().review_budget + 0.01
        _write(tmp_path, policy={"metrics": {"review_rate": {"value": over}}})
        c4 = verify(tmp_path, SEED, EPS)[2]
        assert c4.passed is False
        assert f"{over:.4f}" in c4.detail
        assert f"{PolicyConfig().review_budget:.4f}" in c4.detail

    def test_privacy_curve_needs_every_configured_epsilon_and_the_reference(
        self, tmp_path
    ):
        _write(tmp_path, privacy={"metrics": {"points": _points(EPS[1:])}})
        assert verify(tmp_path, SEED, EPS)[3].passed is False

    def test_privacy_curve_needs_a_reference_pr_auc(self, tmp_path):
        _write(tmp_path, privacy={"metrics": {"reference": {"risk": {"pr_auc": None}}}})
        c5 = verify(tmp_path, SEED, EPS)[3]
        assert c5.passed is False and "n/a" in c5.detail

    def test_results_from_another_seed_are_rejected_not_reported(
        self, tmp_path, capsys
    ):
        _write(tmp_path, risk={"seed": 1000})
        with pytest.raises(ResultsError, match="seed 1000"):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2
        assert "PASS" not in capsys.readouterr().out

    def test_a_missing_file_is_unusable_not_a_failed_criterion(self, tmp_path):
        _write(tmp_path)
        (tmp_path / "policy.json").unlink()
        with pytest.raises(ResultsError, match="policy"):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2

    def test_unreadable_json_is_unusable(self, tmp_path):
        _write(tmp_path)
        (tmp_path / "risk.json").write_text("{not json")
        with pytest.raises(ResultsError, match="risk"):
            verify(tmp_path, SEED, EPS)

    def test_a_missing_key_is_unusable(self, tmp_path):
        _write(tmp_path)
        data = json.loads((tmp_path / "policy.json").read_text())
        del data["baselines"]["threshold_policy"]
        (tmp_path / "policy.json").write_text(json.dumps(data))
        with pytest.raises(ResultsError, match="threshold_policy"):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2

    @pytest.mark.parametrize(
        ("name", "path"),
        [
            ("forecast", ["metrics", "gain_by_horizon"]),
            ("privacy", ["metrics", "points"]),
        ],
    )
    def test_a_null_list_is_unusable_not_a_traceback(self, tmp_path, name, path):
        _write(tmp_path)
        data = json.loads((tmp_path / f"{name}.json").read_text())
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = None
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        with pytest.raises(ResultsError, match=path[-1]):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2

    @pytest.mark.parametrize(
        ("name", "edit", "missing"),
        [
            (
                "forecast",
                lambda d: d["metrics"]["gain_by_horizon"][0].pop("ci_low"),
                "ci_low",
            ),
            (
                "forecast",
                lambda d: d["metrics"]["warning"]["gain"].pop("ci_low"),
                "ci_low",
            ),
            (
                "privacy",
                lambda d: d["metrics"]["points"][0].pop("epsilon"),
                "epsilon",
            ),
        ],
    )
    def test_an_entry_missing_its_key_is_unusable_not_a_traceback(
        self, tmp_path, name, edit, missing
    ):
        _write(tmp_path)
        data = json.loads((tmp_path / f"{name}.json").read_text())
        edit(data)
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        with pytest.raises(ResultsError, match=missing):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2

    def test_all_passing_results_exit_zero_and_print_one_line_per_criterion(
        self, tmp_path, capsys
    ):
        _write(tmp_path)
        assert main(["--results", str(tmp_path)]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert [ln.split()[:2] for ln in lines] == [
            ["C2", "PASS"],
            ["C3", "PASS"],
            ["C4", "PASS"],
            ["C5", "PASS"],
        ]

    def test_a_failed_criterion_exits_one(self, tmp_path):
        _write(tmp_path, policy={"metrics": {"expected_cost": {"value": 0.30}}})
        assert main(["--results", str(tmp_path)]) == 1


def _edit(tmp_path, name, edit):
    """Write the passing files, then rewrite one of them through `edit(data)`."""
    _write(tmp_path)
    path = tmp_path / f"{name}.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


def _set(data, path, value):
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _run(tmp_path, i, **overrides):
    """Write the passing files with `overrides`, return criterion `i` (0 is C2)."""
    _write(tmp_path, **overrides)
    return verify(tmp_path, SEED, EPS)[i]


class TestCriterion2WithoutBand:
    @pytest.mark.parametrize("auroc", [0.45, 0.55, 0.93])
    def test_c2_passes_at_any_baseline_auroc_when_the_model_clears_both_baselines(
        self, tmp_path, auroc
    ):
        _write(
            tmp_path,
            risk={
                "baselines": {
                    "window_logistic": {"auroc": {"value": auroc}},
                    "block_tree": {"auroc": {"value": auroc}},
                }
            },
        )
        assert verify(tmp_path, SEED, EPS)[0].passed is True

    @pytest.mark.parametrize(
        ("auroc", "shown"),
        [(0.5301, "block_tree 0.5301"), (None, "block_tree n/a")],
    )
    def test_the_baseline_auroc_is_reported_not_judged_and_null_prints_n_a(
        self, tmp_path, auroc, shown
    ):
        _edit(
            tmp_path,
            "risk",
            lambda d: _set(d, ("baselines", "block_tree", "auroc", "value"), auroc),
        )
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.passed and shown in c2.detail and "reported, not judged" in c2.detail


class TestCriterion3SeedMatchedGain:
    @pytest.mark.parametrize(
        ("wins", "count", "passed"),
        [(6, 10, True), (5, 10, False), (5, 9, True), (4, 9, False)],
    )
    def test_more_than_half_of_the_horizons_must_be_clear_wins(
        self, tmp_path, wins, count, passed
    ):
        gain = _gain([0.05] * wins + [-0.01] * (count - wins))
        c3 = _run(tmp_path, 1, forecast={"metrics": {"gain_by_horizon": gain}})
        assert c3.passed is passed and f"{wins} of {count} horizons" in c3.detail

    @pytest.mark.parametrize(("low", "win"), [(0.0, False), (1e-9, True)])
    def test_a_horizon_whose_interval_touches_zero_is_not_a_win(
        self, tmp_path, low, win
    ):
        gain = _gain([low] + [-1.0] * 9)
        _write(tmp_path, forecast={"metrics": {"gain_by_horizon": gain}})
        assert release.forecast_verdict(_read(tmp_path, "forecast"), 3).wins == int(win)

    def test_a_horizon_without_a_paired_interval_is_never_a_win_and_says_n_a(
        self, tmp_path
    ):
        gain = _gain([None] * 6 + [0.05] * 4)
        c3 = _run(tmp_path, 1, forecast={"metrics": {"gain_by_horizon": gain}})
        assert not c3.passed
        assert "4 of 10 horizons" in c3.detail and "6 n/a" in c3.detail

    @pytest.mark.parametrize(
        ("low", "passed"), [(0.0, False), (0.001, True), (None, False)]
    )
    def test_the_warning_share_must_beat_a_random_score_by_the_interval_lower_bound(
        self, tmp_path, low, passed
    ):
        warning = {"gain": {"ci_low": low}}
        c3 = _run(tmp_path, 1, forecast={"metrics": {"warning": warning}})
        assert c3.passed is passed
        assert "detection 0.5500 vs random 0.1000" in c3.detail

    def test_a_good_ranking_cannot_rescue_a_chance_level_warning_and_the_reverse(
        self, tmp_path
    ):
        # the median lead time is not judged: a high one rescues nothing
        metrics = {"warning": {"gain": {"ci_low": -0.1}}, "lead_time": {"median": 9}}
        _write(tmp_path, forecast={"metrics": metrics})
        assert not release.forecast_verdict(_read(tmp_path, "forecast"), 3).passed
        _write(tmp_path, forecast={"metrics": {"gain_by_horizon": _gain([-0.1] * 10)}})
        v = release.forecast_verdict(_read(tmp_path, "forecast"), 3)
        assert v.warning_passed and not v.ranking_passed and not v.passed

    def test_the_median_lead_time_is_printed_not_judged(self, tmp_path):
        c3 = _run(tmp_path, 1, forecast={"metrics": {"lead_time": {"median": None}}})
        assert c3.passed and "median lead time n/a" in c3.detail

    @pytest.mark.parametrize(
        ("metrics", "needle"),
        [
            ({"seed_spread": {"train_seeds": [42]}}, "train_seeds"),
            ({"gain_by_horizon": []}, "no horizons"),
        ],
    )
    def test_too_few_training_seeds_or_no_horizons_is_unusable_not_a_verdict(
        self, tmp_path, metrics, needle
    ):
        _write(tmp_path, forecast={"metrics": metrics})
        with pytest.raises(ResultsError, match=needle):
            verify(tmp_path, SEED, EPS)


class TestCriteriaBoundaries:
    """Each case states the verdict on a boundary of the spec: "beats", "lower" and
    "above" are strict, "at most the budget" is inclusive."""

    @pytest.mark.parametrize(
        ("majority", "window_logistic", "passed"),
        [
            (0.50, 0.31, False),  # majority is the binding baseline
            (0.46, 0.31, True),
            (0.27, 0.50, False),  # window_logistic is the binding baseline
            (0.27, 0.46, True),
        ],
    )
    def test_the_best_window_baseline_is_the_higher_of_majority_and_logistic(
        self, tmp_path, majority, window_logistic, passed
    ):
        # window ci_low is 0.47
        _write(
            tmp_path,
            risk={
                "baselines": {
                    "majority": {"pr_auc": {"value": majority}},
                    "window_logistic": {"pr_auc": {"value": window_logistic}},
                }
            },
        )
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.passed is passed
        assert f"best baseline {max(majority, window_logistic):.3f}" in c2.detail

    @pytest.mark.parametrize(("ci_low", "passed"), [(0.31, False), (0.311, True)])
    def test_window_interval_must_sit_strictly_above_the_best_baseline(
        self, tmp_path, ci_low, passed
    ):
        # best window baseline is window_logistic at 0.31
        _write(tmp_path, risk={"metrics": {"pr_auc": {"ci_low": ci_low}}})
        assert verify(tmp_path, SEED, EPS)[0].passed is passed

    @pytest.mark.parametrize(("tree", "passed"), [(0.62, False), (0.619, True)])
    def test_block_interval_must_sit_strictly_above_block_tree(
        self, tmp_path, tree, passed
    ):
        # block ci_low is 0.62
        _write(
            tmp_path, risk={"baselines": {"block_tree": {"pr_auc": {"value": tree}}}}
        )
        assert verify(tmp_path, SEED, EPS)[0].passed is passed

    @pytest.mark.parametrize(
        "path",
        [
            ("baselines", "majority", "pr_auc", "value"),
            ("baselines", "window_logistic", "pr_auc", "value"),
            ("baselines", "block_tree", "pr_auc", "value"),
            ("metrics", "pr_auc", "ci_low"),
            ("metrics", "block_level", "pr_auc", "ci_low"),
        ],
        ids=lambda p: ".".join(p),
    )
    def test_a_null_in_criterion_2_fails_it_and_says_n_a(self, tmp_path, path):
        _edit(tmp_path, "risk", lambda d: _set(d, path, None))
        c2 = verify(tmp_path, SEED, EPS)[0]
        assert c2.passed is False and "n/a" in c2.detail

    @pytest.mark.parametrize(
        ("cost", "rate", "passed"),
        [
            (0.27, 0.03, False),  # cost equal to the baseline is not lower
            (0.269, 0.03, True),
            (0.21, BUDGET, True),  # rate exactly at the budget is within it
            (0.21, BUDGET + 0.0001, False),
        ],
    )
    def test_policy_must_be_strictly_cheaper_within_an_inclusive_review_budget(
        self, tmp_path, cost, rate, passed
    ):
        _write(
            tmp_path,
            policy={
                "metrics": {
                    "expected_cost": {"value": cost},
                    "review_rate": {"value": rate},
                }
            },
        )
        assert verify(tmp_path, SEED, EPS)[2].passed is passed

    @pytest.mark.parametrize(
        "path",
        [
            ("metrics", "expected_cost", "value"),
            ("baselines", "threshold_policy", "expected_cost", "value"),
            ("metrics", "review_rate", "value"),
        ],
        ids=lambda p: ".".join(p),
    )
    def test_a_null_in_criterion_4_fails_it_and_says_n_a(self, tmp_path, path):
        _edit(tmp_path, "policy", lambda d: _set(d, path, None))
        c4 = verify(tmp_path, SEED, EPS)[2]
        assert c4.passed is False and "n/a" in c4.detail

    @pytest.mark.parametrize(
        "epsilons",
        [
            [*EPS[:1], EPS[2], EPS[1], *EPS[3:]],
            [*EPS, EPS[-1] * 10],
            [*EPS[:2], EPS[1], *EPS[3:]],
        ],
        ids=["out_of_order", "extra_point", "duplicated_point"],
    )
    def test_privacy_curve_must_be_exactly_the_configured_grid_in_order(
        self, tmp_path, epsilons
    ):
        _write(tmp_path, privacy={"metrics": {"points": _points(epsilons)}})
        assert verify(tmp_path, SEED, EPS)[3].passed is False

    def test_one_point_without_a_pr_auc_fails_the_privacy_curve(self, tmp_path):
        points = _points(EPS)
        points[2]["risk"]["pr_auc"] = None
        _write(tmp_path, privacy={"metrics": {"points": points}})
        c5 = verify(tmp_path, SEED, EPS)[3]
        assert c5.passed is False and "1 point(s) n/a" in c5.detail


class TestDetailsShowTheMargin:
    def test_a_review_rate_just_over_the_budget_prints_two_different_numbers(
        self, tmp_path
    ):
        _write(tmp_path, policy={"metrics": {"review_rate": {"value": 0.0504}}})
        c4 = verify(tmp_path, SEED, EPS)[2]
        assert c4.passed is False
        assert "review rate 0.0504 vs budget 0.0500" in c4.detail

    def test_costs_that_differ_by_a_ten_thousandth_print_differently(self, tmp_path):
        _write(
            tmp_path,
            policy={
                "metrics": {"expected_cost": {"value": 1.0688}},
                "baselines": {"threshold_policy": {"expected_cost": {"value": 1.0687}}},
            },
        )
        c4 = verify(tmp_path, SEED, EPS)[2]
        assert c4.passed is False
        assert "expected cost 1.0688 vs threshold baseline 1.0687" in c4.detail

    def test_cost_and_pr_auc_print_four_decimals(self, tmp_path):
        _write(tmp_path)
        lines = [c.detail for c in verify(tmp_path, SEED, EPS)]
        assert "expected cost 0.2100 vs threshold baseline 0.2700" in lines[2]
        assert "ci_low 0.4700 vs best baseline 0.3100" in lines[0]

    def test_the_median_lead_time_keeps_two_decimals(self, tmp_path):
        _write(tmp_path)
        assert "median lead time 4.00" in verify(tmp_path, SEED, EPS)[1].detail


class TestUnusableResults:
    @pytest.mark.parametrize("name", NAMES)
    def test_a_wrong_seed_in_any_file_is_unusable(self, tmp_path, capsys, name):
        _write(tmp_path, **{name: {"seed": 1000}})
        with pytest.raises(ResultsError, match=f"{name}: seed 1000"):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("unusable results: ")

    @pytest.mark.parametrize("seed", ["2026", 2026.0, True, 2027, 2025, None])
    def test_a_seed_that_is_not_exactly_the_integer_is_unusable(self, tmp_path, seed):
        _write(tmp_path, risk={"seed": seed})
        with pytest.raises(ResultsError, match=re.escape(f"seed {seed!r} is not")):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2

    @pytest.mark.parametrize(
        ("name", "path", "bad"),
        [
            ("risk", ("metrics", "pr_auc", "ci_low"), "0.9"),
            ("risk", ("metrics", "block_level", "pr_auc", "ci_low"), True),
            ("risk", ("baselines", "window_logistic", "auroc", "value"), float("nan")),
            ("risk", ("baselines", "block_tree", "auroc", "value"), float("inf")),
            ("risk", ("baselines", "majority", "pr_auc", "value"), "x"),
            (
                "risk",
                ("baselines", "window_logistic", "pr_auc", "value"),
                -float("inf"),
            ),
            ("risk", ("baselines", "block_tree", "pr_auc", "value"), [0.5]),
            ("forecast", ("metrics", "lead_time", "median"), "4"),
            ("forecast", ("metrics", "lead_time", "median"), float("nan")),
            ("forecast", ("metrics", "gain_by_horizon", 0, "ci_low"), float("inf")),
            ("forecast", ("metrics", "warning", "gain", "ci_low"), True),
            ("policy", ("metrics", "expected_cost", "value"), "0.2"),
            ("policy", ("metrics", "expected_cost", "value"), -float("inf")),
            (
                "policy",
                ("baselines", "threshold_policy", "expected_cost", "value"),
                float("inf"),
            ),
            ("policy", ("metrics", "review_rate", "value"), float("nan")),
            ("privacy", ("metrics", "points", 0, "epsilon"), "0.1"),
            (
                "privacy",
                ("metrics", "points", 2, "risk", "pr_auc", "value"),
                float("nan"),
            ),
            ("privacy", ("metrics", "reference", "risk", "pr_auc", "value"), False),
        ],
    )
    def test_a_value_that_is_not_a_finite_number_is_unusable_not_a_verdict(
        self, tmp_path, capsys, name, path, bad
    ):
        _edit(tmp_path, name, lambda d: _set(d, path, bad))
        dotted = ".".join(str(k) for k in path)
        with pytest.raises(ResultsError, match=re.escape(f"{name}: {dotted} must be")):
            verify(tmp_path, SEED, EPS)
        assert main(["--results", str(tmp_path)]) == 2
        assert capsys.readouterr().out == ""


class TestFinalResults:
    """The committed results are the one final run: evaluation seed, frozen commit."""

    def _committed(self):
        return {
            name: json.loads((ROOT / f"results/v1.0/{name}.json").read_text())
            for name in NAMES
        }

    def test_all_committed_results_come_from_the_evaluation_seed(self):
        for name, data in self._committed().items():
            assert data["seed"] == EvalConfig().eval_seed, name

    def test_all_committed_results_were_made_from_the_frozen_commit(self):
        docs = self._committed()
        for name, data in docs.items():
            assert data["git_sha"] == FREEZE_SHA, name
        assert len({data["config_hash"] for data in docs.values()}) == 1

    def test_every_chart_the_results_produce_is_committed(self):
        charts = sorted(p.name for p in (ROOT / "docs/img").glob("*.png"))
        assert charts == CHARTS
