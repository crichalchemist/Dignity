import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eval.metrics as metrics_module
from core.config import DignityConfig, PolicyConfig
from eval.baselines import block_tree, risk_score_forecast, threshold_policy
from eval.metrics import (
    bootstrap_ci,
    forecast_metrics,
    lead_times,
    model_gate,
    policy_metrics,
    risk_metrics,
    threshold_at_fpr,
)
from eval.plots import (
    plot_forecast_pr_auc,
    plot_lead_times,
    plot_per_type_recall,
    plot_pr_curves,
)
from eval.run import _early_warning, _spread, config_hash, main
from models.head.policy import cost_matrix
from train.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.35, 0.05])
T = np.array(["none"] * 4 + ["wash_trading"] * 2 + ["fee_spike"] * 2, dtype=object)


class TestRiskMetrics:
    def test_point_metrics_match_hand_computed_values(self):
        m = risk_metrics(Y, S, np.arange(8), n_boot=0, seed=0, anomaly_type=T)
        assert m["auroc"]["value"] == pytest.approx(0.6875)
        assert m["pr_auc"]["value"] == pytest.approx(0.8125)
        assert m["recall_at_1pct_fpr"]["value"] == pytest.approx(0.5)
        assert m["brier"]["value"] == pytest.approx(0.209375)
        assert m["prevalence"]["value"] == pytest.approx(0.5)
        assert m["auroc"]["ci_low"] is None

    def test_per_type_recall_shows_which_typology_the_detector_misses(self):
        m = risk_metrics(Y, S, np.arange(8), n_boot=0, seed=0, anomaly_type=T)
        assert m["per_type"]["wash_trading"]["value"] == pytest.approx(1.0)
        assert m["per_type"]["fee_spike"]["value"] == pytest.approx(0.0)
        assert m["per_type"]["fee_spike"]["n_windows"] == 2
        assert "none" not in m["per_type"]

    def test_per_type_is_empty_without_anomaly_types(self):
        m = risk_metrics(Y, S, np.arange(8), n_boot=0, seed=0)
        assert m["per_type"] == {}

    def test_threshold_admits_at_most_one_percent_false_alarms(self):
        y = np.r_[np.zeros(1000), np.ones(10)]
        s = np.r_[np.arange(1000) / 1000, np.full(10, 2.0)]
        t = threshold_at_fpr(y, s, 0.01)
        assert (s[y == 0] >= t).mean() <= 0.01 and (s[y == 1] >= t).all()

    def test_single_class_split_fails_loudly(self):
        with pytest.raises(ValueError, match="both classes"):
            risk_metrics(np.zeros(4), np.full(4, 0.5), np.arange(4), 0, 0)

    def test_intervals_are_reproducible_from_the_seed(self):
        blocks = np.repeat(np.arange(4), 2)
        a = risk_metrics(Y, S, blocks, n_boot=200, seed=3)
        b = risk_metrics(Y, S, blocks, n_boot=200, seed=3)
        assert a["pr_auc"] == b["pr_auc"]
        assert a["pr_auc"]["ci_low"] <= a["pr_auc"]["ci_high"]

    def test_report_is_strict_json_when_scores_are_degenerate(self):
        # Constant scores give a +inf alarm threshold; resamples can hold one
        # class. Neither may leak inf/NaN or numpy scalars into the report.
        blocks = np.repeat(np.arange(4), 2)
        m = risk_metrics(Y, np.full(8, 0.5), blocks, n_boot=30, seed=1, anomaly_type=T)
        json.dumps(m, allow_nan=False)

    def test_undefined_interval_is_reported_as_null_not_nan(self, monkeypatch):
        monkeypatch.setattr(
            metrics_module, "bootstrap_ci", lambda *a: (float("nan"), float("nan"))
        )
        m = risk_metrics(Y, S, np.arange(8), n_boot=5, seed=0, anomaly_type=T)
        assert m["auroc"]["ci_low"] is None and m["auroc"]["ci_high"] is None
        json.dumps(m, allow_nan=False)


class TestBootstrapCI:
    def test_bootstrap_resamples_whole_blocks_not_windows(self):
        # DGNTY-21: stride-1 windows of one block are not independent samples.
        block_index = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # block sizes 3 and 5
        draws = []
        bootstrap_ci(lambda idx: draws.append(idx) or 0.0, block_index, 50, seed=0)
        assert {len(idx) for idx in draws} <= {6, 8, 10}
        for idx in draws:
            assert set(idx) <= set(range(8))
            counts = np.bincount(idx, minlength=8)
            # every block enters whole: same count for all of its windows
            assert len(set(counts[:3])) == 1 and len(set(counts[3:])) == 1
        assert any(len(idx) != 8 for idx in draws)

    def test_all_undefined_resamples_give_nan_interval(self):
        low, high = bootstrap_ci(lambda idx: float("nan"), np.arange(4), 10, seed=0)
        assert np.isnan(low) and np.isnan(high)


class TestModelGate:
    def test_gate_needs_ci_low_above_best_baseline_point_estimate(self):
        base = {"majority": {"value": 0.30}, "window_logistic": {"value": 0.52}}
        ok = model_gate({"value": 0.60, "ci_low": 0.55, "ci_high": 0.65}, base)
        assert ok == {
            "best_baseline": "window_logistic",
            "best_baseline_pr_auc": 0.52,
            "model_pr_auc": 0.60,
            "model_ci_low": 0.55,
            "passed": True,
        }
        assert isinstance(ok["passed"], bool)
        beats_point_only = model_gate(
            {"value": 0.60, "ci_low": 0.51, "ci_high": 0.65}, base
        )
        assert beats_point_only["passed"] is False

    def test_gate_without_a_confidence_interval_fails_loudly(self):
        with pytest.raises(ValueError, match="CI"):
            model_gate(
                {"value": 0.6, "ci_low": None, "ci_high": None}, {"m": {"value": 0.1}}
            )


class TestBaselines:
    def test_block_tree_rows_follow_ascending_block_id(self, easy_frame):
        # eval.run pairs model scores pooled in np.unique(block_index) order with
        # block_tree's rows; the two orders must agree.
        df = easy_frame
        df_train = df[df.block_id < 30]
        labels = df[df.block_id >= 30].groupby("block_id")["block_label"].max()
        normal = labels[labels == 0].index[:2]
        anomalous = labels[labels == 1].index[:1]
        df_test = pd.concat(
            [
                df[df.block_id == b].assign(block_id=new)
                for b, new in ((anomalous[0], 9), (normal[0], 2), (normal[1], 5))
            ],
            ignore_index=True,
        )  # row order: 9, 2, 5
        y, _ = block_tree(df_train, df_test)
        assert y.tolist() == [0, 0, 1]  # ascending block_id: 2, 5, 9

    def test_risk_score_baseline_predicts_the_same_risk_at_every_horizon(self):
        s = np.array([0.1, 0.7, 0.4], np.float32)
        out = risk_score_forecast(s, horizon=4)
        assert out.shape == (3, 4) and (out == s[:, None]).all()
        assert out.dtype == np.float32

    def test_blocks_do_not_count_toward_the_review_budget(self):
        C = cost_matrix(PolicyConfig().costs)
        val_s = np.r_[np.full(100, 0.1), np.full(100, 0.9)]
        val_y = np.r_[np.zeros(100, int), np.ones(100, int)]
        a = threshold_policy(val_s, val_y, np.array([0.1, 0.9]), C, budget=0.05)
        assert a.tolist() == [0, 2]  # 50% blocked is allowed
        assert a.dtype == np.int64

    def test_baseline_never_sends_more_than_capacity_to_review_on_test(self):
        C = cost_matrix(PolicyConfig().costs)
        # val: 1900 normals at 0.0, 100 mixed (50/50) at 0.5 -> reviewing 0.5 is optimal
        val_s = np.r_[np.zeros(1900), np.full(100, 0.5)]
        val_y = np.r_[np.zeros(1900, int), np.tile([0, 1], 50)]
        test_s = np.r_[np.full(100, 0.5), np.zeros(100)]  # 50% in the review band
        a = threshold_policy(val_s, val_y, test_s, C, budget=0.05)
        assert (a == 1).sum() == 10 and (a == 2).sum() == 0


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestPlots:
    def test_charts_are_written_as_png(self, tmp_path):
        y, s = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6])
        plot_pr_curves({"model": (y, s)}, str(tmp_path / "pr.png"))
        plot_per_type_recall(
            {
                "wash_trading": {
                    "value": 0.5,
                    "ci_low": 0.2,
                    "ci_high": 0.8,
                    "n_windows": 4,
                }
            },
            str(tmp_path / "t.png"),
        )
        assert (tmp_path / "pr.png").read_bytes()[:8] == PNG_MAGIC
        assert (tmp_path / "t.png").stat().st_size > 0

    def test_per_type_chart_survives_missing_or_edge_intervals(self, tmp_path):
        # H22: a type without a CI draws a bare bar; a CI that touches the value
        # (or rounds past it) must not hand matplotlib a negative error.
        per_type = {
            "wash_trading": {
                "value": 0.5,
                "ci_low": None,
                "ci_high": None,
                "n_windows": 4,
            },
            "fee_spike": {"value": 0.0, "ci_low": 0.0, "ci_high": 0.3, "n_windows": 3},
            "rug_pull": {"value": 0.4, "ci_low": 0.5, "ci_high": 0.3, "n_windows": 2},
        }
        plot_per_type_recall(per_type, str(tmp_path / "t.png"))
        assert (tmp_path / "t.png").read_bytes()[:8] == PNG_MAGIC

    def test_per_type_chart_accepts_no_types(self, tmp_path):
        plot_per_type_recall({}, str(tmp_path / "t.png"))
        assert (tmp_path / "t.png").stat().st_size > 0


class TestEvalRunSmoke:
    def test_smoke_evaluation_writes_results_schema_and_charts_within_a_minute(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.chdir(tmp_path)  # all smoke outputs land in tmp_path
        t0 = time.perf_counter()
        assert main(["--smoke", "--seed", "tuning"]) == 0
        assert time.perf_counter() - t0 < 60
        printed = capsys.readouterr().out
        assert "majority" in printed and "window_logistic" in printed  # spec 5.3
        out = json.loads((tmp_path / "runs/smoke/results/risk.json").read_text())
        assert set(out) == {"config_hash", "seed", "git_sha", "metrics", "baselines"}
        assert out["seed"] == 1000
        assert {
            "auroc",
            "pr_auc",
            "recall_at_1pct_fpr",
            "brier",
            "prevalence",
            "per_type",
            "block_level",
            "seed_spread",
            "model_gate",
        } <= set(out["metrics"])
        assert set(out["baselines"]) == {"majority", "window_logistic", "block_tree"}
        assert isinstance(out["metrics"]["model_gate"]["passed"], bool)
        for png in ("risk_pr_curve.png", "risk_per_type_recall.png"):
            assert (tmp_path / "runs/smoke/img" / png).stat().st_size > 0

    def test_unbuilt_tasks_are_refused_not_silently_skipped(
        self, capsys, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)

        def boom(*args, **kwargs):
            raise AssertionError("must not run")

        # prepare_data is the first data-touching call of every task path
        monkeypatch.setattr("eval.run.prepare_data", boom)
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--seed", "tuning", "--tasks", "policy"])
        assert exc.value.code == 2
        assert "not implemented until M4" in capsys.readouterr().err

    def test_smoke_preset_cannot_be_combined_with_the_evaluation_seed(
        self, monkeypatch
    ):
        # prepare_data is booby-trapped so the refusal must happen before any work
        def boom(*args, **kwargs):
            raise AssertionError("prepare_data reached")

        monkeypatch.setattr("eval.run.prepare_data", boom)
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--seed", "eval"])
        assert exc.value.code == 2

    def test_config_and_smoke_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--config", "config/smoke.yaml", "--seed", "tuning"])
        assert exc.value.code == 2

    def test_one_hash_identifies_one_configuration(self):
        a = DignityConfig()
        b = DignityConfig()
        b.data.synthetic.seed += 1
        assert config_hash(a) == config_hash(DignityConfig())
        assert config_hash(a) != config_hash(b)
        assert len(config_hash(a)) == 16


class TestForecastMetrics:
    def test_active_anomaly_windows_are_excluded_from_forecast_scoring(self):
        y = np.array([[0, 1], [1, 1], [0, 0], [0, 1], [1, 1], [0, 0]], np.float32)
        score = y.copy()
        score[4:] = 1 - y[4:]  # adversarial, but masked out
        mask = np.array([1, 1, 1, 1, 0, 0], bool)
        m = forecast_metrics(
            y, score, mask, np.array([0, 0, 1, 1, 2, 2]), n_boot=50, seed=0
        )
        assert [h["value"] for h in m["pr_auc_by_horizon"]] == [1.0, 1.0]
        assert m["prevalence_by_horizon"] == [0.25, 0.75]

    def test_horizon_without_onsets_reports_none_not_a_fake_score(self):
        y = np.array([[0, 1], [0, 1], [0, 0], [0, 1], [0, 1], [0, 0]], np.float32)
        mask = np.ones(6, bool)
        m = forecast_metrics(
            y, y.copy(), mask, np.array([0, 0, 1, 1, 2, 2]), n_boot=20, seed=0
        )
        assert m["pr_auc_by_horizon"][0] == {
            "value": None,
            "ci_low": None,
            "ci_high": None,
        }
        assert m["pr_auc_by_horizon"][1]["value"] == 1.0

    def test_single_class_or_empty_horizons_report_none_never_nan(self):
        # horizon 1 is all positive once masked, horizon 0 all negative
        y = np.array([[0, 1], [0, 1], [0, 0], [0, 1]], np.float32)
        mask = np.array([1, 1, 0, 0], bool)
        block = np.array([0, 0, 1, 1])
        m = forecast_metrics(y, y.copy(), mask, block, n_boot=10, seed=0)
        none = {"value": None, "ci_low": None, "ci_high": None}
        assert m["pr_auc_by_horizon"] == [none, none]
        empty = forecast_metrics(y, y.copy(), np.zeros(4, bool), block, 10, 0)
        assert empty["pr_auc_by_horizon"] == [none, none]
        assert empty["prevalence_by_horizon"] == [None, None]
        json.dumps(m, allow_nan=False)
        json.dumps(empty, allow_nan=False)

    def test_forecast_metrics_survive_strict_json(self):
        y = np.array([[0, 1], [1, 1], [0, 0], [0, 1]], np.float32)
        m = forecast_metrics(
            y, y.copy(), np.ones(4, bool), np.array([0, 0, 1, 1]), n_boot=10, seed=0
        )
        json.dumps(m, allow_nan=False)
        assert all(type(p) is float for p in m["prevalence_by_horizon"])


class TestLeadTimes:
    def _fixture(self):
        # H = 3. Block 0: windows end at rows 4..9, onset at row 8 (rows 8-9 active).
        # Block 1: windows end at rows 4..9, onset at row 7; its window ending at
        # row 6 is masked (anomaly active) with a loud alarm that is not a warning.
        y = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]
            + [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            np.float32,
        )
        mask = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], bool)
        alarm = np.array(
            [0.1, 0.5, 0.9, 0.95, 0.99, 0.99] + [0.1, 0.1, 0.99] + [0.1] * 3
        )
        return alarm, y, mask, np.repeat([0, 1], 6), np.tile(np.arange(4, 10), 2)

    def test_lead_time_counts_rows_from_first_alarm_to_onset(self):
        alarm, y, mask, block, end_row = self._fixture()
        r = lead_times(alarm, 0.5, y, mask, block, end_row)
        assert r == {"median": 3.0, "values": [3], "n_onsets": 2, "n_missed": 1}

    def test_lead_times_are_plain_python_numbers_for_json(self):
        alarm, y, mask, block, end_row = self._fixture()
        r = lead_times(alarm, 0.5, y, mask, block, end_row)
        assert all(type(v) is int for v in r["values"])
        assert type(r["median"]) is float
        json.dumps(r, allow_nan=False)

    def test_no_alarm_at_all_gives_null_median(self):
        alarm, y, mask, block, end_row = self._fixture()
        r = lead_times(alarm, float("inf"), y, mask, block, end_row)
        assert r == {"median": None, "values": [], "n_onsets": 2, "n_missed": 2}


def _windows(y, mask, block, end_row):
    """The WindowSet fields ``_early_warning`` reads."""
    return SimpleNamespace(
        y_forecast=np.asarray(y, np.float32),
        forecast_mask=np.asarray(mask, bool),
        block_index=np.asarray(block),
        end_row=np.asarray(end_row),
    )


class TestEarlyWarning:
    def _split(self):
        # H = 2; 200 val windows so a 1% FPR budget can admit a threshold
        rng = np.random.default_rng(0)
        y = (rng.random((200, 2)) < 0.3).astype(np.float32)
        y[:, 0] = y[:, 1]
        ws = _windows(y, np.ones(200, bool), np.arange(200) // 10, np.arange(200) % 10)
        return ws, y

    def test_an_unreachable_false_alarm_budget_is_null_not_infinity(self):
        # J12: every negative outscores every positive, so no finite threshold
        # fits 1% FPR; threshold_at_fpr says +inf and strict JSON must still work
        ws, y = self._split()
        score = 1.0 - y
        r = _early_warning(ws, ws, score, score, n_boot=0, seed=0)
        assert r["threshold"] is None
        assert r["lead_time"]["n_missed"] == r["lead_time"]["n_onsets"]
        json.dumps(r, allow_nan=False)

    def test_a_perfect_detector_gets_a_finite_float_threshold(self):
        ws, y = self._split()
        r = _early_warning(ws, ws, y.copy(), y.copy(), n_boot=0, seed=0)
        assert type(r["threshold"]) is float
        assert set(r) == {
            "pr_auc_by_horizon",
            "prevalence_by_horizon",
            "lead_time",
            "threshold",
        }
        json.dumps(r, allow_nan=False)

    def test_the_alarm_threshold_is_set_on_validation_never_on_test(self):
        ws, y = self._split()
        test = _windows(y, np.ones(200, bool), ws.block_index, ws.end_row)
        # a test split whose scores are all zero would give threshold 0 if the
        # threshold were (wrongly) read from test
        r = _early_warning(ws, test, y.copy(), np.zeros_like(y), n_boot=0, seed=0)
        assert r["threshold"] > 0

    def test_spread_ignores_seeds_without_a_value(self):
        s = _spread([2.0, None, 4.0])
        assert s["values"] == [2.0, None, 4.0] and s["mean"] == 3.0
        assert _spread([None])["mean"] is None


class TestForecastPlots:
    def _result(self, values):
        entry = {"value": 0.5, "ci_low": 0.3, "ci_high": 0.7}
        side = {
            "pr_auc_by_horizon": [
                entry,
                {"value": None, "ci_low": None, "ci_high": None},
            ],
            "prevalence_by_horizon": [0.1, 0.1],
            "lead_time": {
                "median": 3.0 if values else None,
                "values": values,
                "n_onsets": 5,
                "n_missed": 5 - len(values),
            },
            "threshold": 0.4,
        }
        return {"metrics": side, "baselines": {"risk_score_forecast": side}}

    def test_forecast_charts_are_written_as_png(self, tmp_path):
        r = self._result([2, 3, 4])
        plot_forecast_pr_auc(r, str(tmp_path / "pr.png"))
        plot_lead_times(r, str(tmp_path / "lead.png"))
        assert (tmp_path / "pr.png").read_bytes()[:8] == PNG_MAGIC
        assert (tmp_path / "lead.png").read_bytes()[:8] == PNG_MAGIC

    def test_lead_time_chart_survives_a_model_that_never_alarmed(self, tmp_path):
        plot_lead_times(self._result([]), str(tmp_path / "lead.png"))
        assert (tmp_path / "lead.png").stat().st_size > 0


class TestEvalRunForecast:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        import os

        import eval.run as run_module

        tmp = tmp_path_factory.mktemp("forecast_smoke")
        fitted_tasks = []
        real_fit = run_module.fit

        def spy(config, *args, **kwargs):
            fitted_tasks.append(config.model.task)
            return real_fit(config, *args, **kwargs)

        cwd = os.getcwd()
        os.chdir(tmp)
        run_module.fit = spy
        try:
            assert main(["--smoke", "--seed", "tuning", "--tasks", "forecast"]) == 0
            forecast_only = list(fitted_tasks)
            risk_json_after_forecast_only = (
                tmp / "runs/smoke/results/risk.json"
            ).exists()
            fitted_tasks.clear()
            assert (
                main(["--smoke", "--seed", "tuning", "--tasks", "risk,forecast"]) == 0
            )
            both = list(fitted_tasks)
        finally:
            run_module.fit = real_fit
            os.chdir(cwd)
        return SimpleNamespace(
            root=tmp,
            forecast_only=forecast_only,
            risk_json_after_forecast_only=risk_json_after_forecast_only,
            both=both,
        )

    def test_forecast_results_report_every_horizon_against_the_baseline(self, run):
        out = run.root / "runs/smoke/results"
        img = run.root / "runs/smoke/img"
        smoke_cfg = DignityConfig.from_yaml(str(ROOT / "config/smoke.yaml"))
        r = json.loads((out / "forecast.json").read_text())
        H = smoke_cfg.model.horizon
        base = r["baselines"]["risk_score_forecast"]
        assert set(r) == {"config_hash", "seed", "git_sha", "metrics", "baselines"}
        assert len(r["metrics"]["pr_auc_by_horizon"]) == H
        assert len(r["metrics"]["prevalence_by_horizon"]) == H
        assert len(base["pr_auc_by_horizon"]) == H
        assert set(r["baselines"]) == {"risk_score_forecast"}
        assert set(r["metrics"]["lead_time"]) == {
            "median",
            "values",
            "n_onsets",
            "n_missed",
        }
        assert {
            "pr_auc_by_horizon",
            "prevalence_by_horizon",
            "lead_time",
            "threshold",
        } == set(base)
        assert (img / "forecast_pr_auc_by_horizon.png").exists()
        assert (img / "forecast_lead_time.png").exists()

    def test_forecast_results_carry_the_training_seed_spread(self, run):
        # spec 8a: CI over blocks AND spread across training seeds
        r = json.loads((run.root / "runs/smoke/results/forecast.json").read_text())
        spread = r["metrics"]["seed_spread"]
        assert spread["train_seeds"] == [42]
        assert len(spread["mean_pr_auc"]["values"]) == 1
        assert len(spread["lead_time_median"]["values"]) == 1

    def test_risk_is_fitted_not_reported_when_only_forecast_is_asked_for(self, run):
        # the first run asked for forecast only; risk.json belongs to the second
        assert run.forecast_only == ["risk", "forecast"]
        assert not run.risk_json_after_forecast_only

    def test_risk_is_fitted_once_when_both_tasks_are_asked_for(self, run):
        assert run.both == ["risk", "forecast"]
        assert (run.root / "runs/smoke/results/risk.json").exists()

    def test_forecast_never_overwrites_the_risk_checkpoint(self, run):
        # J2: both tasks train under one eval run; each keeps its own best.pt
        base = run.root / "runs/smoke/train/eval_tuning"
        tasks = {}
        for task in ("risk", "forecast"):
            tasks[task] = load_checkpoint(str(base / task / "train_seed_42/best.pt"))[
                "meta"
            ].task
        assert tasks == {"risk": "risk", "forecast": "forecast"}


class TestPolicyMetrics:
    def _metrics(self, n_boot=50):
        a = np.array([0, 1, 2, 2, 0, 1])
        y = np.array([1, 0, 1, 0, 0, 1])
        return policy_metrics(
            a,
            y,
            cost_matrix(PolicyConfig().costs),
            np.array([0, 0, 1, 1, 2, 2]),
            n_boot=n_boot,
            seed=0,
        )

    def test_policy_is_scored_on_cost_capacity_catches_and_false_blocks(self):
        m = self._metrics()
        assert m["expected_cost"]["value"] == pytest.approx(15.5 / 6)
        assert m["review_rate"]["value"] == pytest.approx(2 / 6)
        assert m["block_rate"]["value"] == pytest.approx(2 / 6)
        assert m["anomaly_recall"]["value"] == pytest.approx(2 / 3)
        assert m["false_block_rate"]["value"] == pytest.approx(1 / 3)
        # guard (J30): weak by design, a percentile CI need not bracket the value
        for v in m.values():
            assert v["ci_low"] <= v["ci_high"]

    def test_every_value_is_a_plain_number_or_null_for_strict_json(self):
        for n_boot in (50, 0):
            m = self._metrics(n_boot)
            for v in m.values():
                assert type(v["value"]) is float
                assert all(type(v[k]) in (float, type(None)) for k in v)
            json.dumps(m, allow_nan=False)

    def test_a_split_without_anomalies_has_null_recall_not_nan(self):
        m = policy_metrics(
            np.array([0, 2, 1, 0]),
            np.zeros(4, int),
            cost_matrix(PolicyConfig().costs),
            np.array([0, 0, 1, 1]),
            n_boot=20,
            seed=0,
        )
        assert m["anomaly_recall"] == {"value": None, "ci_low": None, "ci_high": None}
        json.dumps(m, allow_nan=False)
