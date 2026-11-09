import json
import time

import numpy as np
import pandas as pd
import pytest

import eval.metrics as metrics_module
from eval.baselines import block_tree, risk_score_forecast
from eval.metrics import (
    bootstrap_ci,
    forecast_metrics,
    lead_times,
    model_gate,
    risk_metrics,
    threshold_at_fpr,
)
from eval.plots import plot_per_type_recall, plot_pr_curves
from eval.run import config_hash, main

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

    @pytest.mark.parametrize("task", ["forecast", "policy"])
    def test_unbuilt_tasks_are_refused_not_silently_skipped(
        self, task, capsys, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)

        def boom(*args, **kwargs):
            raise AssertionError("must not run")

        monkeypatch.setattr("eval.run.run_risk", boom)
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--seed", "tuning", "--tasks", task])
        assert exc.value.code == 2
        assert "not implemented until M4" in capsys.readouterr().err

    def test_smoke_preset_cannot_be_combined_with_the_evaluation_seed(
        self, monkeypatch
    ):
        # run_risk is booby-trapped so the refusal must happen before any work
        def boom(*args, **kwargs):
            raise AssertionError("run_risk reached")

        monkeypatch.setattr("eval.run.run_risk", boom)
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--seed", "eval"])
        assert exc.value.code == 2

    def test_config_and_smoke_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            main(["--smoke", "--config", "config/smoke.yaml", "--seed", "tuning"])
        assert exc.value.code == 2

    def test_one_hash_identifies_one_configuration(self):
        from core.config import DignityConfig

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
