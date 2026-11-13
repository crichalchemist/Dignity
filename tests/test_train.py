"""Tests for the training engine."""

import ast
import copy
import dataclasses
import importlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler

from core.config import DignityConfig, PolicyConfig, PrivacyConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity
from models.head.policy import cost_matrix
from train.checkpoint import build_model_from_meta, load_checkpoint
from train.cli import main
from train.engine import predict, train_epoch
from train.loop import (
    ChanceGuardWarning,
    FitResult,
    ForecastNotEligibleError,
    _selection_metric,
    _task_loss,
    fit,
    load_backbone,
    prepare_data,
    resolve_pos_weight,
)
from train.losses import expected_cost_loss, forecast_loss, risk_loss

ROOT = Path(__file__).resolve().parents[1]


class TestTrainEpoch:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="autocast only refuses BCELoss on CUDA; CPU autocast is a no-op here",
    )
    def test_amp_risk_training_runs_on_gpu(self):
        # RiskHead ends in a sigmoid and the risk criterion is BCELoss, which
        # PyTorch bans inside an autocast region ("unsafe to autocast"). Every
        # shipped config sets use_amp: true, so the loss has to be computed
        # outside autocast in float32 or GPU training of the risk task crashes
        # on the first batch.
        device = torch.device("cuda")
        torch.manual_seed(0)
        X = torch.randn(8, 20, 3).numpy()
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).float().numpy()
        loader = create_dataloader(X, {"y": y}, batch_size=4, shuffle=False)
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_epoch(
            model,
            loader,
            optimizer,
            risk_loss(None),
            device,
            GradScaler("cuda", enabled=True),
            use_amp=True,
            grad_clip=1.0,
        )

        assert math.isfinite(metrics["loss"])

    def test_training_runs_on_label_free_batches(self):
        X = np.random.randn(6, 20, 3).astype("float32")
        loader = create_dataloader(X, None, batch_size=4, shuffle=False)
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        m = train_epoch(
            model,
            loader,
            opt,
            lambda out, batch: out.mean(),
            torch.device("cpu"),
            GradScaler("cuda", enabled=False),
            use_amp=False,
            grad_clip=1.0,
        )
        assert math.isfinite(m["loss"])

    def test_the_callers_grad_scaler_steps_every_batch(self):
        class CountingScaler(GradScaler):
            steps = 0

            def step(self, optimizer, *a, **k):
                CountingScaler.steps += 1
                return super().step(optimizer, *a, **k)

        X, y = np.random.randn(8, 20, 3), np.array([0, 1] * 4)
        loader = create_dataloader(X, {"y": y}, batch_size=4, shuffle=False)
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_epoch(
            model,
            loader,
            opt,
            risk_loss(None),
            torch.device("cpu"),
            CountingScaler("cpu", enabled=False),
            use_amp=False,
            grad_clip=1.0,
        )
        assert CountingScaler.steps == 2

    def test_forecast_trains_one_cpu_epoch_with_masked_targets(self):
        X = np.random.randn(8, 20, 3).astype("float32")
        targets = {
            "y": (np.random.rand(8, 4) > 0.5).astype("float32"),
            "mask": np.array([True, False] * 4),
        }
        loader = create_dataloader(X, targets, batch_size=4, shuffle=False)
        model = Dignity(
            task="forecast", input_size=3, hidden_size=16, n_layers=1, horizon=4
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        m = train_epoch(
            model,
            loader,
            opt,
            forecast_loss(None),
            torch.device("cpu"),
            GradScaler("cuda", enabled=False),
            use_amp=False,
            grad_clip=1.0,
        )
        assert math.isfinite(m["loss"])

    def test_policy_trains_one_cpu_epoch_on_expected_cost(self):
        X = np.random.randn(8, 20, 3).astype("float32")
        targets = {"y": (np.random.rand(8) > 0.5).astype("int64")}
        loader = create_dataloader(X, targets, batch_size=4, shuffle=False)
        model = Dignity(task="policy", input_size=3, hidden_size=16, n_layers=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        m = train_epoch(
            model,
            loader,
            opt,
            expected_cost_loss(torch.from_numpy(cost_matrix(PolicyConfig().costs))),
            torch.device("cpu"),
            GradScaler("cuda", enabled=False),
            use_amp=False,
            grad_clip=1.0,
        )
        assert math.isfinite(m["loss"])


class TestPredict:
    def test_scores_every_unlabelled_window_even_with_a_two_row_tail(self):
        # DGNTY-31 repro: 66 rows at batch 64 leave a final batch of 2.
        loader = create_dataloader(
            np.random.randn(66, 20, 3), None, batch_size=64, shuffle=False
        )
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1)
        score = predict(model, loader, torch.device("cpu"))["score"]
        assert score.shape == (66,) and ((score >= 0) & (score <= 1)).all()

    def test_forecast_predictions_give_one_score_per_horizon(self):
        loader = create_dataloader(
            np.random.randn(8, 20, 3), None, batch_size=4, shuffle=False
        )
        model = Dignity(
            task="forecast", input_size=3, hidden_size=16, n_layers=1, horizon=4
        )
        score = predict(model, loader, torch.device("cpu"))["score"]
        assert score.shape == (8, 4) and score.dtype == np.float32

    def test_policy_predictions_are_probabilities_over_three_actions(self):
        loader = create_dataloader(
            np.random.randn(8, 20, 3), None, batch_size=4, shuffle=False
        )
        model = Dignity(task="policy", input_size=3, hidden_size=16, n_layers=1)
        probs = predict(model, loader, torch.device("cpu"))["probs"]
        assert probs.shape == (8, 3)
        assert np.allclose(probs.sum(1), 1, atol=1e-5)


class TestExpectedCostLoss:
    def test_undecided_policy_pays_the_average_cost_of_each_state(self):
        C = torch.from_numpy(cost_matrix(PolicyConfig().costs))
        loss = expected_cost_loss(C)
        z = torch.zeros(1, 3)
        assert loss(z, {"y": torch.tensor([1])}).item() == pytest.approx(11.5 / 3)
        assert loss(z, {"y": torch.tensor([0])}).item() == pytest.approx(4.0 / 3)

    def test_confident_cheapest_actions_cost_what_the_table_says(self):
        loss = expected_cost_loss(torch.from_numpy(cost_matrix(PolicyConfig().costs)))
        z = torch.tensor([[50.0, 0.0, 0.0], [0.0, 0.0, 50.0]])  # allow, block
        assert loss(z, {"y": torch.tensor([0, 1])}).item() == pytest.approx(0.25)


class TestForecastLoss:
    def test_windows_with_an_active_anomaly_do_not_train_the_warning(self):
        p = torch.tensor([[0.5, 0.5], [0.01, 0.01]])
        batch = {
            "y": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            "mask": torch.tensor([True, False]),
        }
        assert forecast_loss(None)(p, batch).item() == pytest.approx(math.log(2))

    def test_pos_weight_scales_only_positive_terms(self):
        p = torch.tensor([[0.5, 0.5]])
        batch = {"y": torch.tensor([[1.0, 0.0]]), "mask": torch.tensor([True])}
        assert forecast_loss(3.0)(p, batch).item() == pytest.approx(2 * math.log(2))

    def test_batch_of_only_active_anomaly_windows_gives_zero_not_nan(self):
        p = torch.full((2, 3), 0.3, requires_grad=True)
        batch = {"y": torch.ones(2, 3), "mask": torch.tensor([False, False])}
        loss = forecast_loss(None)(p, batch)
        assert loss.item() == 0.0
        loss.backward()  # still part of the graph


class TestRiskLoss:
    def test_unweighted_loss_is_plain_bce_for_either_head_shape(self):
        batch = {"y": torch.tensor([1, 0])}
        for out in (torch.tensor([[0.8], [0.3]]), torch.tensor([0.8, 0.3])):
            assert risk_loss(None)(out, batch).item() == pytest.approx(
                0.28991, abs=1e-5
            )

    def test_pos_weight_scales_only_the_positive_term(self):
        out, batch = torch.tensor([[0.8], [0.3]]), {"y": torch.tensor([1.0, 0.0])}
        assert risk_loss(3.0)(out, batch).item() == pytest.approx(0.51305, abs=1e-5)


class TestTrainCli:
    def test_cpu_run_forwards_loader_knobs_and_never_enables_amp(
        self, tiny_train_config, monkeypatch, recwarn, tmp_path
    ):
        import train.loop

        calls, real = [], train.loop.create_dataloader

        def spy(*a, **k):
            calls.append(k)
            return real(*a, **{**k, "num_workers": 0})  # no worker processes in tests

        monkeypatch.setattr(train.loop, "create_dataloader", spy)
        main(["--config", str(tiny_train_config(data={"num_workers": 3}))])
        assert [c["num_workers"] for c in calls] == [3, 3]
        assert all(c["pin_memory"] is False for c in calls)
        assert not [
            w
            for w in recwarn
            if "GradScaler" in str(w.message) or "autocast" in str(w.message)
        ]
        assert (tmp_path / "runs" / "best.pt").exists()

    def test_one_grad_scaler_per_run_and_disabled_on_cpu(
        self, tiny_train_config, monkeypatch
    ):
        import train.loop

        made, real = [], train.loop.GradScaler

        def spy(*a, **k):
            s = real(*a, **k)
            made.append(s)
            return s

        monkeypatch.setattr(train.loop, "GradScaler", spy)
        main(["--config", str(tiny_train_config(train={"epochs": 2}))])
        assert len(made) == 1 and made[0].is_enabled() is False

    def test_cli_forwards_resume_to_fit(self, tiny_train_config, monkeypatch):
        import train.cli

        seen = {}

        def fake_fit(config, data, seed, run_dir, resume_from=None):
            seen["resume_from"] = resume_from
            return FitResult("x", [], 0.0, None)

        monkeypatch.setattr(train.cli, "fit", fake_fit)
        cfg = str(tiny_train_config())
        assert main(["--config", cfg, "--resume", "x.pt"]) == 0
        assert seen["resume_from"] == "x.pt"

    def test_cli_trains_tiny_config_and_writes_a_loadable_best_checkpoint(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)  # smoke.yaml paths are relative
        assert main(["--config", str(ROOT / "config/smoke.yaml")]) == 0
        ckpt = load_checkpoint(str(tmp_path / "runs/smoke/train/best.pt"))
        assert ckpt["meta"].task == "risk" and ckpt["epoch"] == 1

    @pytest.mark.parametrize("task", ["risk", "forecast", "policy"])
    def test_every_task_trains_from_the_cli(self, task, tmp_path):
        raw = yaml.safe_load((ROOT / "config/smoke.yaml").read_text())
        if task == "policy":
            # the documented flow: risk first, then the policy warm-starts from it
            raw["model"]["task"] = "risk"
            raw["train"]["checkpoint_dir"] = str(tmp_path / "risk")
            (tmp_path / "r.yaml").write_text(yaml.safe_dump(raw))
            assert main(["--config", str(tmp_path / "r.yaml")]) == 0
            raw["train"]["init_backbone_from"] = str(tmp_path / "risk" / "best.pt")
        raw["model"]["task"] = task
        raw["train"]["checkpoint_dir"] = str(tmp_path / "run")
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump(raw))
        assert main(["--config", str(cfg)]) == 0
        assert load_checkpoint(tmp_path / "run" / "best.pt")["meta"].task == task

    def test_documented_cli_pair_trains_under_the_shipped_privacy_block(self, tmp_path):
        # privacy noise is unseeded, so the policy run's scaler never equals the
        # risk run's exactly; the warm start must still accept the pair
        raw = yaml.safe_load((ROOT / "config/smoke.yaml").read_text())
        raw["privacy"] = yaml.safe_load((ROOT / "config/train_risk.yaml").read_text())[
            "privacy"
        ]
        raw["data"]["synthetic"]["n_blocks"] = 200  # less noisy scaler than 60
        raw["train"]["checkpoint_dir"] = str(tmp_path / "risk")
        (tmp_path / "r.yaml").write_text(yaml.safe_dump(raw))
        assert main(["--config", str(tmp_path / "r.yaml")]) == 0
        raw["model"]["task"] = "policy"
        raw["train"]["checkpoint_dir"] = str(tmp_path / "policy")
        raw["train"]["init_backbone_from"] = str(tmp_path / "risk" / "best.pt")
        (tmp_path / "p.yaml").write_text(yaml.safe_dump(raw))
        assert main(["--config", str(tmp_path / "p.yaml")]) == 0


def _script_val_metrics(monkeypatch, pr_auc, auroc):
    """Replace the val metrics with scripted per-epoch values (one call each)."""
    calls = iter(zip(pr_auc, auroc, strict=True))

    def scripted(y, score, block_index, n_boot, seed, anomaly_type=None):
        pr, au = next(calls)
        return {
            "auroc": {"value": au},
            "pr_auc": {"value": pr},
            "recall_at_1pct_fpr": {"value": 0.5},
            "brier": {"value": 0.25},
        }

    monkeypatch.setattr("train.loop.risk_metrics", scripted)


class TestFit:
    def test_best_checkpoint_is_the_epoch_with_highest_val_pr_auc(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        _script_val_metrics(monkeypatch, [0.6, 0.9, 0.7], [0.9, 0.6, 0.95])
        r = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.best_metric == 0.9
        assert load_checkpoint(r.best_path)["epoch"] == 2

    def test_best_checkpoint_follows_the_configured_select_metric(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        _script_val_metrics(monkeypatch, [0.6, 0.9, 0.7], [0.9, 0.6, 0.95])
        tiny_fit_config.train.select_metric = "auroc"
        r = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.best_metric == 0.95
        assert load_checkpoint(r.best_path)["epoch"] == 3

    def test_a_run_that_never_selects_a_best_checkpoint_fails_loudly(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        nan = float("nan")
        _script_val_metrics(monkeypatch, [nan] * 3, [0.9, 0.6, 0.95])
        with pytest.raises(RuntimeError, match="pr_auc"):
            fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_a_stale_best_checkpoint_is_not_mistaken_for_this_runs(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        # J2: another run's best.pt sitting in run_dir must not satisfy a fit that
        # itself never selected a checkpoint
        fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert (tmp_path / "best.pt").exists()
        nan = float("nan")
        _script_val_metrics(monkeypatch, [nan] * 3, [0.9, 0.6, 0.95])
        with pytest.raises(RuntimeError, match="pr_auc"):
            fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_a_run_leaves_a_self_describing_record_of_every_epoch(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        r = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.meta.input_size == 3 and r.meta.feature_names == ["a", "b", "c"]
        assert set(r.history[0]) == {
            "epoch",
            "lr",
            "train_loss",
            "val_auroc",
            "val_pr_auc",
            "val_recall_at_1pct_fpr",
            "val_brier",
        }
        assert json.loads((tmp_path / "history.json").read_text()) == r.history
        assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()

    def test_early_stopping_halts_after_patience_epochs_without_improvement(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.lr = 0.0
        tiny_fit_config.train.epochs, tiny_fit_config.train.patience = 10, 1
        r = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert len(r.history) == 2
        assert load_checkpoint(str(tmp_path / "last.pt"))["epoch"] == 2

    def test_the_exit_epoch_is_saved_even_off_the_save_interval(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.lr = 0.0
        tiny_fit_config.train.epochs, tiny_fit_config.train.patience = 10, 1
        tiny_fit_config.train.save_interval = 5
        fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert load_checkpoint(str(tmp_path / "last.pt"))["epoch"] == 2

    def test_the_final_epoch_is_saved_even_off_the_save_interval(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.patience = None
        tiny_fit_config.train.save_interval = 5
        fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert load_checkpoint(str(tmp_path / "last.pt"))["epoch"] == 3

    def test_chance_guard_warns_once_when_val_auroc_stays_below_floor(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.chance_guard_auroc = 1.01
        tiny_fit_config.train.guard_epochs = 1
        with pytest.warns(ChanceGuardWarning, match="chance guard") as rec:
            fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert sum(w.category is ChanceGuardWarning for w in rec) == 1

    def test_learning_rate_follows_a_decaying_schedule(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        r = fit(tiny_fit_config, tiny_split, 0, str(tmp_path))
        # epochs=3, base lr 3e-4: the lr each epoch trained with
        assert [h["lr"] for h in r.history] == pytest.approx(
            [3e-4, 3e-4 * 0.75, 3e-4 * 0.25]
        )

    def test_resumed_run_continues_the_same_lr_schedule(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        full = fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "full"))
        tiny_fit_config.train.epochs = 2
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "split"))
        tiny_fit_config.train.epochs = 3
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # scheduler-order warning
            resumed = fit(
                tiny_fit_config,
                tiny_split,
                0,
                str(tmp_path / "split"),
                resume_from=str(tmp_path / "split" / "last.pt"),
            )
        assert resumed.history[-1]["lr"] == pytest.approx(full.history[2]["lr"])

    def test_resume_keeps_old_history_rows_and_returns_only_new_ones(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.epochs = 2
        first = fit(tiny_fit_config, tiny_split, 0, str(tmp_path))
        tiny_fit_config.train.epochs = 3
        resumed = fit(
            tiny_fit_config,
            tiny_split,
            0,
            str(tmp_path),
            resume_from=str(tmp_path / "last.pt"),
        )
        assert [h["epoch"] for h in resumed.history] == [3]
        saved = json.loads((tmp_path / "history.json").read_text())
        assert saved == first.history + resumed.history

    def test_resume_does_not_overwrite_better_checkpoint(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.train.lr, tiny_fit_config.train.epochs = 0.0, 2
        first = fit(tiny_fit_config, tiny_split, 0, str(tmp_path))
        tiny_fit_config.train.epochs = 3
        fit(
            tiny_fit_config,
            tiny_split,
            0,
            str(tmp_path),
            resume_from=str(tmp_path / "last.pt"),
        )
        # an equal epoch 3 must not replace it
        assert load_checkpoint(first.best_path)["epoch"] == 1


def _script_forecast_val_metrics(monkeypatch, pr_auc, auroc):
    """Scripted per-epoch forecast val metrics: one risk_metrics call per horizon."""
    horizon = 10  # tiny_split labels are 10 wide and every column has both classes
    calls = {"n": 0}

    def scripted(y, score, block_index, n_boot, seed, anomaly_type=None):
        epoch = calls["n"] // horizon
        calls["n"] += 1
        return {
            "auroc": {"value": auroc[epoch]},
            "pr_auc": {"value": pr_auc[epoch]},
            "recall_at_1pct_fpr": {"value": 0.5},
            "brier": {"value": 0.25},
        }

    monkeypatch.setattr("train.loop.risk_metrics", scripted)


def _script_forecast_columns(monkeypatch, pr_auc, auroc):
    """Per-column scripted val metrics, same values every epoch.

    risk_metrics is called once per scorable horizon, in column order, so call k of
    an epoch gets pr_auc[k] / auroc[k]. Returns the class counts it was called with.
    """
    seen = []

    def scripted(y, score, block_index, n_boot, seed, anomaly_type=None):
        k = len(seen) % len(auroc)
        seen.append(len(np.unique(y)))
        return {
            "auroc": {"value": auroc[k]},
            "pr_auc": {"value": pr_auc[k]},
            "recall_at_1pct_fpr": {"value": 0.5},
            "brier": {"value": 0.25},
        }

    monkeypatch.setattr("train.loop.risk_metrics", scripted)
    return seen


@pytest.fixture
def forecast_fit_config(tiny_fit_config):
    tiny_fit_config.model.task = "forecast"
    tiny_fit_config.model.horizon = 10
    # tiny random data promises no lift over the risk score; the gate has its own tests
    tiny_fit_config.train.forecast_eligibility = False
    return tiny_fit_config


def _script_lift(monkeypatch, values):
    """Scripted per-epoch val lift_upper: one call per epoch."""
    calls = {"n": 0}

    def scripted(y, score, mask):
        value = values[calls["n"]]
        calls["n"] += 1
        return value

    monkeypatch.setattr("train.loop.lift_upper", scripted)


class TestFitForecast:
    def test_forecast_fit_saves_a_checkpoint_that_rebuilds_with_its_horizon(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        cfg = forecast_fit_config
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.meta.head_kwargs == {"horizon": cfg.model.horizon}
        # forecast keeps the epoch with the best val lift_upper, not a 0-1 metric
        assert r.best_metric == max(h["val_lift_upper"] for h in r.history) > 0.0
        model = build_model_from_meta(load_checkpoint(r.best_path)["meta"])
        out, _ = model(torch.randn(2, 20, r.meta.input_size))
        assert out.shape == (2, cfg.model.horizon)

    def test_forecast_warm_start_copies_only_backbone_tensors(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        cfg = forecast_fit_config
        cfg.model.task = "risk"
        cfg.train.epochs = 1
        risk = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path / "risk"))
        cfg.model.task = "forecast"
        # lr 0 so that training cannot move a weight away from its start
        cfg.train.lr = 0.0
        cfg.train.init_backbone_from = risk.best_path
        fc = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path / "forecast"))
        rs = load_checkpoint(risk.best_path)["model_state"]
        fs = load_checkpoint(fc.best_path)["model_state"]
        backbone = [k for k in fs if k.startswith("backbone.")]
        assert backbone
        for k in backbone:
            assert k in rs and torch.equal(fs[k], rs[k]), k
        last = [k for k in fs if k.startswith("head.") and k.endswith(".weight")][-1]
        last_risk = [k for k in rs if k.startswith("head.") and k.endswith(".weight")]
        assert fs[last].shape[0] == 10 and rs[last_risk[-1]].shape[0] == 1
        shared = [
            k
            for k in fs
            if k.startswith("head.") and k in rs and fs[k].shape == rs[k].shape
        ]
        assert shared
        for k in shared:
            assert not torch.equal(fs[k], rs[k]), k

    def test_forecast_pos_weight_refuses_a_train_split_without_onsets(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        tiny_split.train.y_forecast[:] = 0
        with pytest.raises(ValueError, match="no positive"):
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_forecast_best_checkpoint_is_the_epoch_with_highest_val_metric(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        # scripted values: selection by the configured metric differs from
        # "keep the first", "keep the last" and "select on the other metric"
        _script_forecast_val_metrics(monkeypatch, [0.6, 0.9, 0.7], [0.9, 0.6, 0.95])
        _script_lift(monkeypatch, [1.6, 1.9, 1.7])
        r = fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        metric = _selection_metric(forecast_fit_config)
        assert metric == "lift_upper"
        assert r.best_metric == max(h[f"val_{metric}"] for h in r.history) == 1.9
        assert load_checkpoint(r.best_path)["epoch"] == 2

    def test_forecast_history_rows_carry_the_per_horizon_mean_metrics(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        r = fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert set(r.history[0]) == {
            "epoch",
            "lr",
            "train_loss",
            "val_auroc",
            "val_pr_auc",
            "val_recall_at_1pct_fpr",
            "val_lift_upper",
        }
        assert json.loads((tmp_path / "history.json").read_text()) == r.history

    def test_forecast_val_skips_horizons_without_both_classes(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        # horizon 0 has no positive among masked val rows; the 9 scorable columns
        # get distinct scripted values so only a mean over them gives 0.45 / 0.7
        pr = [0.05 + 0.1 * k for k in range(9)]
        au = [0.9 - 0.05 * k for k in range(9)]
        seen = _script_forecast_columns(monkeypatch, pr, au)
        tiny_split.val.y_forecast[:, 0] = 0
        r = fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert seen and set(seen) == {2}
        assert len(seen) == 3 * 9  # 3 epochs, 9 scorable horizons
        for row in r.history:
            assert row["val_pr_auc"] == pytest.approx(0.45)
            assert row["val_auroc"] == pytest.approx(0.7)

    def test_forecast_val_with_no_positive_in_any_horizon_fails_loudly(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        tiny_split.val.y_forecast[:] = 0
        with pytest.raises(RuntimeError, match="no best checkpoint"):
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_forecast_chance_guard_fires_when_only_the_last_horizon_is_below(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        # mean AUROC 0.82 and column 0 are fine; only column H-1 is under 0.5
        _script_forecast_columns(monkeypatch, [0.5] * 10, [0.9] * 9 + [0.1])
        forecast_fit_config.train.chance_guard_auroc = 0.5
        forecast_fit_config.train.guard_epochs = 1
        with pytest.warns(ChanceGuardWarning, match="chance guard") as rec:
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert sum(w.category is ChanceGuardWarning for w in rec) == 1

    def test_forecast_chance_guard_ignores_a_weak_first_horizon(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        _script_forecast_columns(monkeypatch, [0.5] * 10, [0.1] + [0.9] * 9)
        forecast_fit_config.train.chance_guard_auroc = 0.5
        forecast_fit_config.train.guard_epochs = 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", ChanceGuardWarning)
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_forecast_chance_guard_is_skipped_when_the_last_horizon_is_one_class(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        # every scorable column is below the guard, but column H-1 cannot be scored
        _script_forecast_columns(monkeypatch, [0.5] * 9, [0.1] * 9)
        tiny_split.val.y_forecast[:, 9] = 0
        forecast_fit_config.train.chance_guard_auroc = 0.5
        forecast_fit_config.train.guard_epochs = 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", ChanceGuardWarning)
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_an_epoch_that_does_not_beat_the_risk_score_on_val_is_never_kept(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        cfg = forecast_fit_config
        cfg.train.epochs = 4
        cfg.train.forecast_eligibility = True
        # pr_auc makes epoch 1 look best; its lift 1.4 is below the floor 1.5
        _script_forecast_val_metrics(
            monkeypatch, [0.9, 0.5, 0.5, 0.5], [0.9, 0.5, 0.5, 0.5]
        )
        _script_lift(monkeypatch, [1.4, 2.0, 1.2, 1.6])
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path), eligibility_floor=1.5)
        assert load_checkpoint(r.best_path)["epoch"] == 2
        assert r.best_epoch == 2
        assert r.best_metric == 2.0

    def test_forecast_selection_follows_lift_upper_whatever_select_metric_says(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        cfg = forecast_fit_config
        cfg.train.select_metric = "auroc"
        _script_forecast_val_metrics(monkeypatch, [0.5] * 3, [0.95, 0.6, 0.6])
        _script_lift(monkeypatch, [1.2, 1.8, 1.5])
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path))
        assert _selection_metric(cfg) == "lift_upper"
        assert load_checkpoint(r.best_path)["epoch"] == 2
        assert r.best_epoch == 2

    def test_without_a_risk_checkpoint_chance_lift_is_not_eligible(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        forecast_fit_config.train.forecast_eligibility = True
        _script_lift(monkeypatch, [1.0, 1.0, 1.0])
        # 1.0 is chance, not above the default floor of 1.0
        with pytest.raises(ForecastNotEligibleError):
            fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert not (tmp_path / "best.pt").exists()

    def test_without_a_risk_checkpoint_the_first_epoch_above_chance_is_kept(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        forecast_fit_config.train.forecast_eligibility = True
        forecast_fit_config.train.epochs = 2
        _script_lift(monkeypatch, [0.9, 1.05])
        r = fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.best_epoch == 2

    def test_fit_fails_loudly_when_no_epoch_is_eligible(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        forecast_fit_config.train.forecast_eligibility = True
        _script_lift(monkeypatch, [1.1, 1.1, 1.1])
        with pytest.raises(ForecastNotEligibleError, match="1.1") as exc:
            fit(
                forecast_fit_config,
                tiny_split,
                seed=0,
                run_dir=str(tmp_path),
                eligibility_floor=1.5,
            )
        err = exc.value
        assert "1.5" in str(err)
        assert err.seed == 0
        assert err.run_dir == str(tmp_path)
        assert isinstance(err, RuntimeError)
        assert not (tmp_path / "best.pt").exists()

    def test_patience_counts_only_after_the_first_eligible_epoch(
        self, tiny_split, forecast_fit_config, tmp_path, monkeypatch
    ):
        cfg = forecast_fit_config
        cfg.train.forecast_eligibility = True
        cfg.train.epochs = 6
        cfg.train.patience = 2
        _script_lift(monkeypatch, [1.0, 1.0, 1.0, 1.6, 1.5, 1.4])
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path), eligibility_floor=1.5)
        # epochs 1-3 are not eligible and do not count as stale; 5 and 6 do
        assert len(r.history) == 6
        assert r.best_epoch == 4

    def test_a_resumed_forecast_run_compares_against_the_checkpoints_own_selection_metric(
        self, tiny_split, forecast_fit_config, tiny_fit_config, tmp_path
    ):
        cfg = forecast_fit_config
        cfg.train.epochs = 1
        fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path / "f"))
        cfg.train.epochs = 2
        r = fit(
            cfg,
            tiny_split,
            seed=0,
            run_dir=str(tmp_path / "f"),
            resume_from=str(tmp_path / "f" / "last.pt"),
        )
        assert [h["epoch"] for h in r.history] == [2]

        risk = DignityConfig()
        risk.device = "cpu"
        risk.model.hidden_size = 16
        risk.model.n_layers = 1
        risk.data.batch_size = 32
        risk.data.num_workers = 0
        risk.train.epochs = 1
        risk.train.use_amp = False
        risk.train.save_interval = 1
        fit(risk, tiny_split, seed=0, run_dir=str(tmp_path / "r"))
        with pytest.raises(ValueError, match="selected on"):
            fit(
                cfg,
                tiny_split,
                seed=0,
                run_dir=str(tmp_path / "r2"),
                resume_from=str(tmp_path / "r" / "last.pt"),
            )

    def test_forecast_config_trains_through_the_cli(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        raw = yaml.safe_load((ROOT / "config/smoke.yaml").read_text())
        raw["model"]["task"] = "forecast"
        path = tmp_path / "forecast_smoke.yaml"
        path.write_text(yaml.safe_dump(raw))
        DignityConfig.from_yaml(str(path))  # the edited preset is itself valid
        assert main(["--config", str(path)]) == 0


def _script_policy_actions(monkeypatch, action_per_epoch):
    """Replace decide_actions with one scripted action per epoch (every window).

    Returns the review budgets it was called with.
    """
    budgets, calls = [], iter(action_per_epoch)

    def scripted(probs, review_budget):
        budgets.append(review_budget)
        return np.full(len(probs), next(calls), dtype=np.int64)

    monkeypatch.setattr("train.loop.decide_actions", scripted)
    return budgets


@pytest.fixture
def policy_fit_config(tiny_fit_config):
    tiny_fit_config.model.task = "policy"
    return tiny_fit_config


class TestFitPolicy:
    def test_policy_fit_saves_a_checkpoint_that_rebuilds_with_three_actions(
        self, tiny_split, policy_fit_config, tmp_path
    ):
        cfg = policy_fit_config
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.meta.head_kwargs == {"n_actions": 3}
        C = cost_matrix(cfg.policy.costs)
        assert -C.max() <= r.best_metric <= 0.0  # neg_cost
        out, _ = build_model_from_meta(r.meta)(
            torch.randn(2, cfg.data.seq_len, r.meta.input_size)
        )
        assert out.shape == (2, 3)

    def test_policy_selects_the_epoch_with_the_lowest_capped_validation_cost(
        self, tiny_split, policy_fit_config, tmp_path, monkeypatch
    ):
        # J11: costs are label-free (allow 2.0, review 0.5, block 1.0), so the
        # scripted epoch actions (allow, review, block) give neg_cost -2.0, -0.5,
        # -1.0; only real selection on the capped decision lands on epoch 2.
        cfg = policy_fit_config
        cfg.policy.costs = {
            "allow": {"anomaly": 2.0, "normal": 2.0},
            "review": {"anomaly": 0.5, "normal": 0.5},
            "block": {"anomaly": 1.0, "normal": 1.0},
        }
        budgets = _script_policy_actions(monkeypatch, [0, 1, 2])
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path))
        assert [h["val_neg_cost"] for h in r.history] == [-2.0, -0.5, -1.0]
        assert r.best_metric == -0.5
        assert load_checkpoint(r.best_path)["epoch"] == 2
        assert budgets == [cfg.policy.review_budget] * 3

    def test_policy_history_rows_carry_only_the_cost_metric(
        self, tiny_split, policy_fit_config, tmp_path
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ChanceGuardWarning)  # guard is skipped
            r = fit(policy_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        assert set(r.history[0]) == {"epoch", "lr", "train_loss", "val_neg_cost"}

    def test_a_policy_run_that_never_improves_names_neg_cost_in_its_error(
        self, tiny_split, policy_fit_config, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "train.loop._val_metrics",
            lambda *a, **k: ({"val_neg_cost": float("nan")}, None),
        )
        with pytest.raises(RuntimeError, match="neg_cost"):
            fit(policy_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))

    def test_resume_refuses_a_checkpoint_selected_on_another_metric(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        # a risk checkpoint's best_metric is a PR-AUC; comparing a policy
        # neg_cost against it would silently never save a best.pt
        policy_cfg = copy.deepcopy(tiny_fit_config)
        policy_cfg.model.task = "policy"
        risk = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path / "r"))
        last = str(tmp_path / "r" / "last.pt")
        assert risk.meta.task == "risk"
        with pytest.raises(ValueError, match="neg_cost"):
            fit(
                policy_cfg,
                tiny_split,
                0,
                str(tmp_path / "p"),
                resume_from=last,
            )

    def test_a_policy_run_resumes_from_its_own_checkpoint(
        self, tiny_split, policy_fit_config, tmp_path
    ):
        policy_fit_config.train.epochs = 2
        fit(policy_fit_config, tiny_split, 0, str(tmp_path))
        policy_fit_config.train.epochs = 3
        resumed = fit(
            policy_fit_config,
            tiny_split,
            0,
            str(tmp_path),
            resume_from=str(tmp_path / "last.pt"),
        )
        assert [h["epoch"] for h in resumed.history] == [3]

    def test_resume_refuses_an_m1_era_checkpoint_without_select_metric(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        # M1-era meta has no select_metric: best_metric may be any metric, so
        # resuming would compare a PR-AUC against an unknown number (J6)
        tiny_fit_config.train.epochs = 1
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path))
        last = tmp_path / "last.pt"
        raw = torch.load(last, weights_only=True)
        del raw["meta"]["config"]["train"]["select_metric"]
        torch.save(raw, last)
        tiny_fit_config.train.epochs = 2
        with pytest.raises(ValueError, match="was selected on None"):
            fit(tiny_fit_config, tiny_split, 0, str(tmp_path), resume_from=str(last))


def _backbone_state(path):
    state = load_checkpoint(path)["model_state"]
    return {k: v for k, v in state.items() if k.startswith("backbone.")}


class TestLoadBackbone:
    @pytest.fixture
    def risk_best(self, tiny_split, tiny_fit_config, tmp_path):
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        r = fit(tiny_fit_config, tiny_split, seed=0, run_dir=str(tmp_path / "risk"))
        return r.best_path

    def _policy_model(self, hidden_size=16):
        return Dignity(task="policy", input_size=3, hidden_size=hidden_size, n_layers=1)

    def test_warm_start_copies_the_risk_backbone_and_leaves_the_head_alone(
        self, risk_best, tiny_split, tiny_fit_config
    ):
        torch.manual_seed(1)
        model = self._policy_model()
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
        load_backbone(model, risk_best, tiny_split, None, tiny_fit_config.data)
        want = _backbone_state(risk_best)
        assert want
        got = model.state_dict()
        for k, v in want.items():
            assert torch.equal(got[k], v), k
        for k, v in model.head.state_dict().items():
            assert torch.equal(v, head_before[k]), k

    def test_warm_start_refuses_a_checkpoint_trained_on_other_features(
        self, risk_best, tiny_split, tiny_fit_config
    ):
        other = dataclasses.replace(tiny_split, feature_names=["a", "b", "z"])
        with pytest.raises(ValueError, match="feature"):
            load_backbone(
                self._policy_model(), risk_best, other, None, tiny_fit_config.data
            )

    def test_warm_start_refuses_a_backbone_of_another_width(
        self, risk_best, tiny_split, tiny_fit_config
    ):
        with pytest.raises(ValueError, match="shape"):
            load_backbone(
                self._policy_model(hidden_size=8),
                risk_best,
                tiny_split,
                None,
                tiny_fit_config.data,
            )

    def test_warm_start_refuses_a_backbone_fitted_to_differently_scaled_inputs(
        self, risk_best, tiny_split, tiny_fit_config
    ):
        # a backbone fitted to inputs scaled 111x differently would warm-start
        # silently: the tensors all fit, only the numbers it was trained on differ
        rescaled = dataclasses.replace(
            tiny_split, scaler_scale=tiny_split.scaler_scale * 111
        )
        with pytest.raises(ValueError, match="scaler_scale"):
            load_backbone(
                self._policy_model(), risk_best, rescaled, None, tiny_fit_config.data
            )
        shifted = dataclasses.replace(
            tiny_split, scaler_center=tiny_split.scaler_center + 1
        )
        with pytest.raises(ValueError, match="scaler_center"):
            load_backbone(
                self._policy_model(), risk_best, shifted, None, tiny_fit_config.data
            )

    def test_warm_start_refuses_a_backbone_fitted_under_another_privacy_config(
        self, risk_best, tiny_split, tiny_fit_config
    ):
        # the checkpoint was fitted without a privacy stage
        private = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 10]}
            },
        )
        with pytest.raises(ValueError, match="privacy"):
            load_backbone(
                self._policy_model(),
                risk_best,
                tiny_split,
                private,
                tiny_fit_config.data,
            )

    def test_under_privacy_scaler_noise_is_tolerated_but_a_2x_mismatch_is_not(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.privacy = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 10]}
            },
        )
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        best = fit(tiny_fit_config, tiny_split, 0, str(tmp_path)).best_path
        privacy = tiny_fit_config.privacy
        noisy = dataclasses.replace(
            tiny_split,
            scaler_center=tiny_split.scaler_center + 0.04,
            scaler_scale=tiny_split.scaler_scale * 1.05,
        )
        load_backbone(self._policy_model(), best, noisy, privacy, tiny_fit_config.data)
        doubled = dataclasses.replace(
            tiny_split, scaler_scale=tiny_split.scaler_scale * 2
        )
        with pytest.raises(ValueError, match="scaler_scale"):
            load_backbone(
                self._policy_model(), best, doubled, privacy, tiny_fit_config.data
            )

    def test_warm_start_refuses_a_seeded_checkpoint_for_a_system_mode_run(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        # seeded-stream noise and system noise are different releases; a backbone
        # fitted on one must not silently warm-start a run on the other
        features = {
            "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 10]}
        }
        tiny_fit_config.privacy = PrivacyConfig(
            epsilon_total=1.0, deterministic_rng=True, features=features
        )
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        best = fit(tiny_fit_config, tiny_split, 0, str(tmp_path)).best_path
        system = PrivacyConfig(epsilon_total=1.0, features=features)
        with pytest.raises(ValueError, match="privacy"):
            load_backbone(
                self._policy_model(), best, tiny_split, system, tiny_fit_config.data
            )

    @pytest.mark.parametrize(
        "change, key",
        [
            (lambda d: setattr(d.synthetic, "seed", d.synthetic.seed + 1), "synthetic"),
            (
                lambda d: setattr(d.synthetic, "n_blocks", d.synthetic.n_blocks + 50),
                "synthetic",
            ),
        ],
    )
    def test_warm_start_refuses_a_backbone_trained_on_another_dataset(
        self, tiny_split, tiny_fit_config, tmp_path, change, key
    ):
        # same privacy block, scaler within the noise tolerance: only the data
        # config tells that risk-train blocks would land in this run's val/test
        tiny_fit_config.privacy = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 10]}
            },
        )
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        best = fit(tiny_fit_config, tiny_split, 0, str(tmp_path)).best_path
        other = copy.deepcopy(tiny_fit_config.data)
        change(other)
        with pytest.raises(ValueError, match=f"data config.*{key}"):
            load_backbone(
                self._policy_model(),
                best,
                tiny_split,
                tiny_fit_config.privacy,
                other,
            )

    def test_warm_start_accepts_the_same_privacy_config_it_was_fitted_under(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.privacy = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 10]}
            },
        )
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        best = fit(tiny_fit_config, tiny_split, 0, str(tmp_path)).best_path
        load_backbone(
            self._policy_model(),
            best,
            tiny_split,
            tiny_fit_config.privacy,
            tiny_fit_config.data,
        )
        with pytest.raises(ValueError, match="privacy"):
            load_backbone(
                self._policy_model(), best, tiny_split, None, tiny_fit_config.data
            )

    def test_warm_start_refuses_a_deeper_backbone_instead_of_loading_layer_zero(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        # strict=False once dropped the extra layers, so a 2-layer risk
        # checkpoint silently warm-started a 1-layer policy from layer 0 only
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.model.n_layers = 2
        tiny_fit_config.train.epochs = 1
        deep = fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "deep")).best_path
        with pytest.raises(ValueError, match="extra"):
            load_backbone(
                self._policy_model(), deep, tiny_split, None, tiny_fit_config.data
            )

    def test_warm_start_names_the_missing_backbone_tensors(
        self, tiny_split, tiny_fit_config, tmp_path
    ):
        tiny_fit_config.model.task = "risk"
        tiny_fit_config.train.epochs = 1
        best = fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "r")).best_path
        model = Dignity(task="policy", input_size=3, hidden_size=16, n_layers=2)
        with pytest.raises(ValueError, match="missing"):
            load_backbone(model, best, tiny_split, None, tiny_fit_config.data)

    def test_resume_restores_full_state_so_the_warm_start_is_skipped(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        tiny_fit_config.model.task = "policy"
        tiny_fit_config.train.epochs = 1
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path))
        calls = []
        monkeypatch.setattr("train.loop.load_backbone", lambda *a: calls.append(a))
        tiny_fit_config.train.epochs = 2
        tiny_fit_config.train.init_backbone_from = str(tmp_path / "gone.pt")
        resumed = fit(
            tiny_fit_config,
            tiny_split,
            0,
            str(tmp_path),
            resume_from=str(tmp_path / "last.pt"),
        )
        assert calls == [] and [h["epoch"] for h in resumed.history] == [2]

    def test_only_policy_and_forecast_fits_with_a_configured_path_warm_start(
        self, tiny_split, tiny_fit_config, tmp_path, monkeypatch
    ):
        calls = []
        monkeypatch.setattr("train.loop.load_backbone", lambda *a: calls.append(a))
        tiny_fit_config.train.epochs = 1
        tiny_fit_config.train.init_backbone_from = "risk.pt"
        tiny_fit_config.model.task = "risk"
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "risk"))
        assert calls == []

        # a stubbed warm start leaves a tiny random model: no eligibility gate here
        tiny_fit_config.model.task = "forecast"
        tiny_fit_config.train.forecast_eligibility = False
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "forecast"))
        assert len(calls) == 1 and calls[0][0].task == "forecast"

        tiny_fit_config.model.task = "policy"
        tiny_fit_config.train.init_backbone_from = None
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "cold"))
        assert len(calls) == 1

        tiny_fit_config.train.init_backbone_from = "risk.pt"
        fit(tiny_fit_config, tiny_split, 0, str(tmp_path / "warm"))
        assert len(calls) == 2
        model, path, data, privacy, data_cfg = calls[1]
        assert model.task == "policy" and path == "risk.pt"
        assert data is tiny_split and privacy is None
        assert data_cfg == tiny_fit_config.data


class TestPosWeight:
    def test_pos_weight_defaults_to_train_negative_to_positive_ratio(self):
        assert resolve_pos_weight(None, np.array([0, 0, 0, 1])) == 3.0
        assert resolve_pos_weight(2.5, np.array([0, 0, 0, 1])) == 2.5

    def test_pos_weight_refuses_a_train_split_without_positives(self):
        with pytest.raises(ValueError, match="no positive"):
            resolve_pos_weight(None, np.zeros(4))

    def test_forecast_positive_weight_is_capped_at_the_configured_ceiling(self):
        y = np.array([0] * 52 + [1])
        assert resolve_pos_weight(None, y) == 52.0
        assert resolve_pos_weight(None, y, cap=10.0) == 10.0
        assert resolve_pos_weight(None, y, cap=100.0) == 52.0
        # an explicit weight is the owner's choice, never capped
        assert resolve_pos_weight(2.5, y, cap=1.0) == 2.5

    def test_the_forecast_loss_is_built_with_the_capped_weight(
        self, tiny_split, forecast_fit_config
    ):
        cfg = forecast_fit_config
        train = tiny_split.train
        train.y_forecast[:] = 0
        train.y_forecast[np.flatnonzero(train.forecast_mask)[0], 0] = 1
        derived = resolve_pos_weight(None, train.y_forecast[train.forecast_mask])
        assert derived > 10.0
        batch = {
            "y": torch.tensor([[1.0] + [0.0] * 9]),
            "mask": torch.tensor([True]),
        }
        p = torch.full((1, 10), 0.5)
        capped_loss = forecast_loss(10.0)(p, batch).item()

        cfg.train.forecast_pos_weight_cap = 10.0
        assert _task_loss(cfg, train)(p, batch).item() == capped_loss
        cfg.train.forecast_pos_weight_cap = None
        assert _task_loss(cfg, train)(p, batch).item() != capped_loss


class TestNotebook:
    def test_notebook_trains_through_shared_fit_and_keeps_colab_badge(self):
        # The notebook once duplicated the CLI loop and drifted from it (spec 5.3).
        nb = json.loads((ROOT / "notebooks/train_risk.ipynb").read_text())
        code = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        )
        md = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        )
        assert "colab.research.google.com/assets/colab-badge.svg" in md
        assert "from train.loop import fit" in code
        assert "train_epoch(" not in code and "validate_epoch(" not in code

    def test_every_notebook_import_resolves(self):
        # A reader who opens the Colab badge runs these imports first; a renamed or
        # deleted function must fail here, not in their browser.
        nb = json.loads((ROOT / "notebooks/train_risk.ipynb").read_text())
        roots = {"core", "data", "models", "train", "eval", "export"}
        missing = []
        for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
            src = "\n".join(
                line
                for line in "".join(cell["source"]).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            for node in ast.walk(ast.parse(src)):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module.split(".")[0] in roots
                ):
                    mod = importlib.import_module(node.module)
                    missing += [
                        f"{node.module}.{a.name}"
                        for a in node.names
                        if not hasattr(mod, a.name)
                    ]
        assert missing == []


def _seeded_volume_privacy() -> PrivacyConfig:
    return PrivacyConfig(
        epsilon_total=1.0,
        deterministic_rng=True,
        features={
            "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 1000]}
        },
    )


class TestPrepareData:
    def test_every_split_is_windowed_alike_and_blocks_never_straddle_splits(
        self, tiny_train_config
    ):
        from core.config import DignityConfig

        config = DignityConfig.from_yaml(tiny_train_config())
        _, split = prepare_data(config)
        sets = (split.train, split.val, split.test)
        assert {w.X.ndim for w in sets} == {3}
        assert len({w.X.shape[1] for w in sets}) == 1
        assert {w.X.shape[2] for w in sets} == {len(split.feature_names)}
        train_blocks = set(split.train.block_index.tolist())
        assert train_blocks.isdisjoint(split.val.block_index.tolist())
        assert train_blocks.isdisjoint(split.test.block_index.tolist())
        assert set(split.val.block_index.tolist()).isdisjoint(
            split.test.block_index.tolist()
        )

    def test_the_pipeline_is_seeded_with_exactly_the_data_seed(
        self, tiny_train_config, monkeypatch
    ):
        import train.loop as loop

        seen = {}

        class Spy(loop.TransactionPipeline):
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(loop, "TransactionPipeline", Spy)
        config = DignityConfig.from_yaml(tiny_train_config())
        config.data.synthetic.seed = 7
        prepare_data(config)
        assert seen["privacy_seed"] == 7

    def test_seeded_privacy_mode_reproduces_the_privatized_train_windows(
        self, tiny_train_config
    ):
        # prepare_data must hand the data seed to the pipeline: without it the
        # deterministic_rng config cannot run, and a rerun would not match.
        config = DignityConfig.from_yaml(tiny_train_config())
        config.privacy = PrivacyConfig(
            epsilon_total=1.0,
            deterministic_rng=True,
            features={
                "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 1000]}
            },
        )
        _, first = prepare_data(config)
        _, second = prepare_data(config)
        np.testing.assert_array_equal(first.train.X, second.train.X)

    def test_the_default_privacy_seed_is_the_data_seed(self, tiny_train_config):
        # every caller that omits privacy_seed keeps the noise it had before the
        # argument existed
        config = DignityConfig.from_yaml(tiny_train_config())
        config.privacy = _seeded_volume_privacy()
        _, default = prepare_data(config)
        _, explicit = prepare_data(config, privacy_seed=config.data.synthetic.seed)
        np.testing.assert_array_equal(default.train.X, explicit.train.X)

    def test_a_new_privacy_seed_changes_only_the_noise(self, tiny_train_config):
        # a sweep replicate must re-noise the same data, never re-draw or re-split it
        config = DignityConfig.from_yaml(tiny_train_config())
        config.privacy = _seeded_volume_privacy()
        seed = config.data.synthetic.seed
        _, base = prepare_data(config)
        _, other = prepare_data(config, privacy_seed=seed + 1)
        assert not np.array_equal(base.train.X, other.train.X)
        np.testing.assert_array_equal(base.train.y_risk, other.train.y_risk)
        np.testing.assert_array_equal(base.train.block_index, other.train.block_index)
        np.testing.assert_array_equal(base.test.block_index, other.test.block_index)


# Declared 2026-11-09 from tuning-seed validation evidence, before the evaluation
# seed was touched (val AUROC min 0.735 at eps 1 over privacy seeds 0-2; chance is
# 0.5). Easy-preset data: a "still learns" floor, not a point on the privacy curve.
# That evidence predates R3, which gave the easy preset build-ups and decoys; the
# floor was not re-derived on that data, and the test still passes at 0.70.
PRIVACY_AUROC_FLOOR = 0.70


class TestPrivacyUtility:
    def test_privacy_config_still_learns(self, easy_config, tmp_path):
        # The shipped privacy block, loaded from config/train_risk.yaml itself, must
        # leave enough signal for the risk model to learn. Easy data isolates the
        # privacy stage's cost from generator difficulty and keeps this CPU-fast.
        shipped = DignityConfig.from_yaml(str(ROOT / "config/train_risk.yaml"))
        cfg = dataclasses.replace(
            shipped,
            device="cpu",
            privacy=dataclasses.replace(shipped.privacy, deterministic_rng=True),
            data=dataclasses.replace(
                shipped.data,
                batch_size=128,
                split=(0.7, 0.15, 0.15),  # 150 blocks: keep 105 to train on
                num_workers=0,
                synthetic=dataclasses.replace(
                    easy_config.data.synthetic,
                    n_blocks=150,
                    seed=shipped.eval.tuning_seed,
                ),
            ),
            model=dataclasses.replace(shipped.model, hidden_size=32),
            train=dataclasses.replace(
                shipped.train,
                lr=1e-3,
                epochs=10,
                patience=None,
                chance_guard_auroc=None,
                checkpoint_dir=str(tmp_path),
            ),
        )
        df = SyntheticGenerator(cfg.data.synthetic).generate()
        data = TransactionPipeline(
            seq_len=cfg.data.seq_len,
            features=cfg.data.features,
            privacy=cfg.privacy,
            privacy_seed=0,
        ).process_blocks(
            df,
            cfg.data.split,
            cfg.model.horizon,
            cfg.data.label_overlap,
            np.random.default_rng(0),
        )
        result = fit(cfg, data, seed=0, run_dir=str(tmp_path))
        best = load_checkpoint(result.best_path)
        model = build_model_from_meta(best["meta"])
        model.load_state_dict(best["model_state"])
        loader = create_dataloader(data.val.X, batch_size=128, shuffle=False)
        scores = predict(model, loader, torch.device("cpu"))["score"]
        auroc = roc_auc_score(data.val.y_risk, scores)
        assert auroc >= PRIVACY_AUROC_FLOOR, f"AUROC {auroc:.3f} under privacy"

    def test_shipped_privacy_config_trains_through_the_cli_in_system_mode(
        self, tmp_path, capsys
    ):
        raw = yaml.safe_load((ROOT / "config/train_risk.yaml").read_text())
        assert "deterministic_rng" not in raw["privacy"]  # shipped = SystemRandom
        # 80 blocks at contamination 0.5 and a pinned seed: val and test hold both
        # classes
        raw["data"]["synthetic"] = {
            **raw["data"].get("synthetic", {}),
            "n_blocks": 80,
            "contamination": 0.5,
            "seed": 0,
        }
        raw["data"].update(num_workers=0, batch_size=128)
        raw["model"].update(hidden_size=16, n_layers=1)
        raw["train"].update(
            epochs=1,
            checkpoint_dir=str(tmp_path / "run"),
            chance_guard_auroc=None,
            patience=None,
        )
        raw["device"] = "cpu"
        (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
        assert main(["--config", str(tmp_path / "c.yaml")]) == 0
        assert "Privacy: mode=system;" in capsys.readouterr().out
        meta = load_checkpoint(tmp_path / "run" / "best.pt")["meta"]
        assert meta.config["privacy"]["deterministic_rng"] is False
        assert (
            meta.config["privacy"]["features"].keys()
            == raw["privacy"]["features"].keys()
        )
