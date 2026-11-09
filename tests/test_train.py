"""Tests for the training engine."""

import ast
import importlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.amp import GradScaler

from core.config import DignityConfig
from data.loader import create_dataloader
from models.dignity import Dignity
from train.checkpoint import build_model_from_meta, load_checkpoint
from train.cli import main
from train.engine import predict, train_epoch
from train.loop import (
    ChanceGuardWarning,
    FitResult,
    fit,
    prepare_data,
    resolve_pos_weight,
)
from train.losses import forecast_loss, risk_loss

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
    return tiny_fit_config


class TestFitForecast:
    def test_forecast_fit_saves_a_checkpoint_that_rebuilds_with_its_horizon(
        self, tiny_split, forecast_fit_config, tmp_path
    ):
        cfg = forecast_fit_config
        r = fit(cfg, tiny_split, seed=0, run_dir=str(tmp_path))
        assert r.meta.head_kwargs == {"horizon": cfg.model.horizon}
        assert 0.0 <= r.best_metric <= 1.0
        model = build_model_from_meta(load_checkpoint(r.best_path)["meta"])
        out, _ = model(torch.randn(2, 20, r.meta.input_size))
        assert out.shape == (2, cfg.model.horizon)

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
        r = fit(forecast_fit_config, tiny_split, seed=0, run_dir=str(tmp_path))
        metric = forecast_fit_config.train.select_metric
        assert metric == "pr_auc"
        assert r.best_metric == max(h[f"val_{metric}"] for h in r.history) == 0.9
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

    def test_forecast_config_trains_through_the_cli(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        raw = yaml.safe_load((ROOT / "config/smoke.yaml").read_text())
        raw["model"]["task"] = "forecast"
        path = tmp_path / "forecast_smoke.yaml"
        path.write_text(yaml.safe_dump(raw))
        DignityConfig.from_yaml(str(path))  # the edited preset is itself valid
        assert main(["--config", str(path)]) == 0


class TestPosWeight:
    def test_pos_weight_defaults_to_train_negative_to_positive_ratio(self):
        assert resolve_pos_weight(None, np.array([0, 0, 0, 1])) == 3.0
        assert resolve_pos_weight(2.5, np.array([0, 0, 0, 1])) == 2.5

    def test_pos_weight_refuses_a_train_split_without_positives(self):
        with pytest.raises(ValueError, match="no positive"):
            resolve_pos_weight(None, np.zeros(4))


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
