"""Tests for the training engine."""

import math

import numpy as np
import pytest
import torch
from torch.amp import GradScaler

from data.loader import create_dataloader
from models.dignity import Dignity
from train.checkpoint import load_checkpoint
from train.engine import predict, train_epoch
from train.losses import risk_loss


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


class TestPredict:
    def test_scores_every_unlabelled_window_even_with_a_two_row_tail(self):
        # DGNTY-31 repro: 66 rows at batch 64 leave a final batch of 2.
        loader = create_dataloader(
            np.random.randn(66, 20, 3), None, batch_size=64, shuffle=False
        )
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1)
        score = predict(model, loader, torch.device("cpu"))["score"]
        assert score.shape == (66,) and ((score >= 0) & (score <= 1)).all()


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
        import train.cli

        calls, real = [], train.cli.create_dataloader

        def spy(*a, **k):
            calls.append(k)
            return real(*a, **{**k, "num_workers": 0})  # no worker processes in tests

        monkeypatch.setattr(train.cli, "create_dataloader", spy)
        train.cli.main(["--config", str(tiny_train_config(data={"num_workers": 3}))])
        assert [c["num_workers"] for c in calls] == [3, 3]
        assert all(c["pin_memory"] is False for c in calls)
        assert not [
            w
            for w in recwarn
            if "GradScaler" in str(w.message) or "autocast" in str(w.message)
        ]
        assert (tmp_path / "runs" / "dignity_risk_best.pt").exists()

    def test_one_grad_scaler_per_run_and_disabled_on_cpu(
        self, tiny_train_config, monkeypatch
    ):
        import train.cli

        made, real = [], train.cli.GradScaler

        def spy(*a, **k):
            s = real(*a, **k)
            made.append(s)
            return s

        monkeypatch.setattr(train.cli, "GradScaler", spy)
        train.cli.main(["--config", str(tiny_train_config(train={"epochs": 2}))])
        assert len(made) == 1 and made[0].is_enabled() is False

    def test_resume_does_not_overwrite_better_checkpoint(
        self, tiny_train_config, tmp_path
    ):
        import train.cli

        runs = tmp_path / "runs"
        train.cli.main(["--config", str(tiny_train_config(name="e1"))])
        for name in ("dignity_risk_epoch1.pt", "dignity_risk_best.pt"):
            raw = torch.load(runs / name, weights_only=True)
            raw["best_metric"] = 1e-6  # unbeatable: BCE on real data is far above it
            torch.save(raw, runs / name)
        cfg2 = tiny_train_config(name="e2", train={"epochs": 2})
        train.cli.main(
            ["--config", str(cfg2), "--resume", str(runs / "dignity_risk_epoch1.pt")]
        )
        assert load_checkpoint(runs / "dignity_risk_best.pt")["epoch"] == 1
        assert load_checkpoint(runs / "dignity_risk_epoch2.pt")["best_metric"] == 1e-6
