"""Tests for training engine — cascade training path."""

import tempfile
import os

import torch
import torch.nn as nn
import pytest

from models.dignity import Dignity
from train.engine import (
    make_cosine_scheduler,
    save_checkpoint,
    load_checkpoint,
    train_cascade_epoch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_WEIGHTS = {"regime": 0.2, "risk": 0.3, "alpha": 0.3, "policy": 0.2}


def _cascade_loader(n_batches: int = 3, B: int = 4, seq_len: int = 50, features: int = 32):
    """Return a list acting as a DataLoader: each item is (x, cascade_labels)."""
    batches = []
    for _ in range(n_batches):
        x = torch.randn(B, seq_len, features)
        labels = {
            "regime": torch.randint(0, 4, (B,)),
            "var": torch.rand(B, 1),
            "alpha": torch.randn(B, 1).clamp(-1, 1),
            "action": torch.randint(0, 3, (B,)),
        }
        batches.append((x, labels))
    return batches


def _small_cascade_model():
    return Dignity(task="cascade", input_size=32, hidden_size=64, n_layers=1)


# ---------------------------------------------------------------------------
# TestTrainCascadeEpoch
# ---------------------------------------------------------------------------

class TestTrainCascadeEpoch:

    def test_returns_dict(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        assert isinstance(metrics, dict)

    def test_returns_total_loss_key(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        assert "loss" in metrics

    def test_returns_per_head_loss_keys(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        for key in ("regime_loss", "risk_loss", "alpha_loss", "policy_loss"):
            assert key in metrics, f"missing key: {key}"

    def test_loss_is_positive_float(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        assert isinstance(metrics["loss"], float) and metrics["loss"] > 0

    def test_model_parameters_updated(self):
        """Verify optimizer.step() actually fires — params must change."""
        model = _small_cascade_model()
        params_before = [p.clone().detach() for p in model.parameters()]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        train_cascade_epoch(
            model, _cascade_loader(n_batches=2), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        params_after = list(model.parameters())
        changed = any(
            not torch.equal(b, a.detach())
            for b, a in zip(params_before, params_after)
        )
        assert changed, "no model parameters changed after training step"

    def test_scheduler_stepped_when_provided(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = make_cosine_scheduler(opt, T_max=10)
        lr_before = opt.param_groups[0]["lr"]
        train_cascade_epoch(
            model, _cascade_loader(n_batches=1), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False, scheduler=scheduler,
        )
        lr_after = opt.param_groups[0]["lr"]
        # After one scheduler.step(), LR should have changed
        assert lr_after != lr_before

    def test_model_in_train_mode_after_epoch(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_cascade_epoch(
            model, _cascade_loader(), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
        )
        assert model.training


# ---------------------------------------------------------------------------
# TestMakeCosineScheduler
# ---------------------------------------------------------------------------

class TestMakeCosineScheduler:

    def test_returns_lr_scheduler(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = make_cosine_scheduler(opt, T_max=100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_lr_decreases_monotonically_over_half_period(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = make_cosine_scheduler(opt, T_max=10, eta_min=1e-6)
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # First half should be strictly decreasing
        half = lrs[:6]
        assert all(a >= b for a, b in zip(half, half[1:]))

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_lr_reaches_eta_min_at_T_max(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        eta_min = 1e-6
        sched = make_cosine_scheduler(opt, T_max=5, eta_min=eta_min)
        for _ in range(5):
            sched.step()
        assert pytest.approx(opt.param_groups[0]["lr"], abs=1e-8) == eta_min


# ---------------------------------------------------------------------------
# TestRiskGateTraining
# ---------------------------------------------------------------------------

class TestRiskGateTraining:

    def test_runs_with_gate_enabled(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(n_batches=2), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
            risk_gate_training=True, max_drawdown=0.03,
        )
        assert "loss" in metrics and metrics["loss"] > 0

    def test_runs_with_gate_disabled(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(n_batches=2), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
            risk_gate_training=False,
        )
        assert "loss" in metrics and metrics["loss"] > 0

    def test_zero_max_drawdown_fires_gate_on_all_samples(self):
        """max_drawdown=0 forces gate on every sample — training must still produce finite loss."""
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_cascade_epoch(
            model, _cascade_loader(n_batches=2), opt, _TASK_WEIGHTS,
            device=torch.device("cpu"), use_amp=False,
            risk_gate_training=True, max_drawdown=0.0,
        )
        assert metrics["loss"] > 0
        assert not (metrics["loss"] != metrics["loss"])  # NaN check without math.isnan


# ---------------------------------------------------------------------------
# Helpers for convergence and checkpoint tests
# ---------------------------------------------------------------------------

def _fixed_loader(n_batches: int = 4, B: int = 8, seq_len: int = 50, features: int = 32) -> list:
    """Pre-seeded dataset — same tensors every call, required for convergence assertions."""
    torch.manual_seed(42)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(B, seq_len, features)
        labels = {
            "regime": torch.randint(0, 4, (B,)),
            "var": torch.rand(B, 1),
            "alpha": torch.randn(B, 1).clamp(-1, 1),
            "action": torch.randint(0, 3, (B,)),
        }
        batches.append((x, labels))
    return batches


def _no_dropout_model() -> Dignity:
    return Dignity(task="cascade", input_size=32, hidden_size=64, n_layers=1, dropout=0.0)


# ---------------------------------------------------------------------------
# TestConvergenceSmoke
# ---------------------------------------------------------------------------

class TestConvergenceSmoke:
    """10-epoch regression gate — catches gradient flow breakage from future refactors."""

    @pytest.mark.slow
    def test_loss_decreases_over_10_epochs(self):
        torch.manual_seed(42)
        model = _no_dropout_model()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        loader = _fixed_loader()
        losses = []
        for _ in range(10):
            m = train_cascade_epoch(
                model, loader, opt, _TASK_WEIGHTS,
                device=torch.device("cpu"), use_amp=False, risk_gate_training=False,
            )
            losses.append(m["loss"])
        assert losses[-1] < losses[0], (
            f"loss did not decrease over 10 epochs: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    @pytest.mark.slow
    def test_no_nan_loss_over_10_epochs(self):
        torch.manual_seed(42)
        model = _no_dropout_model()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        loader = _fixed_loader()
        for epoch in range(10):
            m = train_cascade_epoch(
                model, loader, opt, _TASK_WEIGHTS,
                device=torch.device("cpu"), use_amp=False, risk_gate_training=False,
            )
            assert m["loss"] == m["loss"], f"NaN loss at epoch {epoch}"  # NaN != NaN

    @pytest.mark.slow
    def test_regime_accuracy_above_chance_after_10_epochs(self):
        torch.manual_seed(42)
        model = _no_dropout_model()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        loader = _fixed_loader()
        for _ in range(10):
            train_cascade_epoch(
                model, loader, opt, _TASK_WEIGHTS,
                device=torch.device("cpu"), use_amp=False, risk_gate_training=False,
            )
        model.train(False)
        correct = total = 0
        with torch.no_grad():
            for x, labels in loader:
                preds = model(x)["regime_probs"].argmax(dim=-1)
                correct += (preds == labels["regime"]).sum().item()
                total += labels["regime"].numel()
        accuracy = correct / total
        assert accuracy > 0.25, f"regime accuracy {accuracy:.3f} not above 4-class chance"


# ---------------------------------------------------------------------------
# TestCheckpointRoundTrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:

    def test_all_cascade_outputs_match_after_reload(self):
        torch.manual_seed(0)
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "test.pt")
            save_checkpoint(model, opt, epoch=1, metrics={"loss": 0.5}, path=ckpt_path)

            model2 = _small_cascade_model()
            load_checkpoint(model2, optimizer=None, path=ckpt_path, device=torch.device("cpu"))

            model.train(False)
            model2.train(False)
            x = torch.randn(2, 50, 32)
            with torch.no_grad():
                out1 = model(x)
                out2 = model2(x)

        for key in out1:
            assert torch.allclose(out1[key], out2[key]), f"output mismatch after reload: '{key}'"

    def test_checkpoint_contains_required_keys(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "test.pt")
            save_checkpoint(model, opt, epoch=3, metrics={"loss": 0.42}, path=ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        for key in ("epoch", "model_state_dict", "optimizer_state_dict", "metrics"):
            assert key in ckpt, f"checkpoint missing required key: '{key}'"

    def test_epoch_number_preserved(self):
        model = _small_cascade_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "test.pt")
            save_checkpoint(model, opt, epoch=7, metrics={}, path=ckpt_path)
            epoch = load_checkpoint(
                model, optimizer=None, path=ckpt_path, device=torch.device("cpu")
            )

        assert epoch == 7
