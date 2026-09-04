"""Tests for the training engine."""

import math

import pytest
import torch
import torch.nn as nn

from data.loader import create_dataloader
from models.dignity import Dignity
from train.engine import train_epoch, validate_epoch


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
        loader = create_dataloader(X, y, batch_size=4, shuffle=False)
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            criterion=nn.BCELoss(),
            device=device,
            use_amp=True,
        )

        assert math.isfinite(metrics["loss"])


class TestValidateEpoch:
    def test_reports_accuracy_for_binary_risk_labels(self):
        # The risk head emits [B, 1] probabilities that validate_epoch squeezes to
        # [B] before scoring, so the accuracy metric must key off the squeezed
        # shape. Without it dignity-train never prints "Val Accuracy" for risk.
        torch.manual_seed(0)
        X = torch.randn(8, 20, 3).numpy()
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).float().numpy()
        loader = create_dataloader(X, y, batch_size=4, shuffle=False)
        model = Dignity(task="risk", input_size=3, hidden_size=16, n_layers=1)

        metrics = validate_epoch(
            model=model,
            dataloader=loader,
            criterion=nn.BCELoss(),
            device=torch.device("cpu"),
        )

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
