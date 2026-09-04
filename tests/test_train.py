"""Tests for the training engine."""

import torch
import torch.nn as nn

from data.loader import create_dataloader
from models.dignity import Dignity
from train.engine import validate_epoch


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
