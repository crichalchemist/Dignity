"""Early-warning head: P(a new anomaly starts within the next h rows), h = 1…H."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastHead(nn.Module):
    """
    Task head for early warning.

    Emits one probability per horizon h = 1…H, non-decreasing in h by
    construction (cumulative-hazard form): an onset inside (t, t+h] is also
    inside (t, t+h+1].
    """

    def __init__(
        self,
        input_size: int,
        horizon: int = 10,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_size: Size of backbone context vector
            horizon: Number of horizons H (rows ahead) to score
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()

        self.horizon = horizon

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

        # Start with p_H near 0.1 (z = 0 would give ~0.999).
        nn.init.constant_(
            self.net[-1].bias, math.log(math.expm1(-math.log(1 - 0.1) / horizon))
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Onset probabilities [batch_size, horizon], non-decreasing in h
        """
        z = self.net(context)
        return 1 - torch.exp(-torch.cumsum(F.softplus(z), dim=-1))
