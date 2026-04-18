"""Regime classification head — calm / trending / volatile / crisis."""

import torch
import torch.nn as nn


class RegimeHead(nn.Module):
    """Classifies market regime from backbone context.

    Regimes (default 4): 0=calm, 1=trending, 2=volatile, 3=crisis.
    Output is a softmax probability vector — regimes are mutually exclusive.
    """

    def __init__(
        self,
        input_size: int,
        n_regimes: int = 4,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_regimes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, input_size] → regime_probs: [B, n_regimes] (softmax)."""
        return torch.softmax(self.net(context), dim=-1)
