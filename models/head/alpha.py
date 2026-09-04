"""Alpha scoring head — risk-adjusted return prediction, regime-conditioned."""

import torch
import torch.nn as nn


class AlphaHead(nn.Module):
    """Predicts risk-adjusted alpha conditioned on regime context.

    Input is cat(backbone_context [B,256], regime_probs [B,4]) = [B, 260].
    Output is a scalar alpha score in [-1, 1] (tanh): positive = expected
    outperformance, negative = underperformance, zero = neutral.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, input_size] → alpha_score: [B, 1] in [-1, 1]."""
        return self.net(context)
