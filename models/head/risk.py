"""Risk assessment head — regime-conditioned dual output."""

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """Regime-conditioned risk head with dual sigmoid outputs.

    In the cascade, input is cat(context [B,256], regime_probs [B,4]) = [B, 260].
    Returns a tuple:
        var_estimate   [B, 1] — fraction of portfolio at risk (0..1)
        position_limit [B, 1] — allowed position size as fraction of max (0..1)

    Both are sigmoid-bounded so downstream gates can apply hard thresholds
    without clamping.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared trunk — learns a single risk representation
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Parallel output branches
        self.var_branch = nn.Sequential(nn.Linear(hidden_size // 2, 1), nn.Sigmoid())
        self.pos_branch = nn.Sequential(nn.Linear(hidden_size // 2, 1), nn.Sigmoid())

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """context: [B, input_size] → (var_estimate [B,1], position_limit [B,1])."""
        features = self.trunk(context)
        return self.var_branch(features), self.pos_branch(features)
