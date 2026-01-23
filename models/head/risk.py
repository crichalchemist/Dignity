"""Risk scoring head for transaction risk assessment."""

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """
    Task head for transaction risk scoring.

    Outputs a risk score between 0 and 1:
    - 0: Low risk (normal transaction)
    - 1: High risk (suspicious/anomalous)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.1):
        """
        Args:
            input_size: Size of backbone context vector
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute risk score from context vector.

        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Risk scores [batch_size, 1]
        """
        return self.net(context)
