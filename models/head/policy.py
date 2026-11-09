"""Triage head: logits over ACTIONS = allow / review / block.

Trained cost-sensitively (train/losses.expected_cost_loss); ONNX consumers get
logits and apply softmax, then decide_actions.
"""

import numpy as np
import torch
import torch.nn as nn

ACTIONS = ("allow", "review", "block")


def cost_matrix(costs: dict[str, dict[str, float]]) -> np.ndarray:
    """Cost table as float32 [3, 2]; rows follow ACTIONS, column 0 is normal
    (y = 0) and column 1 is anomaly (y = 1), so C[a, y] indexes directly."""
    return np.array(
        [[costs[a]["normal"], costs[a]["anomaly"]] for a in ACTIONS], np.float32
    )


class PolicyHead(nn.Module):
    """Task head producing triage logits over ACTIONS."""

    def __init__(
        self,
        input_size: int,
        n_actions: int = 3,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_size: Size of backbone context vector
            n_actions: Number of actions; must equal len(ACTIONS)
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()

        if n_actions != len(ACTIONS):
            raise ValueError(
                f"n_actions={n_actions} but the cost table prices ACTIONS={ACTIONS}"
            )
        self.n_actions = n_actions

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )
        self.policy = nn.Linear(hidden_size, n_actions)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Logits [batch_size, n_actions]
        """
        return self.policy(self.shared(context))
