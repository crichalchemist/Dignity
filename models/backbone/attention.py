"""Additive attention mechanism for sequence weighting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.

    Computes attention weights over sequence to produce
    a weighted context vector focusing on important timesteps.
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Size of hidden representations
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Attention parameters
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Compute attention-weighted context vector.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            mask: Optional mask [batch_size, seq_len] (1 = keep, 0 = mask)

        Returns:
            Tuple of:
            - context: Attention-weighted sum [batch_size, hidden_size]
            - weights: Attention weights [batch_size, seq_len]
        """
        # Compute attention scores
        # [B, T, H] -> [B, T, H] -> [B, T, 1]
        scores = self.v(torch.tanh(self.W(hidden_states)))
        scores = scores.squeeze(-1)  # [B, T]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        weights = F.softmax(scores, dim=1)  # [B, T]

        # Compute weighted context
        # [B, T] -> [B, T, 1] * [B, T, H] -> [B, T, H] -> [B, H]
        context = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)

        return context, weights
