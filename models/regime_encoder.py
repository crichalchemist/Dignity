"""Lightweight MLP encoder for regime feature vectors."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeEncoder(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 32, emb_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x_regime: torch.Tensor) -> torch.Tensor:
        """
        x_regime: [batch_size, input_dim]
        returns:  [batch_size, emb_dim]
        """
        h = F.relu(self.fc1(x_regime))
        emb = F.relu(self.fc2(h))
        return emb
