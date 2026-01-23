"""Policy head for reinforcement learning (optional)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Task head for policy learning (A3C/PPO).

    Outputs:
    - Action logits (for discrete actions)
    - State value estimate

    This is optional and only used if doing RL-based training.
    """

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
            n_actions: Number of discrete actions
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()

        self.n_actions = n_actions

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )

        # Policy head (actor)
        self.policy = nn.Linear(hidden_size, n_actions)

        # Value head (critic)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, context: torch.Tensor) -> tuple:
        """
        Compute policy logits and value estimate.

        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Tuple of:
            - action_logits: [batch_size, n_actions]
            - value: [batch_size, 1]
        """
        shared_features = self.shared(context)

        action_logits = self.policy(shared_features)
        value = self.value(shared_features)

        return action_logits, value

    def sample_action(self, context: torch.Tensor) -> tuple:
        """
        Sample action from policy distribution.

        Args:
            context: Context vector [batch_size, input_size]

        Returns:
            Tuple of:
            - action: Sampled action [batch_size]
            - log_prob: Log probability [batch_size]
            - value: Value estimate [batch_size, 1]
        """
        action_logits, value = self.forward(context)

        # Sample from categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value
