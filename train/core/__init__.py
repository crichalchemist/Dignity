"""
Core training modules for the Sequence agent.

This package contains the primary training logic for:
- Single-task agent training
- Multi-task agent training
- Reinforcement learning agent training
"""

from .agent_train import train_model
from .agent_train_multitask import train_multitask

__all__ = [
    "train_model",
    "train_multitask",
]
