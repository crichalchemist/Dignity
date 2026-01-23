"""Training utilities including early stopping and checkpoint management."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting during training."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        """Initialize early stopping.

        Parameters
        ----------
        patience : int
            Number of epochs to wait after last time validation metric improved.
        min_delta : float
            Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop based on the score.

        Parameters
        ----------
        score : float
            Validation score to monitor (higher is better for accuracy, lower for loss).

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        elif score > self.best_score + self.min_delta:
            # Improvement found
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False

    def reset(self) -> None:
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class CheckpointManager:
    """Manage model checkpoints with top-N retention policy."""

    def __init__(self, save_dir: Path, top_n: int = 3):
        """Initialize checkpoint manager.

        Parameters
        ----------
        save_dir : Path
            Directory to save checkpoints.
        top_n : int
            Number of best checkpoints to retain.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = top_n
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, state_dict: dict, score: float, epoch: int, model_name: str = "model") -> None:
        """Save checkpoint and manage retention policy.

        Parameters
        ----------
        state_dict : Dict
            Model state dictionary to save.
        score : float
            Validation score for this checkpoint.
        epoch : int
            Current training epoch.
        model_name : str
            Base name for the checkpoint file.
        """
        checkpoint_path = self.save_dir / f"{model_name}_epoch{epoch}_score{score:.4f}.pt"

        # Clone tensors to avoid references that might change, keep non-tensors as-is
        cpu_state = {}
        for k, v in state_dict.items():
            if hasattr(v, 'cpu'):
                cpu_state[k] = v.cpu()
            else:
                cpu_state[k] = v
        torch.save(cpu_state, checkpoint_path)

        # Add to checkpoint list and sort by score (descending for accuracy, ascending for loss)
        self.checkpoints.append((score, checkpoint_path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove old checkpoints beyond top_n
        while len(self.checkpoints) > self.top_n:
            _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                logger.info(f"Removed old checkpoint: {old_path}")

        logger.info(f"Saved checkpoint: {checkpoint_path} (score: {score:.4f})")

    def get_best_checkpoint(self) -> Path | None:
        """Get the path to the best checkpoint.

        Returns
        -------
        Optional[Path]
            Path to the best checkpoint, or None if no checkpoints exist.
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]

    def cleanup(self) -> None:
        """Remove all managed checkpoints."""
        for _, checkpoint_path in self.checkpoints:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        self.checkpoints.clear()
        logger.info("Cleaned up all checkpoints")


class MetricComparator:
    """Helper class for comparing metrics based on task type."""

    def __init__(self, task_type: str = "classification"):
        """Initialize metric comparator.

        Parameters
        ----------
        task_type : str
            Type of task: "classification" (higher is better) or "regression" (lower is better).
        """
        self.task_type = task_type

    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best.

        Parameters
        ----------
        current : float
            Current metric value.
        best : float
            Best metric value so far.

        Returns
        -------
        bool
            True if current is better than best.
        """
        if self.task_type == "classification":
            return current > best
        else:
            return current < best

    def initialize_best(self) -> float:
        """Initialize best metric value.

        Returns
        -------
        float
            Initial best metric value.
        """
        if self.task_type == "classification":
            return -float("inf")
        else:
            return float("inf")
