"""Early stopping and checkpoint management utilities for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EarlyStopping:
    """Early stopping handler to halt training when validation score plateaus.

    Monitors a metric (e.g., validation loss) and stops training if it doesn't
    improve for `patience` consecutive checks.

    Parameters
    ----------
    patience : int
        Number of checks without improvement before stopping.
    min_delta : float, optional
        Minimum change to qualify as an improvement. Default: 0.0.
    mode : str, optional
        Either 'min' (lower is better) or 'max' (higher is better). Default: 'min'.
    """

    patience: int
    min_delta: float = 0.0
    mode: str = 'min'

    def __post_init__(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        score : float
            Current metric value to compare.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        is_improvement = self._is_improvement(score)

        if not is_improvement:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if score represents an improvement over best_score."""
        threshold = self.best_score + (self.min_delta if self.mode == 'max' else -self.min_delta)
        if self.mode == 'min':
            return score < threshold
        else:  # mode == 'max'
            return score > threshold

    def reset(self) -> None:
        """Reset early stopping state for a new training run."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class CheckpointManager:
    """Manages checkpoint saving with top-N retention policy.

    Automatically saves checkpoints and keeps only the top N by score,
    deleting older checkpoints to avoid disk bloat.

    Parameters
    ----------
    save_dir : Path or str
        Directory to save checkpoints in.
    top_n : int, optional
        Number of best checkpoints to retain. Default: 3.
    mode : str, optional
        Either 'min' (lower is better) or 'max' (higher is better). Default: 'min'.
    """

    def __init__(self, save_dir: Path | str, top_n: int = 3, mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = top_n
        self.mode = mode
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, state_dict: dict, score: float, epoch: int, suffix: str = '') -> Path:
        """Save a checkpoint and manage retention.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary to save.
        score : float
            Metric score for ranking (e.g., validation loss).
        epoch : int
            Current epoch number for naming.
        suffix : str, optional
            Additional suffix for checkpoint filename. Default: ''.

        Returns
        -------
        Path
            Path to saved checkpoint.
        """
        filename = f"checkpoint_epoch{epoch}_score{score:.4f}{suffix}.pt"
        path = self.save_dir / filename

        torch.save(state_dict, path)
        self.checkpoints.append((score, path))

        # Sort by score: best first
        reverse = self.mode == 'max'
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Delete old checkpoints beyond top_n
        while len(self.checkpoints) > self.top_n:
            _, old_path = self.checkpoints.pop()
            old_path.unlink(missing_ok=True)

        return path

    def load_best(self) -> dict | None:
        """Load the best checkpoint found so far.

        Returns
        -------
        dict or None
            State dictionary of best checkpoint, or None if no checkpoints saved yet.
        """
        if not self.checkpoints:
            return None
        _, best_path = self.checkpoints[0]
        return torch.load(best_path, weights_only=False)

    def best_checkpoint_path(self) -> Path | None:
        """Get path to the best checkpoint.

        Returns
        -------
        Path or None
            Path to best checkpoint, or None if no checkpoints saved yet.
        """
        if not self.checkpoints:
            return None
        _, best_path = self.checkpoints[0]
        return best_path

    def clear(self) -> None:
        """Delete all managed checkpoints and reset state."""
        for _, path in self.checkpoints:
            path.unlink(missing_ok=True)
        self.checkpoints.clear()
