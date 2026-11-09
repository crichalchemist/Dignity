"""PyTorch Dataset and DataLoader utilities."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TransactionDataset(Dataset):
    """PyTorch Dataset for transaction sequences.

    Samples are dicts: ``{"x": float32 tensor, **targets}``. Targets keep their
    numpy dtype, and every tensor stays on the CPU (the engine moves batches).
    """

    def __init__(self, X: np.ndarray, targets: dict[str, np.ndarray] | None = None):
        """
        Args:
            X: Sequences [n_samples, seq_len, n_features]
            targets: Optional named per-sample targets, each [n_samples, ...]
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.targets = {k: torch.as_tensor(v) for k, v in (targets or {}).items()}

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.X[idx], **{k: v[idx] for k, v in self.targets.items()}}


def create_dataloader(
    X: np.ndarray,
    targets: dict[str, np.ndarray] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from sequences.

    Args:
        X: Sequences [n_samples, seq_len, n_features]
        targets: Optional named per-sample targets
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin host memory for faster CUDA transfer

    Returns:
        DataLoader yielding dict batches
    """
    dataset = TransactionDataset(X, targets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
