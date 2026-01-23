"""PyTorch Dataset and DataLoader utilities."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple


class TransactionDataset(Dataset):
    """PyTorch Dataset for transaction sequences."""
    
    def __init__(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 device: str = 'cpu'):
        """
        Args:
            X: Sequences [n_samples, seq_len, n_features]
            y: Optional labels/targets
            device: Device to load tensors on
        """
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device) if y is not None else None
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def create_dataloader(X: np.ndarray,
                     y: Optional[np.ndarray] = None,
                     batch_size: int = 64,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     device: str = 'cpu') -> DataLoader:
    """
    Create a DataLoader from sequences.
    
    Args:
        X: Sequences [n_samples, seq_len, n_features]
        y: Optional labels/targets
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of dataloader workers
        device: Device to load data on
        
    Returns:
        DataLoader
    """
    dataset = TransactionDataset(X, y, device=device)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
