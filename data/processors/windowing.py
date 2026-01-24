"""Sequence windowing dataset for PyTorch.

This module provides a PyTorch Dataset for creating sliding window sequences
from time series data, with configurable window size and stride.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """
    PyTorch Dataset for sliding window sequences over time series data.

    Creates overlapping windows from input data array with configurable
    window size and stride. Compatible with PyTorch DataLoader for batching.

    Args:
        data: Input array of shape (T, F) where T=timesteps, F=features
        sequence_length: Length of each window in timesteps (default: 10000)
        stride: Step size between consecutive windows (default: 1000)

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(50000, 15)  # 50k timesteps, 15 features
        >>> dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        >>> len(dataset)  # Number of windows: (50000 - 10000) // 1000 + 1
        41
        >>> window = dataset[0]  # First window
        >>> window.shape
        torch.Size([10000, 15])
        >>> 
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=True)
        >>> batch = next(iter(loader))
        >>> batch.shape
        torch.Size([4, 10000, 15])
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10000,
        stride: int = 1000
    ):
        """Initialize the windowing dataset.

        Args:
            data: Input array of shape (T, F)
            sequence_length: Length of each window in timesteps
            stride: Step size between consecutive windows

        Raises:
            ValueError: If data is too short for at least one window
            ValueError: If sequence_length or stride are not positive
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be np.ndarray, got {type(data)}")
        
        if len(data.shape) != 2:
            raise ValueError(f"data must be 2D array (T, F), got shape {data.shape}")
        
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        
        if len(data) < sequence_length:
            raise ValueError(
                f"data length ({len(data)}) must be >= sequence_length ({sequence_length})"
            )
        
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Compute number of windows: (T - seq_len) // stride + 1
        self.num_windows = (len(data) - sequence_length) // stride + 1
    
    def __len__(self) -> int:
        """Return the number of windows."""
        return self.num_windows
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a window by index.

        Args:
            idx: Window index (0 to num_windows - 1)

        Returns:
            Window tensor of shape (sequence_length, features) as Float32

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_windows} windows"
            )
        
        # Compute start index for this window
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Slice the window and convert to tensor
        window = self.data[start_idx:end_idx, :]
        
        # Convert to Float32 tensor
        return torch.from_numpy(window).float()
