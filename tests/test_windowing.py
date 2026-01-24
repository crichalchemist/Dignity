"""Tests for sequence windowing dataset."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.processors.windowing import SequenceWindowDataset


class TestSequenceWindowDataset:
    """Test sequence windowing dataset."""

    def test_initialization(self):
        """Test dataset can be instantiated."""
        data = np.random.randn(20000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        assert dataset is not None
        assert dataset.sequence_length == 10000
        assert dataset.stride == 1000

    def test_correct_number_of_windows(self):
        """Test correct number of windows computed."""
        # Test case 1: Exact fit
        data = np.random.randn(20000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        # (20000 - 10000) // 1000 + 1 = 11
        assert len(dataset) == 11

        # Test case 2: Partial last window excluded
        data = np.random.randn(20500, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        # (20500 - 10000) // 1000 + 1 = 11 (same, last 500 not enough for full window)
        assert len(dataset) == 11

        # Test case 3: Minimum case (exactly one window)
        data = np.random.randn(10000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        # (10000 - 10000) // 1000 + 1 = 1
        assert len(dataset) == 1

        # Test case 4: Large stride (non-overlapping windows)
        data = np.random.randn(50000, 15)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=10000)
        # (50000 - 10000) // 10000 + 1 = 5
        assert len(dataset) == 5

    def test_window_slicing_correctness(self):
        """Test that window slicing returns correct data."""
        # Create data with known pattern
        data = np.arange(30000).reshape(-1, 1).astype(np.float32)  # [0, 1, 2, ..., 29999]
        
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        # Check first window: [0:10000]
        window_0 = dataset[0]
        assert isinstance(window_0, torch.Tensor)
        assert window_0.shape == (10000, 1)
        assert window_0[0, 0].item() == 0
        assert window_0[9999, 0].item() == 9999
        
        # Check second window: [1000:11000]
        window_1 = dataset[1]
        assert window_1.shape == (10000, 1)
        assert window_1[0, 0].item() == 1000
        assert window_1[9999, 0].item() == 10999
        
        # Check last window
        last_idx = len(dataset) - 1
        window_last = dataset[last_idx]
        assert window_last.shape == (10000, 1)
        # Last window starts at: (len(dataset) - 1) * stride
        expected_start = last_idx * 1000
        assert window_last[0, 0].item() == expected_start

    def test_window_overlap(self):
        """Test that consecutive windows overlap correctly."""
        data = np.arange(20000).reshape(-1, 1).astype(np.float32)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        window_0 = dataset[0]
        window_1 = dataset[1]
        
        # Window 0: [0:10000], Window 1: [1000:11000]
        # They should overlap from 1000:10000
        # window_0[1000:10000] should equal window_1[0:9000]
        overlap_0 = window_0[1000:10000, 0]
        overlap_1 = window_1[0:9000, 0]
        
        assert torch.allclose(overlap_0, overlap_1)

    def test_tensor_shape_and_dtype(self):
        """Test tensors have correct shape and dtype."""
        data = np.random.randn(50000, 15).astype(np.float64)  # Double precision input
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        window = dataset[0]
        
        # Check shape
        assert window.shape == (10000, 15)
        
        # Check dtype is Float32 (even though input was float64)
        assert window.dtype == torch.float32

    def test_dataloader_compatibility_batching(self):
        """Test compatibility with PyTorch DataLoader for batching."""
        data = np.random.randn(30000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        # Create DataLoader with batching
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Get first batch
        batch = next(iter(loader))
        
        # Check batch shape: (batch_size, sequence_length, features)
        assert batch.shape == (batch_size, 10000, 10)
        assert batch.dtype == torch.float32

    def test_dataloader_compatibility_shuffling(self):
        """Test DataLoader works with shuffling."""
        data = np.random.randn(20000, 5)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Should be able to iterate through
        batches = list(loader)
        assert len(batches) > 0
        
        # Each batch should have correct shape
        for batch in batches:
            assert batch.shape[1:] == (10000, 5)  # sequence_length x features
            assert batch.shape[0] <= 2  # batch_size (may be smaller for last batch)

    def test_dataloader_full_iteration(self):
        """Test iterating through entire dataset with DataLoader."""
        data = np.random.randn(15000, 8)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        total_windows = 0
        for batch in loader:
            total_windows += batch.shape[0]
        
        # Should have iterated through all windows
        assert total_windows == len(dataset)

    def test_data_too_short_raises_error(self):
        """Test error when data is shorter than sequence_length."""
        data = np.random.randn(5000, 10)
        
        with pytest.raises(ValueError, match="data length.*must be >= sequence_length"):
            SequenceWindowDataset(data, sequence_length=10000, stride=1000)

    def test_invalid_sequence_length_raises_error(self):
        """Test error for non-positive sequence_length."""
        data = np.random.randn(20000, 10)
        
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            SequenceWindowDataset(data, sequence_length=0, stride=1000)
        
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            SequenceWindowDataset(data, sequence_length=-100, stride=1000)

    def test_invalid_stride_raises_error(self):
        """Test error for non-positive stride."""
        data = np.random.randn(20000, 10)
        
        with pytest.raises(ValueError, match="stride must be positive"):
            SequenceWindowDataset(data, sequence_length=10000, stride=0)
        
        with pytest.raises(ValueError, match="stride must be positive"):
            SequenceWindowDataset(data, sequence_length=10000, stride=-500)

    def test_invalid_data_type_raises_error(self):
        """Test error for non-ndarray data."""
        data_list = [[1, 2], [3, 4]]
        
        with pytest.raises(TypeError, match="data must be np.ndarray"):
            SequenceWindowDataset(data_list, sequence_length=10000, stride=1000)

    def test_invalid_data_shape_raises_error(self):
        """Test error for non-2D data."""
        # 1D array
        data_1d = np.random.randn(20000)
        with pytest.raises(ValueError, match="data must be 2D array"):
            SequenceWindowDataset(data_1d, sequence_length=10000, stride=1000)
        
        # 3D array
        data_3d = np.random.randn(20000, 10, 5)
        with pytest.raises(ValueError, match="data must be 2D array"):
            SequenceWindowDataset(data_3d, sequence_length=10000, stride=1000)

    def test_index_out_of_range_raises_error(self):
        """Test error when accessing invalid index."""
        data = np.random.randn(20000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        # Valid indices: 0 to len(dataset)-1
        num_windows = len(dataset)
        
        # Test negative index
        with pytest.raises(IndexError, match="Index.*out of range"):
            _ = dataset[-1]
        
        # Test index >= num_windows
        with pytest.raises(IndexError, match="Index.*out of range"):
            _ = dataset[num_windows]
        
        with pytest.raises(IndexError, match="Index.*out of range"):
            _ = dataset[num_windows + 10]

    def test_default_parameters(self):
        """Test dataset uses default parameters correctly."""
        data = np.random.randn(50000, 15)
        dataset = SequenceWindowDataset(data)
        
        # Check defaults
        assert dataset.sequence_length == 10000
        assert dataset.stride == 1000
        
        # Check number of windows with defaults
        # (50000 - 10000) // 1000 + 1 = 41
        assert len(dataset) == 41

    def test_multi_feature_data(self):
        """Test with multi-feature data (realistic use case)."""
        # Simulate feature-engineered data with many features
        num_timesteps = 50000
        num_features = 45  # e.g., 3 assets × 15 features each
        
        data = np.random.randn(num_timesteps, num_features)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        window = dataset[0]
        assert window.shape == (10000, num_features)

    def test_stride_larger_than_sequence_length(self):
        """Test with stride > sequence_length (non-overlapping with gaps)."""
        data = np.random.randn(50000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=15000)
        
        # (50000 - 10000) // 15000 + 1 = 3 windows
        # Window 0: [0:10000]
        # Window 1: [15000:25000]
        # Window 2: [30000:40000]
        assert len(dataset) == 3
        
        window_0 = dataset[0]
        window_1 = dataset[1]
        
        # No overlap, there's a gap
        assert window_0.shape == (10000, 10)
        assert window_1.shape == (10000, 10)

    def test_deterministic_output(self):
        """Test that same index returns same window (deterministic)."""
        data = np.random.randn(20000, 10)
        dataset = SequenceWindowDataset(data, sequence_length=10000, stride=1000)
        
        # Get same window multiple times
        window_1a = dataset[5]
        window_1b = dataset[5]
        window_1c = dataset[5]
        
        # Should all be identical
        assert torch.allclose(window_1a, window_1b)
        assert torch.allclose(window_1b, window_1c)

    def test_integration_with_feature_engineer(self):
        """Integration test: Use with feature-engineered data."""
        # This tests the expected usage from Task 11 -> Task 12
        from data.processors.alignment import TimestampAligner
        from data.processors.features import FeatureEngineer
        from data.source.crypto_historical import generate_synthetic_crypto_data

        # Generate and process data (Tasks 10-11)
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-10',  # More data for windowing
            timeframe='1m'
        )
        
        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})
        
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)
        
        # Drop any NaN rows from feature computation
        features = features.dropna()
        
        # Convert to numpy array for windowing
        data_array = features.values.astype(np.float32)
        
        # Create windowing dataset
        dataset = SequenceWindowDataset(
            data_array,
            sequence_length=1000,  # Smaller for test
            stride=100
        )
        
        # Verify it works
        assert len(dataset) > 0
        
        window = dataset[0]
        assert window.shape[0] == 1000
        assert window.shape[1] == features.shape[1]
        
        # Test with DataLoader
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch.shape == (4, 1000, features.shape[1])
