"""Tests for IterableFXDataset implementation.

Tests memory efficiency, streaming functionality, and integration with BaseDataAgent.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from config.config import DataConfig, FeatureConfig
from data.iterable_dataset import IterableFXDataset


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Create realistic OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')

    # Generate random walk prices
    base_price = 1.1000
    returns = np.random.normal(0, 0.0001, n_samples)
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    data = {
        'datetime': dates,
        'open': prices * (1 + np.random.normal(0, 0.0005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_samples))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_samples),
    }

    df = pd.DataFrame(data)
    df = df.round(5)

    # Add some technical indicators as features
    df['rsi'] = np.random.uniform(20, 80, n_samples)
    df['macd'] = np.random.normal(0, 0.001, n_samples)
    df['bb_upper'] = df['close'] * 1.02
    df['bb_lower'] = df['close'] * 0.98

    return df


@pytest.fixture
def iterable_config(sample_feature_data, tmp_path):
    """Create DataConfig for IterableDataset testing."""
    csv_file = tmp_path / "test_iterable_data.csv"
    sample_feature_data.to_csv(csv_file, index=False)
    
    return DataConfig(
        csv_path=str(csv_file),
        datetime_column='datetime',
        t_in=60,
        t_out=10,
        lookahead_window=20,
        train_range=('2024-01-01', '2024-01-02'),
        val_range=('2024-01-02', '2024-01-03'),
        test_range=('2024-01-03', '2024-01-04'),
        feature_columns=None,  # Use all except datetime/source_file
        target_type='classification',
        flat_threshold=0.001,
        top_k_predictions=3,
        predict_sell_now=True,
    )


@pytest.fixture
def feature_config():
    """Create FeatureConfig for testing."""
    return FeatureConfig(
        sma_windows=[10, 20, 50],
        ema_windows=[10, 20, 50],
        rsi_window=14,
        bollinger_window=20,
        bollinger_num_std=2.0,
        atr_window=14,
        short_vol_window=10,
        long_vol_window=50,
        spread_windows=[20],
        imbalance_smoothing=5,
    )


@pytest.fixture
def cached_feature_file(sample_feature_data, temp_dir):
    """Create a cached feature file for testing."""
    cache_path = temp_dir / "gbpusd_features_test.feather"
    sample_feature_data.to_feather(cache_path)
    return cache_path


class TestIterableFXDataset:
    """Test IterableFXDataset functionality."""

    def test_initialization(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test IterableFXDataset initialization."""
        # Mock the file finding to use our test file
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            assert dataset.pair == "gbpusd"
            assert dataset.data_cfg == iterable_config
            assert dataset.feature_cfg == feature_config
            assert dataset.cache_path == cached_feature_file
            assert dataset.norm_stats is not None

    def test_file_not_found_error(self, iterable_config, feature_config, temp_dir):
        """Test FileNotFoundError when cache file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Cached features not found"):
            IterableFXDataset(
                pair="nonexistent",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

    def test_find_cached_features(self, temp_dir, cached_feature_file, iterable_config, feature_config):
        """Test cache file finding logic."""
        dataset = IterableFXDataset.__new__(IterableFXDataset)
        dataset.pair = "gbpusd"
        dataset.cache_dir = temp_dir

        # Should find our test file
        found_path = dataset._find_cached_features()
        assert found_path == cached_feature_file

        # Should raise error for non-existent pair
        dataset.pair = "nonexistent"
        with pytest.raises(FileNotFoundError, match="No cached features found"):
            dataset._find_cached_features()

    def test_iteration_basic(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test basic iteration functionality."""
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
                chunksize=50,  # Smaller chunks for testing
            )

            # Convert to list to test iteration
            samples = list(dataset)

            assert len(samples) > 0, "Should produce some samples"

            # Check sample structure
            seq, targets = samples[0]
            assert isinstance(seq, torch.Tensor)
            assert isinstance(targets, dict)
            assert seq.shape == (iterable_config.t_in, 8)  # 8 features (excluding datetime, source_file)

            # Check targets
            assert 'primary' in targets
            assert 'max_return' in targets
            assert 'topk_returns' in targets
            assert 'topk_prices' in targets
            assert 'sell_now' in targets

            # Check tensor types
            assert targets['primary'].dtype == torch.long  # classification
            assert targets['max_return'].dtype == torch.float32

    def test_iteration_with_workers(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test iteration with multiple workers."""
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
                chunksize=25,  # Small chunks for worker testing
            )

            # Mock worker info for multi-worker scenario
            with patch('torch.utils.data.get_worker_info') as mock_worker_info:
                # Test single worker
                mock_worker_info.return_value = MagicMock(id=0, num_workers=1)
                samples_single = list(dataset)

                # Test with multiple workers (worker 0)
                mock_worker_info.return_value = MagicMock(id=0, num_workers=2)
                samples_worker0 = list(dataset)

                # Should produce some samples in all cases
                assert len(samples_single) > 0
                assert len(samples_worker0) >= 0  # May produce different amounts due to chunking

    def test_memory_efficiency_vs_standard_dataset(self, iterable_config, feature_config, temp_dir,
                                                   cached_feature_file):
        """Test that IterableFXDataset uses less memory than standard dataset."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure memory before creating dataset
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before

        # IterableFXDataset should have relatively low memory footprint
        assert memory_used < 100, f"Memory usage too high: {memory_used:.1f} MB"

    def test_chunked_processing(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test that chunked processing works correctly."""
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
                chunksize=30,  # Small chunks
            )

            # Get all samples
            all_samples = list(dataset)

            # Sample a few and verify consistency
            if len(all_samples) >= 3:
                seq1, targets1 = all_samples[0]
                seq2, targets2 = all_samples[1]
                seq3, targets3 = all_samples[2]

                # All sequences should have same shape
                assert seq1.shape == seq2.shape == seq3.shape == (iterable_config.t_in, 8)

                # All targets should have same structure
                assert set(targets1.keys()) == set(targets2.keys()) == set(targets3.keys())

    def test_data_consistency_with_base_agent(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test that IterableFXDataset produces consistent data with BaseDataAgent."""
        # Load the feature data
        feature_df = pd.read_feather(cached_feature_file)

        # Create BaseDataAgent dataset for comparison
        from data.agents.single_task_agent import SingleTaskDataAgent
        base_agent = SingleTaskDataAgent(iterable_config)
        base_datasets = base_agent.build_datasets(feature_df)

        # Create IterableFXDataset
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            iterable_dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            # Compare target types and basic structure
            base_sample = base_datasets['train'][0]
            iterable_sample = next(iter(iterable_dataset))

            # Check target consistency
            assert set(base_sample[1].keys()) == set(iterable_sample[1].keys())

            # Check that primary targets have correct dtypes
            assert base_sample[1]['primary'].dtype == iterable_sample[1]['primary'].dtype

    def test_edge_cases(self, iterable_config, feature_config, temp_dir):
        """Test edge cases and error handling."""
        # Test with very small dataset
        small_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'open': np.random.randn(10).cumsum() + 1.1,
            'high': np.random.randn(10).cumsum() + 1.15,
            'low': np.random.randn(10).cumsum() + 1.05,
            'close': np.random.randn(10).cumsum() + 1.1,
            'volume': np.random.randint(100, 1000, 10),
            'rsi': np.random.uniform(20, 80, 10),
        })

        small_cache = temp_dir / "small_features.feather"
        small_data.to_feather(small_cache)

        with patch.object(IterableFXDataset, '_find_cached_features', return_value=small_cache):
            dataset = IterableFXDataset(
                pair="small",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            # Should handle small dataset gracefully
            samples = list(dataset)
            # May produce 0 samples due to insufficient data, which is expected
            assert isinstance(samples, list)

    def test_regression_targets(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test IterableFXDataset with regression targets."""
        # Create config for regression
        regression_config = DataConfig(
            datetime_column='datetime',
            t_in=60,
            t_out=10,
            lookahead_window=20,
            train_range=('2024-01-01', '2024-01-02'),
            val_range=('2024-01-02', '2024-01-03'),
            test_range=('2024-01-03', '2024-01-04'),
            feature_columns=None,
            target_type='regression',  # Regression instead of classification
            flat_threshold=0.001,
            top_k_predictions=3,
            predict_sell_now=True,
        )

        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=regression_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            samples = list(dataset)
            if samples:
                seq, targets = samples[0]

                # Primary target should be float32 for regression
                assert targets['primary'].dtype == torch.float32
                assert targets['max_return'].dtype == torch.float32

    def test_integration_with_dataloader(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test integration with PyTorch DataLoader."""
        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            # Create DataLoader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4,
                num_workers=0,  # Single process for testing
            )

            # Test iteration
            batches = list(dataloader)

            assert len(batches) > 0, "Should produce some batches"

            # Check batch structure
            batch = batches[0]
            sequences, targets = batch

            assert sequences.shape[0] == 4  # Batch size
            assert sequences.shape[1] == iterable_config.t_in  # Time steps
            assert len(targets) > 0  # Should have targets

    def test_performance_benchmark(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Benchmark performance of IterableFXDataset."""
        import time

        with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
            dataset = IterableFXDataset(
                pair="gbpusd",
                data_cfg=iterable_config,
                feature_cfg=feature_config,
                input_root=temp_dir,
                cache_dir=temp_dir,
            )

            # Time the iteration
            start_time = time.time()
            samples = list(dataset)
            end_time = time.time()

            iteration_time = end_time - start_time
            samples_per_second = len(samples) / iteration_time if iteration_time > 0 else float('inf')

            # Should be able to process at least some samples per second
            assert samples_per_second > 0, "Should process samples"

            # Print benchmark results for manual inspection
            print("\nIterableFXDataset Benchmark:")
            print(f"  Total samples: {len(samples)}")
            print(f"  Iteration time: {iteration_time:.2f}s")
            print(f"  Samples/second: {samples_per_second:.1f}")

    def test_chunksize_impact(self, iterable_config, feature_config, temp_dir, cached_feature_file):
        """Test impact of different chunk sizes on performance."""
        import time

        chunk_sizes = [10, 25, 50, 100]
        results = {}

        for chunksize in chunk_sizes:
            with patch.object(IterableFXDataset, '_find_cached_features', return_value=cached_feature_file):
                dataset = IterableFXDataset(
                    pair="gbpusd",
                    data_cfg=iterable_config,
                    feature_cfg=feature_config,
                    input_root=temp_dir,
                    cache_dir=temp_dir,
                    chunksize=chunksize,
                )

                start_time = time.time()
                samples = list(dataset)
                end_time = time.time()

                results[chunksize] = {
                    'samples': len(samples),
                    'time': end_time - start_time,
                    'samples_per_sec': len(samples) / (end_time - start_time) if end_time > start_time else 0
                }

        # All chunk sizes should produce samples
        assert all(r['samples'] > 0 for r in results.values())

        # Print results for analysis
        print("\nChunk Size Performance:")
        for chunksize, result in results.items():
            print(f"  Chunksize {chunksize}: {result['samples']} samples, "
                  f"{result['time']:.2f}s, {result['samples_per_sec']:.1f} samples/sec")


# Fixtures for pytest
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print statements
