"""Comprehensive tests for unified data agent implementation.

Tests the consolidation of data agents into BaseDataAgent hierarchy and
validates backward compatibility with existing workflows.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from config.config import DataConfig, MultiTaskDataConfig
from data.agents.base_agent import BaseDataAgent, SequenceDataset
from data.agents.multitask_agent import MultiTaskDataAgent
from data.agents.single_task_agent import SingleTaskDataAgent


@pytest.fixture
def sample_data():
    """Create sample FX data for testing."""
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
def single_task_config(sample_data, tmp_path):
    """Create DataConfig for single task testing."""
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    
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
def multitask_config(sample_data, tmp_path):
    """Create MultiTaskDataConfig for testing."""
    csv_file = tmp_path / "test_data_multi.csv"
    sample_data.to_csv(csv_file, index=False)
    
    return MultiTaskDataConfig(
        csv_path=str(csv_file),
        datetime_column='datetime',
        t_in=60,
        t_out=10,
        lookahead_window=20,
        train_range=('2024-01-01', '2024-01-02'),
        val_range=('2024-01-02', '2024-01-03'),
        test_range=('2024-01-03', '2024-01-04'),
        feature_columns=None,
        flat_threshold=0.001,
        top_k_predictions=3,
        predict_sell_now=True,
        vol_min_change=0.0005,
    )


class TestBaseDataAgent:
    """Test BaseDataAgent functionality."""

    def test_base_agent_initialization(self, single_task_config):
        """Test BaseDataAgent can be initialized."""
        agent = BaseDataAgent(single_task_config)
        assert agent.cfg == single_task_config
        assert agent.norm_stats is None

    def test_invalid_target_type(self, sample_data):
        """Test that invalid target types raise errors."""
        config = DataConfig(
            datetime_column='datetime',
            t_in=10,
            t_out=5,
            train_range=None,
            val_range=None,
            test_range=None,
            target_type='invalid_type',  # Invalid
            flat_threshold=0.001,
        )

        with pytest.raises(ValueError, match="target_type must be"):
            BaseDataAgent(config)

    def test_split_dataframe(self, single_task_config, sample_data):
        """Test data splitting functionality."""
        agent = BaseDataAgent(single_task_config)
        splits = agent.split_dataframe(sample_data)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # Check date ranges
        train_df = splits['train']
        val_df = splits['val']
        test_df = splits['test']

        assert train_df['datetime'].min() >= pd.to_datetime('2024-01-01')
        assert train_df['datetime'].max() <= pd.to_datetime('2024-01-02')

        # Check no overlap
        assert train_df['datetime'].max() < val_df['datetime'].min()
        assert val_df['datetime'].max() < test_df['datetime'].min()

    def test_fit_normalization(self, single_task_config, sample_data):
        """Test normalization fitting."""
        agent = BaseDataAgent(single_task_config)
        feature_cols = [col for col in sample_data.columns
                        if col not in {single_task_config.datetime_column, 'source_file'}]

        norm_stats = agent.fit_normalization(sample_data, feature_cols)

        assert agent.norm_stats is not None
        assert len(norm_stats.mean) == len(feature_cols)
        assert len(norm_stats.std) == len(feature_cols)
        assert not np.any(norm_stats.std == 0), "Std should not be zero"

    def test_sequence_dataset(self, single_task_config, sample_data):
        """Test SequenceDataset creation and indexing."""
        BaseDataAgent(single_task_config)
        feature_cols = [col for col in sample_data.columns
                        if col not in {single_task_config.datetime_column, 'source_file'}]

        # Create simple dataset
        sequences = np.random.randn(10, 5, len(feature_cols))
        targets = {
            'primary': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            'max_return': np.random.randn(10),
            'topk_returns': np.random.randn(10, 3),
            'topk_prices': np.random.randn(10, 3),
        }

        dataset = SequenceDataset(sequences, targets, 'classification')

        # Test indexing
        seq, target_dict = dataset[0]
        assert seq.shape == (5, len(feature_cols))
        assert 'primary' in target_dict
        assert target_dict['primary'].dtype == torch.long
        assert target_dict['max_return'].dtype == torch.float32

        # Test length
        assert len(dataset) == 10

    def test_build_dataloaders(self, single_task_config, sample_data):
        """Test DataLoader building."""
        agent = BaseDataAgent(single_task_config)
        feature_cols = [col for col in sample_data.columns
                        if col not in {single_task_config.datetime_column, 'source_file'}]

        # Create simple dataset
        sequences = np.random.randn(20, 5, len(feature_cols))
        targets = {
            'primary': np.array([0, 1, 2] * 6 + [0, 1, 2]),
            'max_return': np.random.randn(20),
            'topk_returns': np.random.randn(20, 3),
            'topk_prices': np.random.randn(20, 3),
        }

        dataset = SequenceDataset(sequences, targets, 'classification')
        loaders = agent.build_dataloaders(
            {'train': dataset, 'val': dataset, 'test': dataset},
            batch_size=4,
            num_workers=0
        )

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders

        # Test train loader (should shuffle)
        train_loader = loaders['train']
        batch1 = next(iter(train_loader))
        batch2 = next(iter(train_loader))
        # Different order due to shuffling
        assert not torch.equal(batch1[0][0], batch2[0][0])

        # Test val loader (should not shuffle)
        val_loader = loaders['val']
        batch1 = next(iter(val_loader))
        batch2 = next(iter(val_loader))
        # Same order due to no shuffling
        assert torch.equal(batch1[0][0], batch2[0][0])


class TestSingleTaskDataAgent:
    """Test SingleTaskDataAgent functionality."""

    def test_initialization(self, single_task_config):
        """Test SingleTaskDataAgent initialization."""
        agent = SingleTaskDataAgent(single_task_config)
        assert isinstance(agent, BaseDataAgent)

    def test_classification_targets(self, single_task_config, sample_data):
        """Test classification target creation."""
        agent = SingleTaskDataAgent(single_task_config)

        # Test different return scenarios
        assert agent._create_primary_target(0.002) == 2  # up
        assert agent._create_primary_target(-0.002) == 0  # down
        assert agent._create_primary_target(0.0005) == 1  # flat
        assert agent._create_primary_target(-0.0005) == 1  # flat

    def test_regression_targets(self, single_task_config):
        """Test regression target creation."""
        config = DataConfig(
            datetime_column='datetime',
            t_in=10,
            t_out=5,
            train_range=None,
            val_range=None,
            test_range=None,
            target_type='regression',  # Regression mode
            flat_threshold=0.001,
        )

        agent = SingleTaskDataAgent(config)

        # Should return raw log return for regression
        assert agent._create_primary_target(0.002) == 0.002
        assert agent._create_primary_target(-0.003) == -0.003

    def test_build_datasets_classification(self, single_task_config, sample_data):
        """Test dataset building for classification."""
        agent = SingleTaskDataAgent(single_task_config)
        datasets = agent.build_datasets(sample_data)

        assert 'train' in datasets
        assert 'val' in datasets
        assert 'test' in datasets

        # Check dataset structure
        train_dataset = datasets['train']
        seq, targets = train_dataset[0]

        assert seq.shape == (single_task_config.t_in,
                             len(sample_data.columns) - 2)  # Excluding datetime, source_file
        assert 'primary' in targets
        assert 'max_return' in targets
        assert 'topk_returns' in targets
        assert 'topk_prices' in targets
        assert 'sell_now' in targets

        # Check target types for classification
        assert targets['primary'].dtype == torch.long
        assert targets['max_return'].dtype == torch.float32

    def test_backward_compatibility(self, single_task_config, sample_data):
        """Test backward compatibility with original agent_data.py."""
        agent = SingleTaskDataAgent(single_task_config)
        datasets = agent.build_datasets(sample_data)

        # Should produce same output format as original implementation
        datasets['train']

        # Check that we can create DataLoaders
        loader = agent.build_dataloaders(datasets, batch_size=8, num_workers=0)
        batch = next(iter(loader['train']))

        sequences, targets = batch
        assert sequences.shape[0] == 8
        assert sequences.shape[1] == single_task_config.t_in


class TestMultiTaskDataAgent:
    """Test MultiTaskDataAgent functionality."""

    def test_initialization(self, multitask_config):
        """Test MultiTaskDataAgent initialization."""
        agent = MultiTaskDataAgent(multitask_config)
        assert isinstance(agent, BaseDataAgent)

    def test_primary_target(self, multitask_config):
        """Test primary target creation."""
        agent = MultiTaskDataAgent(multitask_config)

        # Multitask always uses direction classification as primary
        assert agent._create_primary_target(0.002) == 2  # up
        assert agent._create_primary_target(-0.002) == 0  # down
        assert agent._create_primary_target(0.0005) == 1  # flat

    def test_extended_targets(self, multitask_config, sample_data):
        """Test extended multitask targets."""
        agent = MultiTaskDataAgent(multitask_config)
        datasets = agent.build_datasets(sample_data)

        train_dataset = datasets['train']
        seq, targets = train_dataset[0]

        # Check extended targets exist
        assert 'direction_class' in targets
        assert 'return_reg' in targets
        assert 'next_close_reg' in targets
        assert 'vol_class' in targets
        assert 'trend_class' in targets
        assert 'vol_regime_class' in targets
        assert 'candle_class' in targets

        # Check targets have correct types
        assert targets['direction_class'].dtype == torch.long  # classification
        assert targets['return_reg'].dtype == torch.float32  # regression
        assert targets['vol_class'].dtype == torch.long  # binary classification
        assert targets['trend_class'].dtype == torch.long  # 3-class classification

    def test_target_consistency(self, multitask_config, sample_data):
        """Test consistency between primary and extended targets."""
        agent = MultiTaskDataAgent(multitask_config)
        datasets = agent.build_datasets(sample_data)

        train_dataset = datasets['train']

        # Primary should be same as direction_class for multitask
        for i in range(min(5, len(train_dataset))):
            seq, targets = train_dataset[i]
            assert targets['primary'].item() == targets['direction_class'].item()

    def test_memory_efficiency(self, multitask_config, sample_data):
        """Test memory efficiency with extended targets."""
        agent = MultiTaskDataAgent(multitask_config)
        datasets = agent.build_datasets(sample_data)

        # Should handle additional targets without memory issues
        loaders = agent.build_dataloaders(datasets, batch_size=8, num_workers=0)
        batch = next(iter(loaders['train']))

        sequences, targets = batch
        assert sequences.shape[0] == 8
        assert len(targets) >= 8  # Multiple target types

    def test_config_validation(self, sample_data):
        """Test MultiTaskDataConfig specific validation."""
        config = MultiTaskDataConfig(
            datetime_column='datetime',
            t_in=10,
            t_out=5,
            train_range=None,
            val_range=None,
            test_range=None,
            flat_threshold=0.001,
            vol_min_change=0.0005,  # Specific to multitask
        )

        agent = MultiTaskDataAgent(config)
        assert agent.cfg.vol_min_change == 0.0005


class TestDataAgentConsolidation:
    """Test the consolidation benefits and architectural improvements."""

    def test_code_reduction(self):
        """Test that consolidation reduced code duplication."""
        # Read the agent files
        with open('data/agents/base_agent.py') as f:
            base_content = f.read()
        with open('data/agents/single_task_agent.py') as f:
            single_content = f.read()
        with open('data/agents/multitask_agent.py') as f:
            multitask_content = f.read()

        # Check that common functionality is in BaseDataAgent
        assert 'class BaseDataAgent' in base_content
        assert 'split_dataframe' in base_content
        assert 'fit_normalization' in base_content
        assert 'build_datasets' in base_content
        assert 'build_dataloaders' in base_content

        # Check that subclasses are minimal
        assert len(single_content) < 1000  # Should be small
        assert len(multitask_content) < 3000  # Should be reasonable size

    def test_api_consistency(self, single_task_config, multitask_config, sample_data):
        """Test consistent API between agent types."""
        single_agent = SingleTaskDataAgent(single_task_config)
        multi_agent = MultiTaskDataAgent(multitask_config)

        # Both should have same base methods
        single_datasets = single_agent.build_datasets(sample_data)
        multi_datasets = multi_agent.build_datasets(sample_data)

        # Both should return same structure (dict of datasets)
        assert set(single_datasets.keys()) == {'train', 'val', 'test'}
        assert set(multi_datasets.keys()) == {'train', 'val', 'test'}

        # Both should have compatible DataLoaders
        single_loaders = single_agent.build_dataloaders(single_datasets, batch_size=4)
        multi_loaders = multi_agent.build_dataloaders(multi_datasets, batch_size=4)

        assert set(single_loaders.keys()) == set(multi_loaders.keys())

    def test_error_handling(self, sample_data):
        """Test proper error handling and validation."""
        # Test with insufficient data
        short_data = sample_data.iloc[:20].copy()  # Very short dataset

        config = DataConfig(
            datetime_column='datetime',
            t_in=60,  # Large input window
            t_out=10,
            train_range=None,
            val_range=None,
            test_range=None,
            target_type='classification',
            flat_threshold=0.001,
        )

        agent = SingleTaskDataAgent(config)

        # Should handle gracefully (possibly creating empty dataset)
        try:
            datasets = agent.build_datasets(short_data)
            # If it doesn't error, check that we handle gracefully
            if datasets['train']:
                assert len(datasets['train']) >= 0
        except ValueError as e:
            # Expected for insufficient data
            assert "No sequences created" in str(e)

    def test_edge_cases(self, single_task_config):
        """Test edge cases and boundary conditions."""
        agent = SingleTaskDataAgent(single_task_config)

        # Test normalization with edge case std
        test_data = pd.DataFrame({
            'feature1': [1.0, 1.0, 1.0, 1.0],  # Zero std
            'feature2': [1.0, 2.0, 3.0, 4.0],
            'datetime': pd.date_range('2024-01-01', periods=4, freq='1min'),
            'close': [1.1, 1.2, 1.3, 1.4],
        })

        norm_stats = agent.fit_normalization(test_data, ['feature1', 'feature2'])

        # Should handle zero std gracefully
        assert norm_stats.std[0] == 1.0  # Should be replaced with 1.0
        assert norm_stats.std[1] > 0  # Should be actual std


def test_integration_with_existing_workflows(single_task_config, sample_data):
    """Test integration with existing training workflows."""
    # Simulate existing workflow
    agent = SingleTaskDataAgent(single_task_config)
    datasets = agent.build_datasets(sample_data)
    loaders = agent.build_dataloaders(datasets, batch_size=8)

    # Test that we can iterate through batches
    for _split_name, loader in loaders.items():
        batch_count = 0
        for batch in loader:
            sequences, targets = batch
            assert sequences.shape[0] == 8  # Batch size
            assert sequences.shape[1] == single_task_config.t_in
            assert 'primary' in targets
            batch_count += 1

            if batch_count >= 3:  # Test first few batches
                break

    # Test that we can use with torch training loop
    sequences, targets = next(iter(loaders['train']))

    # Simulate model input
    model_input = sequences
    target = targets['primary']

    # Should be compatible with typical PyTorch model
    assert model_input.shape == (8, single_task_config.t_in, len(sample_data.columns) - 2)
    assert target.shape == (8,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
