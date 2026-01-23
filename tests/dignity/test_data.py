"""Test data pipeline."""

import pytest
import numpy as np
import pandas as pd
from dignity.data.source.synthetic import SyntheticGenerator
from dignity.data.pipeline import TransactionPipeline
from dignity.data.loader import TransactionDataset, create_dataloader


class TestSyntheticGenerator:
    """Test synthetic data generation."""
    
    def test_normal_sequence(self):
        """Test normal sequence generation."""
        gen = SyntheticGenerator(seed=42)
        data = gen.generate_normal_sequence(length=1000)
        
        assert 'volume' in data
        assert 'price' in data
        assert 'fee_rate' in data
        assert 'tx_count' in data
        
        assert len(data['volume']) == 1000
        assert np.all(data['volume'] > 0)
    
    def test_anomalous_sequence(self):
        """Test anomalous pattern generation."""
        gen = SyntheticGenerator(seed=42)
        
        # Test different anomaly types
        for anomaly_type in ['volume_spike', 'price_manipulation', 'fee_evasion']:
            data = gen.generate_anomalous_sequence(
                length=100,
                anomaly_type=anomaly_type
            )
            
            assert len(data['volume']) == 100
            assert 'volume' in data and 'price' in data
    
    def test_dataset_generation(self):
        """Test balanced dataset generation."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(
            num_normal=100,
            num_anomalous=50,
            seq_len=50
        )
        
        assert len(df) == (100 + 50) * 50  # sequences * seq_len
        assert 'label' in df.columns
        assert set(df['label'].unique()) == {0, 1}


class TestTransactionPipeline:
    """Test data preprocessing pipeline."""
    
    def test_signal_computation(self):
        """Test signal feature computation."""
        gen = SyntheticGenerator(seed=42)
        df_raw = gen.generate_dataset(num_normal=10, num_anomalous=0, seq_len=100)
        
        pipeline = TransactionPipeline(seq_len=50)
        df_processed = pipeline.compute_signals(df_raw)
        
        assert 'volume_volatility' in df_processed.columns
        assert 'volatility' in df_processed.columns
        assert 'momentum' in df_processed.columns
    
    def test_fit_transform(self):
        """Test scaling pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=50, num_anomalous=10, seq_len=100)
        df = df.drop('label', axis=1)
        
        pipeline = TransactionPipeline(seq_len=50)
        X = pipeline.fit_transform(df)
        
        assert X.shape[0] == len(df)
        assert X.shape[1] > 0  # Has features
        
        # Check scaling (should be roughly centered)
        assert np.abs(X.mean()) < 1.0
    
    def test_sequence_creation(self):
        """Test sliding window sequence creation."""
        X = np.random.randn(500, 9)
        y = np.random.randint(0, 2, 500)
        
        pipeline = TransactionPipeline(seq_len=100)
        pipeline.fitted = True
        pipeline.available_features = [f'f{i}' for i in range(9)]
        
        X_seq, y_seq = pipeline.create_sequences(X, y, stride=1)
        
        assert X_seq.shape[0] == 401  # 500 - 100 + 1
        assert X_seq.shape[1] == 100  # seq_len
        assert X_seq.shape[2] == 9  # features
        assert len(y_seq) == 401
    
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=100, num_anomalous=20, seq_len=200)
        labels = df['label'].values
        df = df.drop('label', axis=1)
        
        pipeline = TransactionPipeline(seq_len=100)
        X_seq, y_seq = pipeline.process(df, labels, fit=True, stride=10)
        
        assert X_seq.ndim == 3  # [sequences, seq_len, features]
        assert y_seq.ndim == 1  # [sequences]
        assert len(X_seq) == len(y_seq)


class TestDataLoader:
    """Test PyTorch data loading."""
    
    def test_dataset_creation(self):
        """Test TransactionDataset."""
        X = np.random.randn(100, 50, 9)
        y = np.random.randint(0, 2, 100)
        
        dataset = TransactionDataset(X, y)
        
        assert len(dataset) == 100
        
        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (50, 9)
        assert sample_y.shape == ()
    
    def test_dataloader_creation(self):
        """Test DataLoader creation."""
        X = np.random.randn(100, 50, 9)
        y = np.random.randint(0, 2, 100)
        
        loader = create_dataloader(
            X, y,
            batch_size=16,
            shuffle=True,
            device='cpu'
        )
        
        assert len(loader) == 100 // 16 + 1  # batches
        
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 16  # batch size
        assert batch_x.shape[1:] == (50, 9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
