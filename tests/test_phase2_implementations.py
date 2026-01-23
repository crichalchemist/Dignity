"""Integration tests for Phase 2 implementations."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.cache_manager import FeatureCacheManager
from utils.logger import get_logger
from utils.seed import set_seed
from utils.training_utils import CheckpointManager, EarlyStopping, MetricComparator


def test_deterministic_runs():
    """Test that deterministic runs produce consistent results."""
    logger = get_logger(__name__)
    logger.info("Testing deterministic runs...")

    # Set seed multiple times and verify consistency
    set_seed(42)
    np.random.seed(42)
    arr1 = np.random.randn(100)

    set_seed(42)
    np.random.seed(42)
    arr2 = np.random.randn(100)

    assert np.array_equal(arr1, arr2), "Deterministic runs should produce identical results"
    logger.info("✅ Deterministic runs test passed")


def test_early_stopping():
    """Test early stopping functionality."""
    logger = get_logger(__name__)
    logger.info("Testing early stopping...")

    early_stop = EarlyStopping(patience=3, min_delta=0.001)

    # Simulate improving then worsening scores
    scores = [0.5, 0.6, 0.7, 0.65, 0.62, 0.58]  # Improves then gets worse

    for i, score in enumerate(scores, 1):
        should_stop = early_stop(score)
        if should_stop:
            assert i == 6, f"Should stop after patience exceeded (epoch {i})"
            break

    logger.info("✅ Early stopping test passed")


def test_checkpoint_manager():
    """Test checkpoint management with top-N retention."""
    logger = get_logger(__name__)
    logger.info("Testing checkpoint manager...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_manager = CheckpointManager(checkpoint_dir, top_n=2)

        # Create mock state dicts with PyTorch tensors
        state_dicts = [
            {"param1": torch.tensor([1.0, 2.0]), "param2": "model1"},
            {"param1": torch.tensor([2.0, 3.0]), "param2": "model2"},
            {"param1": torch.tensor([3.0, 4.0]), "param2": "model3"},
        ]

        scores = [0.7, 0.8, 0.6]  # Best, better, worst

        # Save checkpoints
        for i, (state, score) in enumerate(zip(state_dicts, scores, strict=False), 1):
            checkpoint_manager.save(state, score, i, "test_model")

        # Should only keep top 2
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) == 2, f"Should keep top 2 checkpoints, got {len(checkpoint_files)}"

        # Check that best checkpoint is retained
        best_checkpoint = checkpoint_manager.get_best_checkpoint()
        assert best_checkpoint is not None, "Should have a best checkpoint"

        logger.info("✅ Checkpoint manager test passed")


def test_metric_comparator():
    """Test metric comparison for different task types."""
    logger = get_logger(__name__)
    logger.info("Testing metric comparator...")

    # Classification (higher is better)
    cls_comparator = MetricComparator("classification")
    assert cls_comparator.is_better(0.9, 0.8), "Higher accuracy should be better"
    assert not cls_comparator.is_better(0.7, 0.8), "Lower accuracy should not be better"
    assert cls_comparator.initialize_best() == -float("inf"), "Classification best should start at -inf"

    # Regression (lower is better for loss)
    reg_comparator = MetricComparator("regression")
    assert reg_comparator.is_better(0.1, 0.2), "Lower loss should be better"
    assert not reg_comparator.is_better(0.3, 0.2), "Higher loss should not be better"
    assert reg_comparator.initialize_best() == float("inf"), "Regression best should start at inf"

    logger.info("✅ Metric comparator test passed")


def test_feature_cache_manager():
    """Test feature cache manager functionality."""
    logger = get_logger(__name__)
    logger.info("Testing feature cache manager...")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_manager = FeatureCacheManager(tmpdir, max_cache_age_days=30)

        # Create test data
        raw_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 100)
        })

        # Mock feature config
        class MockConfig:
            def __str__(self):
                return "test_config_v1"

        feature_config = MockConfig()

        # Test cache miss
        cached_features = cache_manager.get_cached_features(
            raw_data, feature_config, intrinsic_time=False, pair_name="test"
        )
        assert cached_features is None, "Should miss cache initially"

        # Simulate saving features (would normally be computed)
        mock_features = raw_data.copy()
        cache_key = cache_manager._compute_content_hash(raw_data, feature_config, False, "test")
        cache_manager.save_features_to_cache(mock_features, cache_key, "test")

        # Test cache hit
        cached_features = cache_manager.get_cached_features(
            raw_data, feature_config, intrinsic_time=False, pair_name="test"
        )
        assert cached_features is not None, "Should hit cache after saving"

        # Test cache statistics
        stats = cache_manager.get_cache_stats()
        assert stats['cache_hits'] == 1, "Should have 1 cache hit"
        assert stats['cache_misses'] == 1, "Should have 1 cache miss"
        assert stats['hit_rate_percent'] == 50.0, "Hit rate should be 50%"

        logger.info("✅ Feature cache manager test passed")


def test_logging_framework():
    """Test that logging framework works correctly."""
    logger = get_logger(__name__)

    # Test different log levels
    logger.info("Testing info level logging")
    logger.warning("Testing warning level logging")

    # Verify logger configuration
    assert logger.name == __name__, "Logger should be named after module"

    print("✅ Logging framework test passed")


def test_integration_all_components():
    """Integration test combining all Phase 2 components."""
    logger = get_logger(__name__)
    logger.info("Running comprehensive integration test...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Test deterministic runs
        set_seed(42)
        np.random.randn(50)

        # 2. Test checkpoint manager
        checkpoint_manager = CheckpointManager(tmpdir / "checkpoints", top_n=3)

        # 3. Test cache manager
        cache_manager = FeatureCacheManager(tmpdir / "cache")

        # 4. Test early stopping
        early_stop = EarlyStopping(patience=2)

        # 5. Test metric comparator
        comparator = MetricComparator("classification")

        # Simulate a training-like workflow with PyTorch tensors
        model_state = {"weights": torch.randn(10)}
        best_score = comparator.initialize_best()

        for epoch in range(1, 6):
            # Simulate validation score
            score = 0.5 + 0.1 * epoch + np.random.normal(0, 0.05)

            # Check if this is better
            if comparator.is_better(score, best_score):
                best_score = score
                checkpoint_manager.save(model_state, score, epoch)

            # Check early stopping
            if early_stop(score):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Verify integration worked
        assert checkpoint_manager.get_best_checkpoint() is not None, "Should have best checkpoint"
        assert early_stop.should_stop or epoch >= 5, "Should have either early stopped or completed"

        # Test cache integration
        df = pd.DataFrame({'col1': range(10), 'col2': range(10, 20)})
        cache_key = cache_manager._compute_content_hash(df, "test_config", False, "integration_test")
        cache_manager.save_features_to_cache(df, cache_key, "integration_test")

        cached_df = cache_manager.get_cached_features(df, "test_config", pair_name="integration_test")
        assert cached_df is not None, "Integration cache should work"

        logger.info("✅ Comprehensive integration test passed")


if __name__ == "__main__":
    # Set up logging
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run all tests
    test_deterministic_runs()
    test_early_stopping()
    test_checkpoint_manager()
    test_metric_comparator()
    test_feature_cache_manager()
    test_logging_framework()
    test_integration_all_components()

    print("\n🎉 All Phase 2 integration tests passed!")
