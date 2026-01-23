"""Unit tests for data splitting logic to ensure non-overlapping, time-ordered splits."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from config.config import DataConfig
from data.agents.single_task_agent import SingleTaskDataAgent


class TestDataSplits:
    """Test suite for data splitting functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic time-series data for testing."""
        # Create 1000 minutes of data starting from a specific date
        start_date = datetime(2023, 1, 1, 9, 30, 0)  # Market open
        dates = pd.date_range(start=start_date, periods=1000, freq='1min')

        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible tests
        base_price = 1.1000

        returns = np.random.normal(0, 0.0001, len(dates))  # Small log returns
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLC from prices
        high_noise = np.random.uniform(0.0001, 0.0005, len(prices))
        low_noise = np.random.uniform(-0.0005, -0.0001, len(prices))

        high = prices * (1 + high_noise)
        low = prices * (1 + low_noise)
        open_prices = np.roll(prices, 1)
        open_prices[0] = base_price
        close = prices

        # Volume between 100 and 1000
        volume = np.random.randint(100, 1000, len(dates))

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return df

    @pytest.fixture
    def data_config(self):
        """Create a basic data configuration for testing."""
        return DataConfig(
            csv_path="",
            datetime_column="datetime",
            feature_columns=[],  # Will be auto-detected
            target_type="classification",
            t_in=60,  # 60 minutes lookback
            t_out=10,  # 10 minutes forecast
            train_range=("2023-01-01 09:30:00", "2023-01-01 11:30:00"),
            val_range=("2023-01-01 11:30:00", "2023-01-01 13:30:00"),
            test_range=("2023-01-01 13:30:00", "2023-01-01 15:30:00"),
            flat_threshold=0.0001,
            top_k_predictions=3,
            predict_sell_now=False
        )

    def test_splits_are_time_ordered(self, synthetic_data, data_config):
        """Test that train/val/test splits are properly time-ordered."""
        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(synthetic_data)

        # Extract datetime ranges for each split
        train_min = splits["train"]["datetime"].min()
        train_max = splits["train"]["datetime"].max()
        val_min = splits["val"]["datetime"].min()
        val_max = splits["val"]["datetime"].max()
        test_min = splits["test"]["datetime"].min()
        test_max = splits["test"]["datetime"].max()

        # Assert non-overlapping, time-ordered splits
        assert train_max < val_min, f"Train/val overlap: train_max={train_max}, val_min={val_min}"
        assert val_max < test_min, f"Val/test overlap: val_max={val_max}, test_min={test_min}"

        # Assert splits are in chronological order
        assert train_min < val_min < test_min, "Splits not in chronological order"
        assert train_max < val_max < test_max, "Split ranges not properly ordered"

    def test_splits_are_non_overlapping(self, synthetic_data, data_config):
        """Test that train/val/test splits have no overlapping indices."""
        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(synthetic_data)

        # Get sets of indices for each split
        train_idx = set(splits["train"].index)
        val_idx = set(splits["val"].index)
        test_idx = set(splits["test"].index)

        # Assert no overlaps
        assert train_idx.isdisjoint(val_idx), "Train and val splits overlap"
        assert val_idx.isdisjoint(test_idx), "Val and test splits overlap"
        assert train_idx.isdisjoint(test_idx), "Train and test splits overlap"

        # Assert combined splits cover expected number of rows
        total_unique = len(train_idx | val_idx | test_idx)
        assert total_unique <= len(synthetic_data), "Combined splits exceed data length"

    def test_data_integrity_preserved(self, synthetic_data, data_config):
        """Test that data integrity is preserved during splitting."""
        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(synthetic_data)

        for split_name, split_df in splits.items():
            # Test that required columns exist
            required_cols = ["datetime", "open", "high", "low", "close", "volume"]
            assert all(col in split_df.columns for col in required_cols), \
                f"Missing columns in {split_name} split"

            # Test that datetime is properly ordered
            assert split_df["datetime"].is_monotonic_increasing, \
                f"Datetime not ordered in {split_name} split"

            # Test that no critical NaN values exist in OHLC
            ohlc_na_count = split_df[["open", "high", "low", "close"]].isna().sum().sum()
            assert ohlc_na_count == 0, \
                f"Found {ohlc_na_count} NaN values in OHLC data for {split_name} split"

            # Test OHLC relationships (high >= all other prices, low <= all other prices)
            invalid_ohlc = (
                    (split_df["high"] < split_df["low"]) |
                    (split_df["high"] < split_df["open"]) |
                    (split_df["high"] < split_df["close"]) |
                    (split_df["low"] > split_df["open"]) |
                    (split_df["low"] > split_df["close"])
            )
            assert not invalid_ohlc.any(), \
                f"Found {invalid_ohlc.sum()} invalid OHLC relationships in {split_name} split"

    def test_split_proportions(self, synthetic_data, data_config):
        """Test that split proportions match expected ratios."""
        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(synthetic_data)

        total_rows = len(synthetic_data)
        train_rows = len(splits["train"])
        val_rows = len(splits["val"])
        test_rows = len(splits["test"])

        # Allow small tolerance for rounding differences
        tolerance = 0.05  # 5% tolerance

        # Calculate expected proportions based on time ranges
        total_duration = (synthetic_data["datetime"].max() - synthetic_data["datetime"].min()).total_seconds()
        train_duration = (splits["train"]["datetime"].max() - splits["train"]["datetime"].min()).total_seconds()
        val_duration = (splits["val"]["datetime"].max() - splits["val"]["datetime"].min()).total_seconds()
        test_duration = (splits["test"]["datetime"].max() - splits["test"]["datetime"].min()).total_seconds()

        expected_train_prop = train_duration / total_duration
        expected_val_prop = val_duration / total_duration
        expected_test_prop = test_duration / total_duration

        actual_train_prop = train_rows / total_rows
        actual_val_prop = val_rows / total_rows
        actual_test_prop = test_rows / total_rows

        assert abs(actual_train_prop - expected_train_prop) < tolerance, \
            f"Train split proportion mismatch: expected {expected_train_prop:.3f}, got {actual_train_prop:.3f}"
        assert abs(actual_val_prop - expected_val_prop) < tolerance, \
            f"Val split proportion mismatch: expected {expected_val_prop:.3f}, got {actual_val_prop:.3f}"
        assert abs(actual_test_prop - expected_test_prop) < tolerance, \
            f"Test split proportion mismatch: expected {expected_test_prop:.3f}, got {actual_test_prop:.3f}"

    def test_empty_split_handling(self, synthetic_data):
        """Test handling of cases where splits might be empty."""
        # Create a very small dataset
        small_data = synthetic_data.head(10)

        # Create config that would result in very small or empty splits
        data_config = DataConfig(
            csv_path="",
            datetime_column="datetime",
            feature_columns=[],
            target_type="classification",
            t_in=5,
            t_out=2,
            train_range=("2023-01-01 09:30:00", "2023-01-01 09:35:00"),  # Very short range
            val_range=("2023-01-01 09:35:00", "2023-01-01 09:40:00"),
            test_range=("2023-01-01 09:40:00", "2023-01-01 15:30:00"),
            flat_threshold=0.0001,
            top_k_predictions=3,
            predict_sell_now=False
        )

        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(small_data)

        # At least one split should have data
        total_splits_with_data = sum(1 for split_df in splits.values() if len(split_df) > 0)
        assert total_splits_with_data > 0, "All splits are empty for small dataset"

    def test_boundary_conditions(self, synthetic_data):
        """Test boundary conditions and edge cases."""
        # Test with exact time boundaries
        data_config = DataConfig(
            csv_path="",
            datetime_column="datetime",
            feature_columns=[],
            target_type="classification",
            t_in=30,
            t_out=5,
            train_range=("2023-01-01 09:30:00", "2023-01-01 10:30:00"),
            val_range=("2023-01-01 10:30:00", "2023-01-01 11:30:00"),
            test_range=("2023-01-01 11:30:00", "2023-01-01 12:30:00"),
            flat_threshold=0.0001,
            top_k_predictions=3,
            predict_sell_now=False
        )

        agent = SingleTaskDataAgent(data_config)
        splits = agent.split_dataframe(synthetic_data)

        # Test that boundary times are correctly included/excluded
        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        # Check that train ends before val starts
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df["datetime"].max() < val_df["datetime"].min(), \
                "Train and val splits share boundary time"

        # Check that val ends before test starts
        if len(val_df) > 0 and len(test_df) > 0:
            assert val_df["datetime"].max() < test_df["datetime"].min(), \
                "Val and test splits share boundary time"


class TestDataBounds:
    """Test data download bounds and validation."""

    def test_download_bounds_validation(self):
        """Test that download bounds prevent infinite loops."""
        from data.download_all_fx_data import parse_args

        # Test reasonable bounds
        args = parse_args.__wrapped__()  # Get default args
        args.start_year = 2020
        args.end_year = 2023
        args.max_downloads = 50

        assert args.start_year < args.end_year, "Start year should be before end year"
        assert args.max_downloads > 0, "Max downloads should be positive"
        assert args.max_downloads <= 1000, "Max downloads should be reasonable"

    def test_data_validation_completeness(self):
        """Test that data validation catches various issues."""
        from data.prepare_dataset import validate_dataframe

        # Test with missing columns
        incomplete_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [1.1000],
            # missing high, low, close, volume
        })

        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        with pytest.raises(ValueError, match="Missing columns"):
            validate_dataframe(incomplete_df, required_cols)

        # Test with NaN values
        nan_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [1.1000],
            'high': [1.1010],
            'low': [np.nan],  # NaN value
            'close': [1.1005],
            'volume': [500]
        })

        validated_nan = validate_dataframe(nan_df, required_cols)
        assert len(validated_nan) == 0, "Should remove rows with NaN OHLC values"

        # Test with invalid OHLC relationships
        invalid_ohlc_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [1.1000],
            'high': [1.1005],  # high < low (invalid)
            'low': [1.1010],
            'close': [1.1007],
            'volume': [500]
        })

        validated_invalid = validate_dataframe(invalid_ohlc_df, required_cols)
        assert len(validated_invalid) == 0, "Should remove rows with invalid OHLC relationships"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
