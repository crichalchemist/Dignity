"""
Test suite for Colab data collection pipeline.

Tests the complete end-to-end workflow including:
- Package imports and installation
- Checkpoint state management
- Data download retry logic
- Path configuration fixes
- Fundamental data integration
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mock Colab-specific modules that won't be available in test environment
sys.modules["google.colab"] = Mock()
sys.modules["google.colab.drive"] = Mock()
sys.modules["getpass"] = Mock()

# Mock the colab_data_collection module since it no longer exists
class MockColabConfig:
    def __init__(self):
        self.pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"]
        self.start_year = 2010
        self.output_dir = Path("/tmp/colab_data")

class MockDownloadState:
    def __init__(self):
        self.completed_years = []
        self.failures = {}

class MockValidationResult:
    def __init__(self):
        self.passed = True

ColabConfig = MockColabConfig
DownloadState = MockDownloadState
ValidationResult = MockValidationResult

# Import test targets
from data.downloaders.fred_downloader import download_series
from data.downloaders.comtrade_downloader import download_trade_balance
from data.downloaders.ecb_shocks_downloader import load_ecb_shocks_daily
from utils.retry_utils import retry_with_backoff, rate_limit, RetryContext


class TestColabConfiguration:
    """Test Colab configuration class."""

    def test_config_initialization(self):
        """Test config initializes with correct paths."""
        config = ColabConfig()

        # Test default values
        assert len(config.pairs) == 6
        assert config.start_year == 2010
        assert config.end_year == 2024
        assert config.task_type == "classification"

        # Test critical path fixes
        assert "output_central" in str(config.raw_data_dir)
        assert "output_central" in str(config.prepared_data_dir)
        assert "output_central" in str(config.fundamental_dir)

    def test_config_serialization(self):
        """Test config can be serialized to/from dict."""
        config = ColabConfig()

        # Convert to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["start_year"] == 2010
        assert isinstance(config_dict["root"], str)  # Path should be converted

        # Convert back from dict
        restored_config = ColabConfig.from_dict(config_dict)
        assert restored_config.start_year == config.start_year
        assert isinstance(restored_config.root, Path)  # Should be converted back to Path

    def test_path_consistency(self):
        """Test all paths use output_central consistently."""
        config = ColabConfig()

        # All paths should point to output_central, not data/raw
        raw_path_str = str(config.raw_data_dir)
        prepared_path_str = str(config.prepared_data_dir)
        fundamental_path_str = str(config.fundamental_dir)

        assert "output_central" in raw_path_str
        assert "data/raw" not in raw_path_str
        assert "output_central" in prepared_path_str
        assert "output_central" in fundamental_path_str


class TestDownloadState:
    """Test checkpoint state management."""

    def test_state_initialization(self):
        """Test state starts empty."""
        state = DownloadState()

        assert state.completed == {}
        assert state.failed == {}
        assert state.fundamentals_completed == {}
        assert state.last_checkpoint == ""

    def test_year_completion_tracking(self):
        """Test tracking of completed years."""
        state = DownloadState()

        # Mark years as completed
        state.mark_year_completed("eurusd", 2020)
        state.mark_year_completed("eurusd", 2021)
        state.mark_year_completed("gbpusd", 2020)

        # Test completion status
        assert state.is_year_completed("eurusd", 2020) == True
        assert state.is_year_completed("eurusd", 2021) == True
        assert state.is_year_completed("eurusd", 2019) == False
        assert state.is_year_completed("gbpusd", 2020) == True

        # Test completed data structure
        assert state.completed["eurusd"] == [2020, 2021]
        assert state.completed["gbpusd"] == [2020]

    def test_failure_tracking(self):
        """Test tracking of failed years."""
        state = DownloadState()

        # Mark years as failed
        state.mark_year_failed("eurusd", 2020, "Network timeout")
        state.mark_year_failed("eurusd", 2021, "API rate limit")

        # Test failure data
        assert len(state.failed["eurusd"]) == 2
        assert state.failed["eurusd"][0] == (2020, "Network timeout")
        assert state.failed["eurusd"][1] == (2021, "API rate limit")

    def test_state_serialization(self):
        """Test state can be saved and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "test_state.json"

            # Create and populate state
            original_state = DownloadState()
            original_state.mark_year_completed("eurusd", 2020)
            original_state.mark_year_failed("gbpusd", 2021, "Test error")
            original_state.fundamentals_completed["fred"] = True

            # Save state
            original_state.save_to_drive(state_file)
            assert state_file.exists()

            # Load state
            loaded_state = DownloadState.load_from_drive(state_file)

            # Verify loaded state
            assert loaded_state.is_year_completed("eurusd", 2020) == True
            assert len(loaded_state.failed["gbpusd"]) == 1
            assert loaded_state.fundamentals_completed["fred"] == True

    def test_progress_summary(self):
        """Test progress summary generation."""
        state = DownloadState()

        # Add some progress
        state.mark_year_completed("eurusd", 2020)
        state.mark_year_completed("eurusd", 2021)
        state.mark_year_completed("gbpusd", 2020)

        summary = state.get_summary()

        assert "Download Progress:" in summary
        assert "Completed: 3/" in summary  # 3 completed out of total possible
        assert "Failed: 0" in summary


class TestPackageImports:
    """Test that all required packages can be imported."""

    def test_fred_import(self):
        """Test FRED package import with fallback handling."""
        try:
            from fred import Fred

            assert Fred is not None
        except ImportError:
            # Should be handled gracefully by downloader
            from data.downloaders.fred_downloader import download_series

            # Should raise ImportError when called without proper installation
            with pytest.raises(ImportError, match="fred package not installed"):
                download_series("TEST", "2023-01-01", "2023-01-31")

    def test_comtrade_import(self):
        """Test Comtrade package import with fallback handling."""
        try:
            from comtradeapicall import previewGet

            assert previewGet is not None
        except ImportError:
            # Should be handled gracefully by downloader
            from data.downloaders.comtrade_downloader import download_trade_balance

            # Should raise ImportError when called without proper installation
            with pytest.raises(ImportError, match="comtradeapicall not installed"):
                download_trade_balance("842", 2023, 2023)

    def test_ecb_shocks_vendored(self):
        """Test ECB shocks data is accessible via vendored path."""
        # Test the vendor wrapper exists
        from data.vendors.ecb_shocks import verify_data_exists, get_data_dir

        # Test directory structure exists
        data_dir = get_data_dir()
        assert data_dir.exists()

        # Test verification function
        if verify_data_exists():
            # If data exists, test loading
            try:
                daily_shocks = load_ecb_shocks_daily()
                assert isinstance(daily_shocks, pd.DataFrame)
                assert "date" in daily_shocks.columns
            except FileNotFoundError:
                # Data files might not exist in test environment
                pass


class TestRetryLogic:
    """Test retry decorators and error handling."""

    def test_retry_with_backoff_success(self):
        """Test retry decorator succeeds on first try."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_with_backoff_eventual_success(self):
        """Test retry decorator succeeds after failures."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_with_backoff_exhausted(self):
        """Test retry decorator raises after max retries."""

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def test_function():
            raise ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError, match="Persistent failure"):
            test_function()

    def test_rate_limiting(self):
        """Test rate limiting decorator."""
        import time

        call_times = []

        @rate_limit(calls_per_second=10)  # 0.1 second between calls
        def test_function():
            call_times.append(time.time())
            return "ok"

        # Make multiple calls quickly
        start_time = time.time()
        for _ in range(3):
            test_function()

        # Verify calls were spaced out
        assert len(call_times) == 3
        # Should have delays between calls
        assert call_times[1] - call_times[0] >= 0.08  # Allow some tolerance
        assert call_times[2] - call_times[1] >= 0.08

    def test_retry_context(self):
        """Test retry context manager."""
        call_count = 0

        with RetryContext(max_retries=3, base_delay=0.01) as retry:

            def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ValueError("Try again")
                return "success"

            result = retry.attempt(failing_function)
            assert result == "success"

        stats = retry.get_stats()
        assert stats["attempts"] == 3
        assert stats["successes"] == 1
        assert stats["failures"] == 0


class TestPathFixes:
    """Test that critical path fixes are applied correctly."""

    def test_prepare_dataset_input_root(self):
        """Test that prepare_dataset is called with correct input_root."""
        # This test verifies the fix for data/raw vs output_central
        config = ColabConfig()

        # Check that the path uses output_central
        assert "output_central" in str(config.raw_data_dir)
        assert "data/raw" not in str(config.raw_data_dir)

        # Test command building logic would use correct path
        expected_input_root = str(config.raw_data_dir)
        assert expected_input_root.endswith("output_central")

    def test_colab_notebook_path_consistency(self):
        """Test that Colab notebooks use consistent paths."""
        config = ColabConfig()

        # All data-related paths should be under output_central
        base_data_dir = config.root / "output_central"

        assert config.raw_data_dir == base_data_dir
        assert config.prepared_data_dir == base_data_dir
        assert config.fundamental_dir == base_data_dir / "fundamentals"


class TestFundamentalDataIntegration:
    """Test fundamental data collection and integration."""

    def test_ecb_shocks_eur_pair_filtering(self):
        """Test ECB shocks are filtered for EUR pairs only."""
        try:
            from data.downloaders.ecb_shocks_downloader import get_shocks_for_forex_pair

            # Test EUR pair (should work)
            eur_data = get_shocks_for_forex_pair("EURUSD", "2023-01-01", "2023-12-31")
            # Should return DataFrame or empty DataFrame

            # Test non-EUR pair (should return empty or warning)
            non_eur_data = get_shocks_for_forex_pair("USDJPY", "2023-01-01", "2023-12-31")
            # Should return empty DataFrame or handle gracefully

        except FileNotFoundError:
            # ECB shocks data might not be available in test environment
            pass

    def test_fred_series_mapping(self):
        """Test FRED series are correctly mapped for forex pairs."""
        from data.downloaders.fred_downloader import FOREX_ECONOMIC_SERIES

        # Test that major currencies have series defined
        assert "USD" in FOREX_ECONOMIC_SERIES
        assert "EUR" in FOREX_ECONOMIC_SERIES
        assert "GBP" in FOREX_ECONOMIC_SERIES
        assert "JPY" in FOREX_ECONOMIC_SERIES

        # Test that required indicators are present
        usd_indicators = FOREX_ECONOMIC_SERIES["USD"]
        assert "interest_rate" in usd_indicators
        assert "inflation" in usd_indicators
        assert "gdp" in usd_indicators

    def test_comtrade_country_mapping(self):
        """Test Comtrade countries are correctly mapped for forex pairs."""
        from data.downloaders.comtrade_downloader import FOREX_COUNTRY_MAPPING

        # Test that major pairs have country mappings
        assert "EURUSD" in FOREX_COUNTRY_MAPPING
        assert "GBPUSD" in FOREX_COUNTRY_MAPPING
        assert "USDJPY" in FOREX_COUNTRY_MAPPING

        # Test mapping structure
        eurusd_mapping = FOREX_COUNTRY_MAPPING["EURUSD"]
        assert "base" in eurusd_mapping
        assert "quote" in eurusd_mapping
        assert isinstance(eurusd_mapping["base"], list)
        assert isinstance(eurusd_mapping["quote"], list)


class TestDataValidation:
    """Test data validation pipeline."""

    def test_validation_result_creation(self):
        """Test ValidationResult dataclass."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = ValidationResult(
            total_rows=1000,
            date_range=(start_date, end_date),
            missing_months=["2023-06"],
            corrupt_files=[Path("bad.csv")],
            warnings=["Some warning"],
        )

        assert result.total_rows == 1000
        assert result.date_range == (start_date, end_date)
        assert len(result.missing_months) == 1
        assert len(result.corrupt_files) == 1
        assert len(result.warnings) == 1
        assert result.is_valid() == False  # Has issues

    def test_validation_result_perfect(self):
        """Test ValidationResult with perfect data."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = ValidationResult(
            total_rows=1000,
            date_range=(start_date, end_date),
            missing_months=[],
            corrupt_files=[],
            warnings=[],
        )

        assert result.is_valid() == True

    def test_validation_serialization(self):
        """Test ValidationResult can be serialized to dict."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        result = ValidationResult(
            total_rows=1000,
            date_range=(start_date, end_date),
            missing_months=[],
            corrupt_files=[],
            warnings=[],
        )

        result_dict = result.to_dict()

        assert result_dict["total_rows"] == 1000
        assert result_dict["date_range"][0] == start_date.isoformat()
        assert result_dict["date_range"][1] == end_date.isoformat()
        assert result_dict["is_valid"] == True


class TestColabIntegration:
    """Integration tests for Colab-specific functionality."""

    def test_checkpoint_file_path(self):
        """Test checkpoint file is in Google Drive."""
        config = ColabConfig()

        # Checkpoint should be in Google Drive
        checkpoint_path = config.checkpoint_file
        assert "drive" in str(checkpoint_path)
        assert "Sequence_Data" in str(checkpoint_path)
        assert checkpoint_path.name == "checkpoint.json"

    def test_backup_directory_structure(self):
        """Test backup directories are created correctly."""
        config = ColabConfig()

        backup_dir = config.backup_dir
        assert "drive" in str(backup_dir)
        assert "Sequence_Data" in str(backup_dir)
        assert "backup" in str(backup_dir)

    def test_configuration_dataclass_completeness(self):
        """Test configuration has all required fields."""
        config = ColabConfig()

        # Check all expected attributes exist
        expected_attrs = [
            "pairs",
            "start_year",
            "end_year",
            "root",
            "raw_data_dir",
            "prepared_data_dir",
            "fundamental_dir",
            "checkpoint_file",
            "backup_dir",
            "checkpoint_frequency",
            "backup_frequency",
            "t_in",
            "t_out",
            "task_type",
            "use_intrinsic_time",
            "dc_threshold_up",
            "dc_threshold_down",
        ]

        for attr in expected_attrs:
            assert hasattr(config, attr), f"Missing config attribute: {attr}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
