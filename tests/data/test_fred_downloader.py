"""
Unit tests for FRED downloader.

Tests the Federal Reserve Economic Data API wrapper with mocked responses.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from data.downloaders.fred_downloader import (
    FOREX_ECONOMIC_SERIES,
    download_multiple_series,
    download_series,
    get_forex_economic_indicators,
)


@pytest.fixture
def mock_fred_series_response():
    """Fixture providing mock FRED series observations."""
    return [
        {"date": "2023-01-01", "value": "4.33"},
        {"date": "2023-02-01", "value": "4.57"},
        {"date": "2023-03-01", "value": "4.65"},
        {"date": "2023-04-01", "value": "4.83"},
        {"date": "2023-05-01", "value": "5.06"},
    ]


@pytest.fixture
def mock_fred_series_info():
    """Fixture providing mock FRED series metadata."""
    return {"title": "Federal Funds Effective Rate"}


class TestFredDownloadSeries:
    """Test download_series function."""

    @patch("fred.Fred")
    def test_download_series_success(self, mock_fred_class, mock_fred_series_response, mock_fred_series_info):
        """Test successful series download."""
        # Setup mock
        mock_fred = Mock()
        mock_fred.series.observations.return_value = mock_fred_series_response
        mock_fred.series.details.return_value = mock_fred_series_info
        mock_fred_class.return_value = mock_fred

        # Execute
        df = download_series(
            series_id="FEDFUNDS",
            start_date="2023-01-01",
            end_date="2023-05-01",
            api_key="test_key"
        )

        # Verify
        assert not df.empty
        assert len(df) == 5
        assert list(df.columns) == ["date", "value", "series_id", "series_name"]
        assert df["series_id"].iloc[0] == "FEDFUNDS"
        assert df["series_name"].iloc[0] == "Federal Funds Effective Rate"

        # Verify API called correctly
        mock_fred.series.observations.assert_called_once_with(
            series_id="FEDFUNDS",
            observation_start="2023-01-01",
            observation_end="2023-05-01"
        )
        mock_fred.series.details.assert_called_once_with(series_id="FEDFUNDS")

    @patch("fred.Fred")
    def test_download_series_empty_response(self, mock_fred_class):
        """Test handling of empty API response."""
        mock_fred = Mock()
        mock_fred.series.observations.return_value = []
        mock_fred_class.return_value = mock_fred

        df = download_series(
            series_id="INVALID",
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key"
        )

        assert df.empty

    def test_download_series_no_api_key(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="FRED API key required"):
            download_series(
                series_id="FEDFUNDS",
                start_date="2023-01-01",
                end_date="2023-12-31",
                api_key=None
            )

    @patch("fred.Fred")
    def test_download_series_handles_nan_values(self, mock_fred_class):
        """Test that NaN values are dropped."""
        mock_fred = Mock()
        mock_fred.series.observations.return_value = [
            {"date": "2023-01-01", "value": "4.33"},
            {"date": "2023-02-01", "value": "."},  # Missing value
            {"date": "2023-03-01", "value": "4.65"},
        ]
        mock_fred.series.details.return_value = {"title": "Test Series"}
        mock_fred_class.return_value = mock_fred

        df = download_series(
            series_id="TEST",
            start_date="2023-01-01",
            end_date="2023-03-01",
            api_key="test_key"
        )

        # Should only have 2 rows (NaN dropped)
        assert len(df) == 2
        assert all(df["value"].notna())


class TestFredDownloadMultipleSeries:
    """Test download_multiple_series function."""

    @patch("data.downloaders.fred_downloader.download_series")
    def test_download_multiple_series_success(self, mock_download):
        """Test downloading multiple series."""
        # Setup mock to return different data for each series
        def mock_response(series_id, start_date, end_date, api_key):
            return pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=3, freq="ME"),
                "value": [1.0, 2.0, 3.0],
                "series_id": [series_id] * 3,
                "series_name": [f"Series {series_id}"] * 3
            })

        mock_download.side_effect = mock_response

        # Execute
        series_ids = ["FEDFUNDS", "CPIAUCSL", "UNRATE"]
        df = download_multiple_series(
            series_ids=series_ids,
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key"
        )

        # Verify
        assert not df.empty
        assert len(df) == 9  # 3 series * 3 observations each
        assert set(df["series_id"].unique()) == set(series_ids)
        assert mock_download.call_count == 3

    @patch("data.downloaders.fred_downloader.download_series")
    def test_download_multiple_series_partial_failure(self, mock_download):
        """Test that partial failures don't break entire download."""
        def mock_response(series_id, start_date, end_date, api_key):
            if series_id == "INVALID":
                raise Exception("Series not found")
            return pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=2, freq="ME"),
                "value": [1.0, 2.0],
                "series_id": [series_id] * 2,
                "series_name": [f"Series {series_id}"] * 2
            })

        mock_download.side_effect = mock_response

        # Execute with one invalid series
        df = download_multiple_series(
            series_ids=["FEDFUNDS", "INVALID", "CPIAUCSL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key"
        )

        # Should still return data from valid series
        assert not df.empty
        assert len(df) == 4  # 2 series * 2 observations each
        assert set(df["series_id"].unique()) == {"FEDFUNDS", "CPIAUCSL"}


class TestForexEconomicIndicators:
    """Test get_forex_economic_indicators function."""

    @patch("data.downloaders.fred_downloader.download_series")
    def test_get_forex_indicators_eurusd(self, mock_download):
        """Test downloading indicators for EUR/USD."""
        # Mock response
        def mock_response(series_id, start_date, end_date, api_key):
            return pd.DataFrame({
                "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
                "value": [1.0, 2.0],
                "series_id": [series_id] * 2,
                "series_name": [f"Series {series_id}"] * 2
            })

        mock_download.side_effect = mock_response

        # Execute
        df = get_forex_economic_indicators(
            currency_pair="EURUSD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key"
        )

        # Verify
        assert not df.empty
        assert "currency" in df.columns
        assert "indicator_type" in df.columns
        assert "currency_pair" in df.columns
        assert set(df["currency"].unique()) == {"EUR", "USD"}

        # Check that we attempted to download indicators for both currencies
        call_series_ids = [call.args[0] for call in mock_download.call_args_list]
        assert any(series in FOREX_ECONOMIC_SERIES["USD"].values() for series in call_series_ids)
        assert any(series in FOREX_ECONOMIC_SERIES["EUR"].values() for series in call_series_ids)

    @patch("data.downloaders.fred_downloader.download_series")
    def test_get_forex_indicators_filter_specific(self, mock_download):
        """Test filtering specific indicators."""
        mock_download.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
            "value": [1.0, 2.0],
            "series_id": ["FEDFUNDS"] * 2,
            "series_name": ["Federal Funds Rate"] * 2
        })

        # Execute with specific indicators
        df = get_forex_economic_indicators(
            currency_pair="EURUSD",
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key",
            indicators=["interest_rate", "inflation"]
        )

        # Verify that only specified indicators were requested
        call_series_ids = [call.args[0] for call in mock_download.call_args_list]
        # Should only have interest_rate and inflation series
        assert all(
            series_id in [
                FOREX_ECONOMIC_SERIES["USD"]["interest_rate"],
                FOREX_ECONOMIC_SERIES["USD"]["inflation"],
                FOREX_ECONOMIC_SERIES["EUR"]["interest_rate"],
                FOREX_ECONOMIC_SERIES["EUR"]["inflation"]
            ]
            for series_id in call_series_ids
        )

    def test_get_forex_indicators_invalid_pair(self):
        """Test handling of invalid currency pair format."""
        with pytest.raises(ValueError, match="Invalid currency pair format"):
            get_forex_economic_indicators(
                currency_pair="INVALID",
                start_date="2023-01-01",
                end_date="2023-12-31",
                api_key="test_key"
            )

    @pytest.mark.parametrize("pair", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    @patch("data.downloaders.fred_downloader.download_series")
    def test_get_forex_indicators_multiple_pairs(self, mock_download, pair):
        """Test that all major currency pairs work."""
        mock_download.return_value = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=1, freq="ME"),
            "value": [1.0],
            "series_id": ["TEST"],
            "series_name": ["Test"]
        })

        df = get_forex_economic_indicators(
            currency_pair=pair,
            start_date="2023-01-01",
            end_date="2023-12-31",
            api_key="test_key"
        )

        # Should succeed for all major pairs
        assert isinstance(df, pd.DataFrame)
        assert "currency_pair" in df.columns
        if not df.empty:
            assert df["currency_pair"].iloc[0] == pair.upper()


class TestForexEconomicSeries:
    """Test FOREX_ECONOMIC_SERIES mapping."""

    def test_all_major_currencies_defined(self):
        """Verify all major currencies have economic series defined."""
        expected_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
        assert all(curr in FOREX_ECONOMIC_SERIES for curr in expected_currencies)

    def test_usd_has_complete_indicators(self):
        """Verify USD has all expected indicators."""
        usd_indicators = set(FOREX_ECONOMIC_SERIES["USD"].keys())
        expected_indicators = {
            "interest_rate",
            "inflation",
            "gdp",
            "unemployment",
            "trade_balance",
            "retail_sales"
        }
        assert expected_indicators == usd_indicators

    def test_all_series_ids_are_strings(self):
        """Verify all series IDs are valid strings."""
        for currency, indicators in FOREX_ECONOMIC_SERIES.items():
            for indicator_type, series_id in indicators.items():
                assert isinstance(series_id, str)
                assert len(series_id) > 0
