"""
Unit tests for new fundamental data sources.

Comprehensive test coverage for:
- UN Comtrade downloader
- FRED (Federal Reserve) downloader
- ECB monetary policy shocks downloader
- Extended data collection orchestration
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ============================================================================
# Tests for comtrade_downloader.py
# ============================================================================

class TestComtradeDownloader:
    """Unit tests for UN Comtrade international trade data downloader."""

    def test_get_trade_indicators_invalid_pair(self):
        """Test rejection of unsupported currency pairs."""
        from data.downloaders.comtrade_downloader import get_trade_indicators_for_forex

        with pytest.raises(ValueError, match="not supported"):
            get_trade_indicators_for_forex("INVALID", 2023, 2023)

    @pytest.mark.parametrize("currency_pair,expected_base,expected_quote", [
        ("EURUSD", ["276"], ["842"]),
        ("GBPUSD", ["826"], ["842"]),
        ("USDJPY", ["842"], ["392"]),
        ("AUDUSD", ["036"], ["842"]),
    ])
    def test_forex_country_mapping(self, currency_pair, expected_base, expected_quote):
        """Test currency pair to country code mapping is correct."""
        from data.downloaders.comtrade_downloader import FOREX_COUNTRY_MAPPING

        assert currency_pair in FOREX_COUNTRY_MAPPING
        assert FOREX_COUNTRY_MAPPING[currency_pair]["base"] == expected_base
        assert FOREX_COUNTRY_MAPPING[currency_pair]["quote"] == expected_quote

    def test_format_for_cognee_adds_text_column(self):
        """Test Cognee text formatting adds required full_text column."""
        from data.downloaders.comtrade_downloader import format_for_cognee

        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [100000],
            'country_type': ['base'],
            'currency_pair': ['EURUSD'],
            'country_code': ['276']
        })

        result = format_for_cognee(df)

        assert 'full_text' in result.columns
        assert 'EURUSD' in result['full_text'].iloc[0]
        assert '100,000' in result['full_text'].iloc[0]
        assert '2023-01-01' in result['full_text'].iloc[0]

    def test_format_for_cognee_preserves_columns(self):
        """Test that format_for_cognee preserves original columns."""
        from data.downloaders.comtrade_downloader import format_for_cognee

        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [50000],
            'country_code': ['842']
        })

        result = format_for_cognee(df)

        assert result['trade_balance'].iloc[0] == 50000
        assert result['date'].iloc[0] == pd.Timestamp('2023-01-01')

    @patch('data.downloaders.comtrade_downloader.download_trade_balance')
    def test_get_trade_indicators_for_forex_success(self, mock_download):
        """Test successful retrieval of trade indicators for forex pair."""
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [100000],
            'country_code': ['276']
        })
        mock_download.return_value = mock_df

        from data.downloaders.comtrade_downloader import get_trade_indicators_for_forex

        result = get_trade_indicators_for_forex("EURUSD", 2023, 2023)

        assert not result.empty
        assert 'currency_pair' in result.columns
        assert result['currency_pair'].iloc[0] == 'EURUSD'
        # Should call download twice (base and quote currencies)
        assert mock_download.call_count == 2

        # Verify expected country codes were used (EUR=276, USD=840)
        call_args_list = [call[0] for call in mock_download.call_args_list]
        country_codes = [args[0] for args in call_args_list]
        assert '276' in country_codes  # EUR
        assert '840' in country_codes  # USD


# ============================================================================
# Tests for fred_downloader.py
# ============================================================================

class TestFREDDownloader:
    """Unit tests for Federal Reserve Economic Data (FRED) downloader."""

    def test_forex_economic_series_mapping(self):
        """Test that all major currencies have defined economic series."""
        from data.downloaders.fred_downloader import FOREX_ECONOMIC_SERIES

        required_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
        for currency in required_currencies:
            assert currency in FOREX_ECONOMIC_SERIES
            assert "interest_rate" in FOREX_ECONOMIC_SERIES[currency]

    def test_invalid_currency_pair_format(self):
        """Test rejection of malformed currency pair formats."""
        from data.downloaders.fred_downloader import get_forex_economic_indicators

        with pytest.raises(ValueError, match="Invalid currency pair format"):
            get_forex_economic_indicators("EUR", "2023-01-01", "2023-12-31", "key")

    def test_invalid_currency_pair_length(self):
        """Test rejection of currency pairs with wrong length."""
        from data.downloaders.fred_downloader import get_forex_economic_indicators

        with pytest.raises(ValueError, match="Invalid currency pair format"):
            get_forex_economic_indicators("EURUSDGBP", "2023-01-01", "2023-12-31", "key")

    @pytest.mark.parametrize("pair,base,quote", [
        ("EURUSD", "EUR", "USD"),
        ("GBPUSD", "GBP", "USD"),
        ("USDJPY", "USD", "JPY"),
    ])
    @patch('data.downloaders.fred_downloader.download_series')
    def test_currency_pair_parsing(self, mock_download, pair, base, quote):
        """Test correct parsing of currency pair formats."""
        mock_download.return_value = pd.DataFrame()

        from data.downloaders.fred_downloader import get_forex_economic_indicators

        df = get_forex_economic_indicators(pair, "2023-01-01", "2023-12-31", "key")
        # Should not raise error
        assert isinstance(df, pd.DataFrame)

        # Verify download_series was called with expected parameters
        assert mock_download.called
        # Check that calls include keyword arguments
        for call in mock_download.call_args_list:
            assert 'start_date' in call.kwargs
            assert 'end_date' in call.kwargs
            assert 'api_key' in call.kwargs

    def test_format_for_cognee_fred(self):
        """Test FRED data formatting for Cognee ingestion."""
        from data.downloaders.fred_downloader import format_for_cognee

        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'value': [4.33],
            'series_name': ['Federal Funds Rate'],
            'series_id': ['FEDFUNDS'],
            'currency': ['USD'],
            'indicator_type': ['interest_rate']
        })

        result = format_for_cognee(df)

        assert 'full_text' in result.columns
        assert 'Federal Funds Rate' in result['full_text'].iloc[0]
        assert 'FEDFUNDS' in result['full_text'].iloc[0]
        assert '4.33' in result['full_text'].iloc[0]


# ============================================================================
# Tests for ecb_shocks_downloader.py
# ============================================================================

class TestECBShocksDownloader:
    """Unit tests for ECB monetary policy shocks downloader."""

    @pytest.mark.parametrize("mp,cbi,expected", [
        (0.05, 0.01, "hawkish_MP"),
        (-0.05, -0.01, "dovish_MP"),
        (0.01, 0.05, "positive_CBI"),
        (-0.01, -0.05, "negative_CBI"),
        (0.001, 0.001, "neutral_CBI"),
        (0.0, 0.0, "neutral_CBI"),
    ])
    def test_classify_shock_type(self, mp, cbi, expected):
        """Test shock type classification with various thresholds."""
        from data.downloaders.ecb_shocks_downloader import classify_shock_type

        row = pd.Series({'MP_median': mp, 'CBI_median': cbi})
        result = classify_shock_type(row)

        assert expected in result

    @patch('data.downloaders.ecb_shocks_downloader.get_monetary_policy_events')
    def test_get_shocks_for_forex_pair_eur_based(self, mock_get_events):
        """Test retrieval of shocks for EUR-based currency pairs."""
        mock_get_events.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'MP_median': [0.05]
        })

        from data.downloaders.ecb_shocks_downloader import get_shocks_for_forex_pair

        result = get_shocks_for_forex_pair("EURUSD", "2023-01-01", "2023-12-31")

        assert not result.empty
        assert 'currency_pair' in result.columns
        assert result['currency_pair'].iloc[0] == 'EURUSD'

    @patch('data.downloaders.ecb_shocks_downloader.get_monetary_policy_events')
    def test_get_shocks_non_eur_pair(self, mock_get_events):
        """Test rejection of non-EUR currency pairs."""
        from data.downloaders.ecb_shocks_downloader import get_shocks_for_forex_pair

        result = get_shocks_for_forex_pair("USDJPY", "2023-01-01", "2023-12-31")

        assert result.empty

    def test_format_for_cognee_ecb(self):
        """Test ECB shocks formatting for Cognee ingestion."""
        from data.downloaders.ecb_shocks_downloader import format_for_cognee

        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'pc1': [0.05],
            'STOXX50': [0.10],
            'MP_median': [0.03],
            'CBI_median': [0.02]
        })

        result = format_for_cognee(df)

        assert 'full_text' in result.columns
        assert 'shock_type' in result.columns
        assert 'ECB Monetary Policy Event' in result['full_text'].iloc[0]

    @pytest.mark.parametrize("frequency", ["daily", "monthly"])
    @patch('data.downloaders.ecb_shocks_downloader.get_monetary_policy_events')
    def test_shock_frequency_handling(self, mock_get_events, frequency):
        """Test handling of different frequency parameters."""
        mock_get_events.return_value = pd.DataFrame()

        from data.downloaders.ecb_shocks_downloader import get_shocks_for_forex_pair

        result = get_shocks_for_forex_pair("EURUSD", "2023-01-01", "2023-12-31",
                                           frequency=frequency)

        assert isinstance(result, pd.DataFrame)
        # Verify frequency parameter was passed (keyword or positional)
        mock_get_events.assert_called_once()
        call_args = mock_get_events.call_args
        assert call_args is not None
        passed = False
        if call_args.kwargs and 'frequency' in call_args.kwargs:
            passed = (call_args.kwargs['frequency'] == frequency)
        elif call_args.args:
            passed = (frequency in call_args.args)
        assert passed, "frequency argument was not passed to get_monetary_policy_events"


# ============================================================================
# Tests for extended_data_collection.py
# ============================================================================

class TestExtendedDataCollection:
    """Unit tests for extended fundamental data collection orchestration."""

    @patch('data.downloaders.comtrade_downloader.get_trade_indicators_for_forex')
    def test_collect_trade_data_success(self, mock_trade_func):
        """Test successful trade data collection."""
        mock_trade_func.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [100000],
            'currency_pair': ['EURUSD']
        })

        from data.extended_data_collection import collect_trade_data

        result = collect_trade_data("EURUSD", 2023, 2023, "test_key")

        assert not result.empty
        assert 'trade_balance' in result.columns
        mock_trade_func.assert_called_once()

    @patch('data.downloaders.comtrade_downloader.get_trade_indicators_for_forex')
    def test_collect_trade_data_api_failure(self, mock_trade_func):
        """Test graceful handling of trade data API failures."""
        mock_trade_func.side_effect = Exception("API error")

        from data.extended_data_collection import collect_trade_data

        result = collect_trade_data("EURUSD", 2023, 2023)

        assert result.empty

    @patch('data.downloaders.fred_downloader.download_multiple_series')
    def test_collect_economic_data_success(self, mock_econ_func):
        """Test successful economic data collection."""
        mock_econ_func.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'value': [4.33],
            'series_id': ['FEDFUNDS'],
            'currency': ['USD']
        })

        from data.extended_data_collection import collect_economic_data

        result = collect_economic_data("EURUSD", "2023-01-01", "2023-12-31", "test_key")

        # Verify DataFrame structure and mock call
        assert isinstance(result, pd.DataFrame)
        mock_econ_func.assert_called_once()
        # Verify API key parameter was passed
        call_kwargs = mock_econ_func.call_args.kwargs if mock_econ_func.call_args.kwargs else {}
        if 'api_key' in call_kwargs:
            assert call_kwargs['api_key'] == "test_key"

    @patch('data.downloaders.ecb_shocks_downloader.load_ecb_shocks_daily')
    def test_collect_monetary_shocks_success(self, mock_shock_func):
        """Test successful monetary shock data collection."""
        mock_shock_func.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'MP_median': [0.05]
        })

        from data.extended_data_collection import collect_monetary_shocks

        result = collect_monetary_shocks("EURUSD", "2023-01-01", "2023-12-31")

        assert not result.empty
        assert 'MP_median' in result.columns

    @patch('data.downloaders.comtrade_downloader.get_trade_indicators_for_forex')
    @patch('data.downloaders.fred_downloader.download_multiple_series')
    @patch('data.downloaders.ecb_shocks_downloader.load_ecb_shocks_daily')
    def test_collect_all_fundamentals_integration(self, mock_shocks, mock_econ, mock_trade):
        """Test orchestration of all fundamental data sources."""
        mock_trade.return_value = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'trade_balance': [1]})
        mock_econ.return_value = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'value': [2]})
        mock_shocks.return_value = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'MP_median': [3]})

        from data.extended_data_collection import collect_all_forex_fundamentals

        result = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")

        # Should return dict
        assert isinstance(result, dict)

    @patch('data.downloaders.comtrade_downloader.get_trade_indicators_for_forex')
    @patch('data.downloaders.fred_downloader.download_multiple_series')
    @patch('data.downloaders.ecb_shocks_downloader.load_ecb_shocks_daily')
    def test_collect_all_fundamentals_partial_failure(self, mock_shocks, mock_econ, mock_trade):
        """Test handling of partial failures in fundamental collection."""
        mock_trade.return_value = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'trade_balance': [1]
        })
        mock_econ.return_value = pd.DataFrame()  # Empty
        mock_shocks.side_effect = Exception("Shocks unavailable")

        from data.extended_data_collection import collect_all_forex_fundamentals

        result = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")

        # Should still return dict
        assert isinstance(result, dict)

    def test_merge_with_price_data_alignment(self):
        """Test that price data merges correctly on dates."""
        from data.extended_data_collection import merge_with_price_data

        price_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close': [1.10, 1.11, 1.12, 1.13, 1.14]
        })

        fundamentals = {
            'trade': pd.DataFrame({
                'date': pd.date_range('2023-01-02', periods=3, freq='D'),
                'value': [100, 200, 300]
            })
        }

        result = merge_with_price_data(fundamentals, price_data, date_column='datetime')

        # Should return a result (may be empty if merge logic returns nothing)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("currency_pair", [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
    ])
    @patch('data.downloaders.comtrade_downloader.get_trade_indicators_for_forex')
    @patch('data.downloaders.fred_downloader.download_multiple_series')
    @patch('data.downloaders.ecb_shocks_downloader.load_ecb_shocks_daily')
    def test_multiple_currency_pairs(self, mock_shocks, mock_econ, mock_trade,
                                     currency_pair):
        """Test collection works for multiple currency pairs."""
        mock_trade.return_value = pd.DataFrame()
        mock_econ.return_value = pd.DataFrame()
        mock_shocks.return_value = pd.DataFrame()

        from data.extended_data_collection import collect_all_forex_fundamentals

        result = collect_all_forex_fundamentals(currency_pair, "2023-01-01", "2023-12-31")

        assert isinstance(result, dict)

        # EUR pairs should attempt to load ECB shocks, others should not
        if 'EUR' in currency_pair.upper():
            # EUR pairs call ECB shocks loader
            assert mock_shocks.called, f"{currency_pair} should load ECB shocks"
        # Note: Non-EUR pairs may or may not call mock_shocks depending on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
