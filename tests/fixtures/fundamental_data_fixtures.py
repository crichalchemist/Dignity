"""
Fixtures for fundamental data source testing.

Provides mock API responses and sample data for:
- UN Comtrade API
- FRED API
- ECB Monetary Policy Shocks
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture
def mock_comtrade_response():
    """Mock UN Comtrade API response."""
    return {
        "data": [
            {
                "period": "202301",
                "reporter": "USA",
                "partner": "World",
                "tradeFlow": "Export",
                "primaryValue": 150000000000,
                "customsCode": "C00",
                "motCode": "0",
                "qty": None,
                "netWgt": None
            },
            {
                "period": "202301",
                "reporter": "USA",
                "partner": "World",
                "tradeFlow": "Import",
                "primaryValue": 140000000000,
                "customsCode": "C00",
                "motCode": "0",
                "qty": None,
                "netWgt": None
            },
            {
                "period": "202302",
                "reporter": "USA",
                "partner": "World",
                "tradeFlow": "Export",
                "primaryValue": 152000000000,
                "customsCode": "C00",
                "motCode": "0",
                "qty": None,
                "netWgt": None
            },
            {
                "period": "202302",
                "reporter": "USA",
                "partner": "World",
                "tradeFlow": "Import",
                "primaryValue": 145000000000,
                "customsCode": "C00",
                "motCode": "0",
                "qty": None,
                "netWgt": None
            }
        ],
        "count": 4
    }


@pytest.fixture
def mock_fred_series_response():
    """Mock FRED API series observations response."""
    return [
        {"date": "2023-01-01", "value": "4.33"},
        {"date": "2023-02-01", "value": "4.57"},
        {"date": "2023-03-01", "value": "4.65"},
        {"date": "2023-04-01", "value": "4.83"},
        {"date": "2023-05-01", "value": "5.05"}
    ]


@pytest.fixture
def mock_fred_series_info():
    """Mock FRED API series metadata response."""
    return {
        "id": "FEDFUNDS",
        "title": "Federal Funds Effective Rate",
        "observation_start": "1954-07-01",
        "observation_end": "2023-12-01",
        "frequency": "Monthly",
        "units": "Percent",
        "seasonal_adjustment": "Not Seasonally Adjusted"
    }


@pytest.fixture
def mock_ecb_shocks_daily():
    """Mock ECB monetary policy shocks (daily frequency)."""
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
    return pd.DataFrame({
        "date": dates,
        "MP_median": [0.001, -0.002, 0.0, 0.003, -0.001, 0.0, 0.002, -0.003, 0.001, 0.0],
        "CBI_median": [0.002, 0.001, -0.001, 0.0, 0.003, -0.002, 0.0, 0.001, -0.001, 0.002],
        "MP_mean": [0.0012, -0.0018, 0.0, 0.0028, -0.0009, 0.0, 0.0021, -0.0032, 0.0011, 0.0],
        "CBI_mean": [0.0019, 0.0012, -0.0008, 0.0, 0.0029, -0.0021, 0.0, 0.0009, -0.0012, 0.0018]
    })


@pytest.fixture
def mock_ecb_shocks_monthly():
    """Mock ECB monetary policy shocks (monthly frequency)."""
    dates = pd.date_range("2023-01-01", "2023-05-01", freq="ME")
    return pd.DataFrame({
        "date": dates,
        "MP_median": [0.005, -0.003, 0.002, 0.001, -0.004],
        "CBI_median": [0.003, 0.002, -0.002, 0.004, -0.001],
        "MP_mean": [0.0048, -0.0032, 0.0019, 0.0011, -0.0041],
        "CBI_mean": [0.0029, 0.0021, -0.0018, 0.0038, -0.0009]
    })


@pytest.fixture
def sample_price_data():
    """Sample OHLCV price data for testing merge operations."""
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="h")
    n = len(dates)

    return pd.DataFrame({
        "timestamp": dates,
        "open": 1.08 + (0.001 * pd.Series(range(n))),
        "high": 1.081 + (0.001 * pd.Series(range(n))),
        "low": 1.079 + (0.001 * pd.Series(range(n))),
        "close": 1.0805 + (0.001 * pd.Series(range(n))),
        "volume": 1000 + (10 * pd.Series(range(n)))
    })


@pytest.fixture
def currency_pairs():
    """List of currency pairs for parametrized testing."""
    return [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "USDCAD",
        "EURGBP"
    ]


@pytest.fixture
def date_ranges():
    """Common date ranges for testing."""
    return {
        "short": ("2023-01-01", "2023-01-31"),
        "medium": ("2023-01-01", "2023-06-30"),
        "long": ("2022-01-01", "2023-12-31"),
        "invalid": ("2024-01-01", "2023-01-01")  # End before start
    }


@pytest.fixture
def mock_comtrade_error_response():
    """Mock Comtrade API error response."""
    return {
        "error": "Invalid API key",
        "statusCode": 401,
        "message": "Unauthorized"
    }


@pytest.fixture
def mock_fred_error_response():
    """Mock FRED API error response."""
    return {
        "error_code": 400,
        "error_message": "Bad Request. The value for variable api_key is not registered."
    }


# Expected DataFrame schemas for validation

@pytest.fixture
def expected_comtrade_schema():
    """Expected schema for Comtrade trade balance data."""
    return {
        "columns": ["date", "trade_balance", "exports", "imports", "country"],
        "dtypes": {
            "date": "datetime64[ns]",
            "trade_balance": "float64",
            "exports": "float64",
            "imports": "float64",
            "country": "object"
        }
    }


@pytest.fixture
def expected_fred_schema():
    """Expected schema for FRED economic indicators."""
    return {
        "columns": ["date", "value", "series_id", "series_name", "currency", "indicator_type"],
        "dtypes": {
            "date": "datetime64[ns]",
            "value": "float64",
            "series_id": "object",
            "series_name": "object",
            "currency": "object",
            "indicator_type": "object"
        }
    }


@pytest.fixture
def expected_ecb_shocks_schema():
    """Expected schema for ECB monetary policy shocks."""
    return {
        "columns": ["date", "MP_median", "CBI_median", "MP_mean", "CBI_mean"],
        "dtypes": {
            "date": "datetime64[ns]",
            "MP_median": "float64",
            "CBI_median": "float64",
            "MP_mean": "float64",
            "CBI_mean": "float64"
        }
    }
