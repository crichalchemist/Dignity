"""
Federal Reserve Economic Data (FRED) Downloader - Extended

Enhanced wrapper for the FRED API to download comprehensive economic indicators
for forex trading analysis. Extends the basic economic_indicators.py module
with advanced features from the FRB package.

Data Source:
- FRED API: https://fred.stlouisfed.org/
- Economic series: Interest rates, inflation, GDP, unemployment, etc.

Dependencies:
    Install from local package:
    pip install -e ./new_data_sources/FRB

Usage:
    from data.downloaders.fred_downloader import (
        download_multiple_series,
        get_forex_economic_indicators
    )

    # Download key economic indicators for EUR/USD analysis
    indicators = get_forex_economic_indicators(
        currency_pair="EURUSD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        api_key=os.getenv("FRED_API_KEY")
    )
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger
from utils.retry_utils import api_retry

logger = get_logger(__name__)

# Key economic series for major forex pairs
FOREX_ECONOMIC_SERIES = {
    "USD": {
        "interest_rate": "FEDFUNDS",  # Federal Funds Rate
        "inflation": "CPIAUCSL",  # CPI All Urban Consumers
        "gdp": "GDP",  # Gross Domestic Product
        "unemployment": "UNRATE",  # Unemployment Rate
        "trade_balance": "BOPGSTB",  # Trade Balance: Goods and Services
        "retail_sales": "RSXFS",  # Retail Sales
    },
    "EUR": {
        "interest_rate": "ECBDFR",  # ECB Deposit Facility Rate
        "inflation": "EA19CPALTT01GYM",  # Euro Area HICP
        "gdp": "CLVMNACSCAB1GQEA19",  # Euro Area Real GDP
        "unemployment": "LRHUTTTTEZM156S",  # Euro Area Unemployment
    },
    "GBP": {
        "interest_rate": "GBRCBBIRBM",  # UK Bank Rate
        "inflation": "GBRCPIALLMINMEI",  # UK CPI
        "gdp": "CLVMNACSCAB1GQGB",  # UK Real GDP
        "unemployment": "GBRUN",  # UK Unemployment Rate
    },
    "JPY": {
        "interest_rate": "JPNCBIR",  # Japan Call Rate
        "inflation": "JPNCPIALLMINMEI",  # Japan CPI
        "gdp": "JPNRGDPEXP",  # Japan Real GDP
        "unemployment": "JPNUE",  # Japan Unemployment Rate
    },
    "AUD": {
        "interest_rate": "AUSCBIR",  # Australia Cash Rate
        "inflation": "AUSCPIALLMINMEI",  # Australia CPI
        "gdp": "AUSGDP",  # Australia GDP
    },
    "CAD": {
        "interest_rate": "CANCBIR",  # Canada Overnight Rate
        "inflation": "CANCPIALLMINMEI",  # Canada CPI
        "gdp": "CANGDP",  # Canada GDP
    },
}


@api_retry
def _fetch_series_data(fred_client, series_id: str, start_date: str, end_date: str):
    """Internal function for API call with retry logic.
    
    Args:
        fred_client: Initialized Fred client
        series_id: FRED series identifier
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        
    Returns:
        Series data from FRED API
    """
    return fred_client.series.observations(
        series_id=series_id,
        observation_start=start_date,
        observation_end=end_date
    )


def download_series(
        series_id: str,
        start_date: str,
        end_date: str,
        api_key: str | None = None
) -> pd.DataFrame:
    """
    Download a single FRED series.

    Args:
        series_id: FRED series ID (e.g., "FEDFUNDS")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        api_key: FRED API key (or set FRED_API_KEY env var)

    Returns:
        DataFrame with columns: [date, value, series_id, series_name]

    Example:
        >>> df = download_series("FEDFUNDS", "2023-01-01", "2023-12-31")
        >>> print(df.head())
                date  value  series_id    series_name
        0 2023-01-01   4.33  FEDFUNDS  Federal Funds Rate
    """
    try:
        from fred import Fred
    except ImportError:
        raise ImportError(
            "fred package not installed. Install with: "
            "pip install -e ./new_data_sources/FRB"
        )

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Get one at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY environment variable or pass api_key parameter."
        )

    logger.info(f"[fred] Downloading series '{series_id}' from {start_date} to {end_date}")

    try:
        fred = Fred(api_key=api_key)

        # Get series data with retry
        series_data = _fetch_series_data(fred, series_id, start_date, end_date)

        if not series_data:
            logger.warning(f"[fred] No data returned for series {series_id}")
            return pd.DataFrame()

        # Get series metadata (may return None)
        series_info = fred.series.details(series_id=series_id)
        if series_info and isinstance(series_info, dict):
            series_name = series_info.get('title', series_id)
        else:
            logger.warning(f"[fred] Could not retrieve metadata for series {series_id}")
            series_name = f"Unknown Series: {series_id}"

        # Convert to DataFrame
        df = pd.DataFrame(series_data)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['series_id'] = series_id
        df['series_name'] = series_name

        # Remove rows with null values
        df = df.dropna(subset=['value'])

        logger.info(f"[fred] Downloaded {len(df)} observations for {series_id}")
        return df[['date', 'value', 'series_id', 'series_name']]

    except Exception as e:
        logger.error(f"[fred] Error downloading series {series_id}: {e}")
        raise


def download_multiple_series(
        series_ids: list[str],
        start_date: str,
        end_date: str,
        api_key: str | None = None
) -> pd.DataFrame:
    """
    Download multiple FRED series and combine into a single DataFrame.

    Args:
        series_ids: List of FRED series IDs
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        api_key: FRED API key

    Returns:
        Combined DataFrame with all series data

    Example:
        >>> series = ["FEDFUNDS", "CPIAUCSL", "UNRATE"]
        >>> df = download_multiple_series(series, "2023-01-01", "2023-12-31")
    """
    all_data = []

    for series_id in series_ids:
        try:
            df = download_series(series_id, start_date, end_date, api_key)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.warning(f"[fred] Failed to download {series_id}: {e}")
            continue

    if not all_data:
        logger.warning("[fred] No series data retrieved")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['series_id', 'date']).reset_index(drop=True)

    logger.info(f"[fred] Retrieved {len(result)} total observations from {len(all_data)} series")
    return result


def get_forex_economic_indicators(
        currency_pair: str,
        start_date: str,
        end_date: str,
        api_key: str | None = None,
        indicators: list[str] | None = None
) -> pd.DataFrame:
    """
    Download economic indicators relevant to a forex pair.

    Args:
        currency_pair: Forex pair (e.g., "EURUSD", "GBPUSD")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        api_key: FRED API key
        indicators: Optional list of specific indicators to download
                   (e.g., ["interest_rate", "inflation"])
                   If None, downloads all available indicators

    Returns:
        DataFrame with economic indicators for both currencies

    Example:
        >>> df = get_forex_economic_indicators("EURUSD", "2023-01-01", "2023-12-31")
        >>> print(df.head())
                date  value  series_id        series_name currency indicator_type
        0 2023-01-01   4.33  FEDFUNDS  Federal Funds Rate      USD   interest_rate
    """
    pair_upper = currency_pair.upper()

    # Extract base and quote currencies
    if len(pair_upper) != 6:
        raise ValueError(f"Invalid currency pair format: {currency_pair}")

    base_currency = pair_upper[:3]
    quote_currency = pair_upper[3:]

    all_data = []

    # Download indicators for both currencies
    for currency in [base_currency, quote_currency]:
        if currency not in FOREX_ECONOMIC_SERIES:
            logger.warning(f"[fred] No economic series defined for {currency}")
            continue

        series_map = FOREX_ECONOMIC_SERIES[currency]

        # Filter indicators if specified
        if indicators:
            series_map = {k: v for k, v in series_map.items() if k in indicators}

        for indicator_type, series_id in series_map.items():
            try:
                df = download_series(series_id, start_date, end_date, api_key)
                if not df.empty:
                    df['currency'] = currency
                    df['indicator_type'] = indicator_type
                    df['currency_pair'] = pair_upper
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"[fred] Failed to download {series_id} for {currency}: {e}")
                continue

    if not all_data:
        logger.warning(f"[fred] No economic indicators retrieved for {currency_pair}")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['currency', 'indicator_type', 'date']).reset_index(drop=True)

    logger.info(f"[fred] Retrieved {len(result)} economic indicator observations for {currency_pair}")
    return result


def format_for_cognee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format FRED data for Cognee ingestion with descriptive text.

    Args:
        df: Raw FRED DataFrame

    Returns:
        DataFrame with 'full_text' column for semantic search
    """
    df = df.copy()

    df['full_text'] = df.apply(
        lambda row: (
            f"{row['series_name']} for {row.get('currency', 'N/A')} "
            f"on {row['date'].strftime('%Y-%m-%d')}: {row['value']:.2f}. "
            f"Indicator type: {row.get('indicator_type', 'economic')}. "
            f"Series ID: {row['series_id']}."
        ),
        axis=1
    )

    return df
