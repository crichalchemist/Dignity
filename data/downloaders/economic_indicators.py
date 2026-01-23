"""
Economic Indicators Downloader

Download economic calendar data and central bank announcements for FX/crypto analysis.
Provides fundamental economic context to complement GDELT sentiment data.

Data Sources:
- ✅ FRED API (Federal Reserve Economic Data): Interest rates, CPI, unemployment, etc.
- ✅ Yahoo Finance Economic Calendar: Scheduled economic events
- 🔄 Manual parsing: Central bank meeting minutes (Fed, ECB, BOE, BOJ)

Dependencies:
    pip install fredapi yfinance beautifulsoup4

Usage:
    from data.downloaders.economic_indicators import (
        download_fred_series,
        download_economic_calendar,
        combine_economic_data
    )

    # Download Fed Funds Rate
    fed_funds = download_fred_series(
        series_id="FEDFUNDS",
        start_date="2020-01-01",
        end_date="2024-12-31",
        api_key=os.getenv("FRED_API_KEY")
    )

    # Download economic calendar
    calendar = download_economic_calendar(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    # Combine for Cognee ingestion
    combined = combine_economic_data([fed_funds, calendar])
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def download_fred_series(
        series_id: str,
        start_date: str,
        end_date: str,
        api_key: str | None = None
) -> pd.DataFrame:
    """
    Download economic time series from FRED (Federal Reserve Economic Data).

    Args:
        series_id: FRED series ID (e.g., "FEDFUNDS", "CPIAUCSL", "UNRATE")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        api_key: FRED API key (or set FRED_API_KEY env var)
                 Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html

    Returns:
        DataFrame with columns: [date, value, series_name, full_text]
        The 'full_text' column contains a description suitable for Cognee ingestion

    Example:
        >>> df = download_fred_series("FEDFUNDS", "2023-01-01", "2023-12-31", api_key)
        >>> print(df.head())
                date  value series_name                     full_text
        0 2023-01-01   4.33   FEDFUNDS  Fed Funds Rate: 4.33% on 2023-01-01
    """
    import os

    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi not installed. Install with: pip install fredapi"
        )

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Get one at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY environment variable or pass api_key parameter."
        )

    logger.info(f"[fred] Downloading series '{series_id}' from {start_date} to {end_date}")

    fred = Fred(api_key=api_key)

    # Download series
    series_data = fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date
    )

    # Get series metadata
    series_info = fred.get_series_info(series_id)
    series_name = series_info.get("title", series_id)
    series_units = series_info.get("units", "")

    # Convert to DataFrame
    df = pd.DataFrame({
        'date': series_data.index,
        'value': series_data.values,
        'series_id': series_id,
        'series_name': series_name
    })

    # Create text description for Cognee
    df['full_text'] = df.apply(
        lambda row: f"{series_name}: {row['value']}{series_units} on {row['date'].strftime('%Y-%m-%d')}. "
                    f"This is an economic indicator tracked by the Federal Reserve.",
        axis=1
    )

    # Add metadata columns for Cognee processor
    df['title'] = df['series_name']
    df['event_type'] = 'economic_indicator'
    df['bank_name'] = 'Federal Reserve'

    logger.info(f"[fred] Downloaded {len(df)} observations for {series_id}")

    return df


def download_economic_calendar(
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """
    Download economic calendar events.

    Note: This is a placeholder implementation. For production, consider:
    - Trading Economics API (paid): https://tradingeconomics.com/api
    - Investing.com scraper (use with caution - check ToS)
    - FX Factory Calendar API

    Args:
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"

    Returns:
        DataFrame with columns: [date, event_name, country, actual, forecast, previous, full_text, title]
    """
    logger.warning("[calendar] Economic calendar download not fully implemented")
    logger.info("[calendar] Returning empty DataFrame - implement with Trading Economics API or scraper")

    # Return empty DataFrame with correct schema
    return pd.DataFrame(columns=[
        'date', 'event_name', 'country', 'actual', 'forecast', 'previous',
        'full_text', 'title', 'event_type', 'bank_name'
    ])


def download_central_bank_minutes(
        bank: str,
        start_year: int,
        end_year: int
) -> pd.DataFrame:
    """
    Download central bank meeting minutes and policy announcements.

    Note: This requires web scraping central bank websites.
    Implementation is bank-specific and should respect robots.txt.

    Args:
        bank: Central bank name ("fed", "ecb", "boe", "boj")
        start_year: Start year (e.g., 2020)
        end_year: End year (e.g., 2024)

    Returns:
        DataFrame with columns: [date, title, full_text, bank_name, event_type]
    """
    logger.warning(f"[{bank}] Central bank minutes download not implemented")
    logger.info(f"[{bank}] Returning empty DataFrame - implement scraper for {bank.upper()} website")

    # Return empty DataFrame with correct schema
    return pd.DataFrame(columns=[
        'date', 'title', 'full_text', 'bank_name', 'event_type'
    ])


def combine_economic_data(
        dataframes: list[pd.DataFrame],
        deduplicate: bool = True
) -> pd.DataFrame:
    """
    Combine multiple economic data sources into single DataFrame for Cognee ingestion.

    Args:
        dataframes: List of DataFrames from different sources
        deduplicate: Whether to remove duplicate dates

    Returns:
        Combined DataFrame with standardized schema:
        [date, title, full_text, bank_name, event_type]
    """
    logger.info(f"[combine] Combining {len(dataframes)} economic data sources")

    # Filter out empty DataFrames
    valid_dfs = [df for df in dataframes if len(df) > 0]

    if not valid_dfs:
        logger.warning("[combine] No data to combine - all sources returned empty")
        return pd.DataFrame(columns=['date', 'title', 'full_text', 'bank_name', 'event_type'])

    # Concatenate
    combined = pd.concat(valid_dfs, ignore_index=True)

    # Ensure date column is datetime
    combined['date'] = pd.to_datetime(combined['date'])

    # Sort by date
    combined = combined.sort_values('date').reset_index(drop=True)

    # Deduplicate if requested
    if deduplicate:
        initial_len = len(combined)
        combined = combined.drop_duplicates(subset=['date', 'title']).reset_index(drop=True)
        if len(combined) < initial_len:
            logger.info(f"[combine] Removed {initial_len - len(combined)} duplicates")

    logger.info(
        f"[combine] Combined dataset: {len(combined)} records from {combined['date'].min()} to {combined['date'].max()}")

    return combined


# Preset series for FX/crypto trading
FOREX_FRED_SERIES = {
    # US Indicators
    "FEDFUNDS": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (CPI)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "PAYEMS": "Non-Farm Payrolls",

    # Eurozone Indicators (some available on FRED)
    "ECBDFR": "ECB Deposit Facility Rate",

    # UK Indicators
    "GBRCPIALLMINMEI": "UK CPI",

    # Currency Indices
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index",
    "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
    "DEXUSUK": "U.S. / U.K. Foreign Exchange Rate",
}


def download_forex_fred_bundle(
        start_date: str,
        end_date: str,
        api_key: str | None = None,
        series_ids: list[str] | None = None
) -> pd.DataFrame:
    """
    Download bundle of FX-relevant FRED series.

    Args:
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"
        api_key: FRED API key
        series_ids: Optional list of series IDs to download
                   (defaults to FOREX_FRED_SERIES keys)

    Returns:
        Combined DataFrame with all series
    """
    if series_ids is None:
        series_ids = list(FOREX_FRED_SERIES.keys())

    logger.info(f"[fred bundle] Downloading {len(series_ids)} FRED series")

    dataframes = []

    for series_id in series_ids:
        try:
            df = download_fred_series(series_id, start_date, end_date, api_key)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"[fred bundle] Failed to download {series_id}: {e}")
            continue

    return combine_economic_data(dataframes)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Download economic indicators")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--fred-api-key", help="FRED API key (or set FRED_API_KEY env var)")
    parser.add_argument("--series", nargs="+", help="FRED series IDs to download")
    parser.add_argument("--output", type=Path, help="Output CSV file path")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ECONOMIC INDICATORS DOWNLOADER")
    logger.info("=" * 80)
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("=" * 80)

    api_key = args.fred_api_key or os.getenv("FRED_API_KEY")

    if not api_key:
        logger.error("FRED API key required. Set FRED_API_KEY environment variable or use --fred-api-key")
        logger.info("Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    # Download data
    if args.series:
        # Download specific series
        dataframes = []
        for series_id in args.series:
            try:
                df = download_fred_series(series_id, args.start_date, args.end_date, api_key)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to download {series_id}: {e}")

        combined = combine_economic_data(dataframes)
    else:
        # Download default FX bundle
        combined = download_forex_fred_bundle(args.start_date, args.end_date, api_key)

    # Show results
    logger.info(f"\nDownloaded {len(combined)} total records")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    if len(combined) > 0:
        logger.info("\nSample records:")
        print(combined[['date', 'title', 'full_text']].head(10))

    # Save if requested
    if args.output:
        combined.to_csv(args.output, index=False)
        logger.info(f"\nSaved to: {args.output}")
