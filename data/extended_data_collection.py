"""
Extended Data Collection Pipeline

Integrates new data sources into the existing pipeline:
1. UN Comtrade - International trade data
2. FRED - Enhanced Federal Reserve economic data
3. ECB Shocks - Monetary policy surprise indicators

This module provides a unified interface for collecting all fundamental
economic data sources to complement price and sentiment data.

Usage:
    from data.extended_data_collection import (
        collect_all_forex_fundamentals,
        collect_trade_data,
        collect_economic_data,
        collect_monetary_shocks
    )

    # Collect all fundamental data for a forex pair
    data = collect_all_forex_fundamentals(
        currency_pair="EURUSD",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def collect_trade_data(
        currency_pair: str,
        start_year: int,
        end_year: int,
        comtrade_api_key: str | None = None
) -> pd.DataFrame:
    """
    Collect international trade data for a forex pair.

    Args:
        currency_pair: Forex pair (e.g., "EURUSD")
        start_year: Start year
        end_year: End year
        comtrade_api_key: UN Comtrade API key (optional, uses preview without it)

    Returns:
        DataFrame with trade balance data
    """
    try:
        from data.downloaders.comtrade_downloader import (
            format_for_cognee,
            get_trade_indicators_for_forex,
        )

        logger.info(f"[extended] Collecting trade data for {currency_pair}")

        df = get_trade_indicators_for_forex(
            currency_pair=currency_pair,
            start_year=start_year,
            end_year=end_year,
            subscription_key=comtrade_api_key
        )

        if not df.empty:
            df = format_for_cognee(df)

        logger.info(f"[extended] Collected {len(df)} trade data records")
        return df

    except Exception as e:
        logger.error(f"[extended] Error collecting trade data: {e}")
        return pd.DataFrame()


def collect_economic_data(
        currency_pair: str,
        start_date: str,
        end_date: str,
        fred_api_key: str | None = None,
        indicators: list[str] | None = None
) -> pd.DataFrame:
    """
    Collect economic indicator data for a forex pair.

    Args:
        currency_pair: Forex pair (e.g., "EURUSD")
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"
        fred_api_key: FRED API key
        indicators: Optional list of specific indicators

    Returns:
        DataFrame with economic indicator data
    """
    try:
        from data.downloaders.fred_downloader import (
            format_for_cognee,
            get_forex_economic_indicators,
        )

        logger.info(f"[extended] Collecting economic data for {currency_pair}")

        df = get_forex_economic_indicators(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date,
            api_key=fred_api_key,
            indicators=indicators
        )

        if not df.empty:
            df = format_for_cognee(df)

        logger.info(f"[extended] Collected {len(df)} economic indicator records")
        return df

    except Exception as e:
        logger.error(f"[extended] Error collecting economic data: {e}")
        return pd.DataFrame()


def collect_monetary_shocks(
        currency_pair: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily"
) -> pd.DataFrame:
    """
    Collect ECB monetary policy shocks for EUR pairs.

    Args:
        currency_pair: Forex pair (must contain EUR, e.g., "EURUSD")
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"
        frequency: "daily" or "monthly"

    Returns:
        DataFrame with monetary policy shock data
    """
    try:
        from data.downloaders.ecb_shocks_downloader import (
            format_for_cognee,
            get_shocks_for_forex_pair,
        )

        if "EUR" not in currency_pair.upper():
            logger.info(f"[extended] Skipping ECB shocks for non-EUR pair {currency_pair}")
            return pd.DataFrame()

        logger.info(f"[extended] Collecting ECB monetary shocks for {currency_pair}")

        df = get_shocks_for_forex_pair(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )

        if not df.empty:
            df = format_for_cognee(df)

        logger.info(f"[extended] Collected {len(df)} ECB shock records")
        return df

    except Exception as e:
        logger.error(f"[extended] Error collecting monetary shocks: {e}")
        return pd.DataFrame()


def collect_all_forex_fundamentals(
        currency_pair: str,
        start_date: str,
        end_date: str,
        comtrade_api_key: str | None = None,
        fred_api_key: str | None = None,
        include_sources: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    Collect all fundamental data sources for a forex pair.

    Args:
        currency_pair: Forex pair (e.g., "EURUSD", "GBPUSD")
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"
        comtrade_api_key: UN Comtrade API key (optional)
        fred_api_key: FRED API key
        include_sources: Optional list to filter sources
                        ["trade", "economic", "shocks"]
                        If None, collects all sources

    Returns:
        Dictionary with keys: "trade", "economic", "shocks"
        Each containing a DataFrame with the respective data

    Example:
        >>> data = collect_all_forex_fundamentals(
        ...     currency_pair="EURUSD",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>> print(f"Trade records: {len(data['trade'])}")
        >>> print(f"Economic records: {len(data['economic'])}")
        >>> print(f"Shock records: {len(data['shocks'])}")
    """
    logger.info(f"[extended] Collecting all fundamental data for {currency_pair}")

    # Default to all sources
    if include_sources is None:
        include_sources = ["trade", "economic", "shocks"]

    result = {}

    # Extract years from dates with robust parsing
    try:
        start_year = datetime.strptime(start_date, "%Y-%m-%d").year
        end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    except ValueError:
        logger.error("[extended] Invalid date format for start_date or end_date. Expected YYYY-MM-DD")
        raise ValueError("Dates must be in YYYY-MM-DD format.")

    # Collect trade data
    if "trade" in include_sources:
        result["trade"] = collect_trade_data(
            currency_pair=currency_pair,
            start_year=start_year,
            end_year=end_year,
            comtrade_api_key=comtrade_api_key
        )

    # Collect economic indicators
    if "economic" in include_sources:
        result["economic"] = collect_economic_data(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date,
            fred_api_key=fred_api_key
        )

    # Collect monetary policy shocks
    if "shocks" in include_sources:
        result["shocks"] = collect_monetary_shocks(
            currency_pair=currency_pair,
            start_date=start_date,
            end_date=end_date,
            frequency="daily"
        )

    # Log summary
    total_records = sum(len(df) for df in result.values())
    logger.info(f"[extended] Collected {total_records} total fundamental records:")
    for source, df in result.items():
        logger.info(f"  - {source}: {len(df)} records")

    return result


def save_fundamental_data(
        data: dict[str, pd.DataFrame],
        output_dir: Path | str,
        currency_pair: str,
        file_format: str = "parquet"
) -> dict[str, Path]:
    """
    Save fundamental data to disk.

    Args:
        data: Dictionary of DataFrames from collect_all_forex_fundamentals
        output_dir: Directory to save files
        currency_pair: Forex pair name for file naming
        file_format: "parquet" or "csv"

    Returns:
        Dictionary mapping source names to saved file paths

    Example:
        >>> data = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")
        >>> paths = save_fundamental_data(data, "data/fundamentals", "EURUSD")
        >>> print(paths)
        {'trade': Path('data/fundamentals/EURUSD_trade.parquet'), ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for source, df in data.items():
        if df.empty:
            logger.warning(f"[extended] Skipping empty {source} data")
            continue

        filename = f"{currency_pair}_{source}.{file_format}"
        filepath = output_dir / filename

        if file_format == "parquet":
            df.to_parquet(filepath, index=False)
        elif file_format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        saved_paths[source] = filepath
        logger.info(f"[extended] Saved {len(df)} {source} records to {filepath}")

    return saved_paths


def merge_with_price_data(
        fundamentals: dict[str, pd.DataFrame],
        price_df: pd.DataFrame,
        date_column: str = "datetime",
        resample_freq: str = "1h"
) -> pd.DataFrame:
    """
    Merge fundamental data with price data for training.

    Args:
        fundamentals: Dictionary of fundamental DataFrames
        price_df: Price DataFrame with datetime column
        date_column: Name of the datetime column in price_df (or any common timestamp column)
        resample_freq: Resample frequency for fundamental data (e.g., '1h', '1D')

    Returns:
        Merged DataFrame with all data sources aligned by date

    Example:
        >>> fundamentals = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")
        >>> price_df = pd.read_parquet("data/prepared/EURUSD_1h.parquet")
        >>> merged = merge_with_price_data(fundamentals, price_df, resample_freq='1h')
    """
    logger.info("[extended] Merging fundamental data with price data")

    # Detect timestamp column if not found
    possible_date_cols = [date_column, 'date', 'timestamp', 'period']
    timestamp_col = None
    for col in possible_date_cols:
        if col in price_df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(
            f"No timestamp column found. Tried: {possible_date_cols}. "
            f"Available columns: {list(price_df.columns)}"
        )

    # Ensure price data has datetime index
    price_df = price_df.copy()
    price_df[timestamp_col] = pd.to_datetime(price_df[timestamp_col])
    price_df = price_df.set_index(timestamp_col)

    # Start with price data
    merged = price_df

    # Merge each fundamental source
    for source, df in fundamentals.items():
        if df.empty:
            continue

        # Ensure date column is datetime
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Forward fill fundamental data to match price frequency
        df_resampled = df.resample(resample_freq).ffill()

        # Add source prefix to columns
        df_resampled = df_resampled.add_prefix(f"{source}_")

        # Merge with price data
        merged = merged.join(df_resampled, how='left')

    logger.info(f"[extended] Merged dataset has {len(merged)} rows and {len(merged.columns)} columns")
    return merged
