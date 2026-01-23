"""
UN Comtrade International Trade Data Downloader

Downloads international trade statistics to analyze cross-border trade flows
that may impact forex markets. Trade balance data is particularly relevant
for currency valuation and economic sentiment.

Data Source:
- UN Comtrade API: https://comtrade.un.org
- Trade flows, imports/exports, trade balance by country/commodity

Dependencies:
    Install from local package:
    pip install -e ./new_data_sources/comtradeapicall

Usage:
    from data.downloaders.comtrade_downloader import (
        download_trade_balance,
        get_trade_indicators_for_forex
    )

    # Get trade balance for EUR/USD relevant countries
    trade_data = get_trade_indicators_for_forex(
        currency_pair="EURUSD",
        start_year=2023,
        end_year=2023,
        subscription_key=os.getenv("COMTRADE_API_KEY")
    )
"""

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

# Country mapping for major forex pairs
FOREX_COUNTRY_MAPPING = {
    "EURUSD": {"base": ["276"], "quote": ["842"]},  # Germany (EU proxy), USA
    "GBPUSD": {"base": ["826"], "quote": ["842"]},  # UK, USA
    "USDJPY": {"base": ["842"], "quote": ["392"]},  # USA, Japan
    "AUDUSD": {"base": ["036"], "quote": ["842"]},  # Australia, USA
    "EURJPY": {"base": ["276"], "quote": ["392"]},  # Germany (EU proxy), Japan
    "EURGBP": {"base": ["276"], "quote": ["826"]},  # Germany (EU proxy), UK
    "USDCAD": {"base": ["842"], "quote": ["124"]},  # USA, Canada
    "USDCHF": {"base": ["842"], "quote": ["756"]},  # USA, Switzerland
}


def download_trade_balance(
    reporter_code: str, start_year: int, end_year: int, subscription_key: str | None = None
) -> pd.DataFrame:
    """
    Download trade balance data for a country.

    Args:
        reporter_code: UN Comtrade country code (e.g., "842" for USA)
        start_year: Start year (e.g., 2020)
        end_year: End year (e.g., 2023)
        subscription_key: UN Comtrade API subscription key (required for bulk downloads)

    Returns:
        DataFrame with columns: [date, trade_balance, imports, exports, country_code]

    Example:
        >>> df = download_trade_balance("842", 2023, 2023, subscription_key)
        >>> print(df.head())
                date  trade_balance  imports    exports country_code
        0 2023-01-01      -50000000  150000000  100000000         842
    """
    try:
        from comtradeapicall import previewGet
    except ImportError:
        raise ImportError(
            "comtradeapicall not installed. Install with: "
            "pip install -e ./new_data_sources/comtradeapicall"
        )

    logger.info(
        f"[comtrade] Downloading trade balance for country {reporter_code}, {start_year}-{end_year}"
    )

    try:
        # Build selection criteria
        # For monthly data, period must be in YYYYMM format
        periods = [
            f"{year}{month:02d}"
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        ]

        selection_criteria = {
            "typeCode": "C",  # Commodities
            "freqCode": "M",  # Monthly
            "clCode": "HS",  # Harmonized System
            "period": periods,  # YYYYMM format for monthly data
            "reporterCode": reporter_code,
            "cmdCode": "TOTAL",  # All commodities
            "flowCode": ["M", "X"],  # Imports and Exports
            "partnerCode": None,  # All partners
            "partner2Code": None,
        }

        @api_retry(max_retries=3, base_delay=3.0, rate_limit_calls=0.3)  # Conservative for Comtrade
        def _fetch_trade_data(api_func, **kwargs):
            """Internal function for API call with retry logic."""
            return api_func(**kwargs)

        # Use preview if no subscription key, otherwise use full API
        if subscription_key:
            df = _fetch_trade_data(
                previewGet.getTradeBalance, subscription_key=subscription_key, **selection_criteria
            )
        else:
            logger.warning(
                "[comtrade] No subscription key provided, using preview (limited to 500 records)"
            )
            df = _fetch_trade_data(previewGet.previewTradeBalance, **selection_criteria)

        if df is None or df.empty:
            logger.warning(f"[comtrade] No data returned for country {reporter_code}")
            return pd.DataFrame()

        # Process and normalize
        df = df.copy()
        df["date"] = pd.to_datetime(df["period"], format="%Y%m")
        df["country_code"] = reporter_code

        # Calculate trade balance - validate columns exist
        if "trade_balance" not in df.columns:
            required_cols = {"exports", "imports"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                # Build missing columns list in deterministic order
                ordered_required = ["exports", "imports"]
                ordered_missing = [col for col in ordered_required if col in missing_cols]
                logger.error(
                    f"[comtrade] Missing required columns for trade balance calculation: {ordered_missing}"
                )
                raise ValueError(
                    f"Missing columns {ordered_missing} required to compute trade_balance"
                )

        # Select columns - include imports/exports per docstring
        # Build deterministically: date, trade_balance, optional (imports, exports), country_code
        columns_to_select = ["date", "trade_balance"]
        if "imports" in df.columns:
            columns_to_select.append("imports")
        if "exports" in df.columns:
            columns_to_select.append("exports")
        columns_to_select.append("country_code")

        result = df[columns_to_select].copy()

        logger.info(f"[comtrade] Downloaded {len(result)} trade balance records")
        return result

    except Exception as e:
        logger.error(f"[comtrade] Error downloading trade balance: {e}")
        raise


def get_trade_indicators_for_forex(
    currency_pair: str, start_year: int, end_year: int, subscription_key: str | None = None
) -> pd.DataFrame:
    """
    Download trade indicators relevant to a forex pair.

    Args:
        currency_pair: Forex pair symbol (e.g., "EURUSD", "GBPUSD")
        start_year: Start year
        end_year: End year
        subscription_key: UN Comtrade API subscription key

    Returns:
        DataFrame with trade balance data for both countries in the pair

    Example:
        >>> df = get_trade_indicators_for_forex("EURUSD", 2023, 2023)
        >>> print(df.head())
                date currency_pair country_type  trade_balance country_code
        0 2023-01-01       EURUSD         base       15000000          276
        1 2023-02-01       EURUSD         base       12000000          276
    """
    pair_upper = currency_pair.upper()

    if pair_upper not in FOREX_COUNTRY_MAPPING:
        raise ValueError(
            f"Currency pair {currency_pair} not supported. "
            f"Supported pairs: {list(FOREX_COUNTRY_MAPPING.keys())}"
        )

    countries = FOREX_COUNTRY_MAPPING[pair_upper]
    all_data = []

    # Download for base currency country
    for base_country in countries["base"]:
        try:
            df_base = download_trade_balance(base_country, start_year, end_year, subscription_key)
            if not df_base.empty:
                df_base["currency_pair"] = pair_upper
                df_base["country_type"] = "base"
                all_data.append(df_base)
        except Exception as e:
            logger.warning(
                f"[comtrade] Failed to download data for base country {base_country} "
                f"in pair {pair_upper}: {e}. Continuing with other countries."
            )

    # Download for quote currency country
    for quote_country in countries["quote"]:
        try:
            df_quote = download_trade_balance(quote_country, start_year, end_year, subscription_key)
            if not df_quote.empty:
                df_quote["currency_pair"] = pair_upper
                df_quote["country_type"] = "quote"
                all_data.append(df_quote)
        except Exception as e:
            logger.warning(
                f"[comtrade] Failed to download data for quote country {quote_country} "
                f"in pair {pair_upper}: {e}. Continuing with other countries."
            )

    if not all_data:
        logger.warning(f"[comtrade] No data retrieved for {currency_pair}")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    logger.info(f"[comtrade] Retrieved {len(result)} trade indicator records for {currency_pair}")
    return result


def format_for_cognee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format Comtrade data for Cognee ingestion.

    Args:
        df: Raw Comtrade DataFrame

    Returns:
        DataFrame with 'full_text' column for semantic search
    """
    df = df.copy()

    # Validate required columns
    required_cols = ["date", "trade_balance"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Coerce date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    nat_count = df["date"].isna().sum()
    if nat_count > 0:
        logger.warning(f"[comtrade] {nat_count} rows have invalid dates, dropping them")
        df = df.dropna(subset=["date"])

    # Coerce trade_balance to numeric
    df["trade_balance"] = pd.to_numeric(df["trade_balance"], errors="coerce")
    nan_count = df["trade_balance"].isna().sum()
    if nan_count > 0:
        logger.warning(f"[comtrade] {nan_count} rows have invalid trade_balance, dropping them")
        df = df.dropna(subset=["trade_balance"])

    # Ensure optional columns with descriptive defaults
    if "country_type" not in df.columns:
        df["country_type"] = "unknown_country_type"
    if "currency_pair" not in df.columns:
        df["currency_pair"] = "unknown_currency_pair"
    if "country_code" not in df.columns:
        df["country_code"] = "unknown_country_code"

    # Vectorized string composition using pandas operations
    formatted_date = df["date"].dt.strftime("%Y-%m-%d")
    formatted_balance = df["trade_balance"].apply(lambda x: f"{x:,.0f}")
    
    df["full_text"] = (
        "Trade balance for " + df["country_type"] + " currency " +
        "in " + df["currency_pair"] + " on " + formatted_date + ": " +
        "$" + formatted_balance + ". " +
        "Country code: " + df["country_code"] + "."
    )

    return df
