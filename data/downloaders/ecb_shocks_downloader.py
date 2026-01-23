"""
ECB Monetary Policy Shocks Downloader

Loads and processes ECB monetary policy and central bank information shocks
from Jarocinski & Karadi (2020). These shocks are valuable for forex trading
as they capture surprise changes in monetary policy sentiment.

Data Source:
- Jarocinski, M. and Karadi, P. (2020)
  "Deconstructing Monetary Policy Surprises - The Role of Information Shocks"
  AEJ:Macro, https://doi.org/10.1257/mac.20180090
- Updated ECB monetary policy shocks dataset

Dependencies:
    No external dependencies required - data is in CSV format

Usage:
    from data.downloaders.ecb_shocks_downloader import (
        load_ecb_shocks_daily,
        load_ecb_shocks_monthly,
        get_monetary_policy_events
    )

    # Load daily shocks
    daily_shocks = load_ecb_shocks_daily()

    # Load monthly aggregated shocks
    monthly_shocks = load_ecb_shocks_monthly()
"""

import os
from functools import cache
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_project_root() -> Path:
    """
    Resolve the project root directory robustly.
    
    Strategy:
    1. Check PROJECT_ROOT environment variable
    2. Walk up from this file looking for markers (.git, pyproject.toml, setup.py)
    3. Fallback to parent directories
    4. Raise clear error if not found
    """
    # Strategy 1: Environment variable
    if env_root := os.getenv("PROJECT_ROOT"):
        root = Path(env_root)
        if root.is_dir():
            return root

    # Strategy 2: Look for markers (.git, pyproject.toml, setup.py)
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Limit search depth
        if any((current / marker).exists() for marker in [".git", "pyproject.toml", "setup.py"]):
            return current
        current = current.parent

    # Strategy 3: Raise clear error
    raise RuntimeError(
        "Could not determine project root. Set PROJECT_ROOT environment variable or "
        "ensure .git, pyproject.toml, or setup.py exists in project hierarchy."
    )


# Lazy path resolution to avoid import-time failures
@cache
def get_ecb_shocks_dir() -> Path:
    """Get vendored ECB shocks data directory."""
    root = _resolve_project_root()
    return root / "data" / "vendors" / "ecb_shocks" / "data"

@cache
def get_daily_shocks_file() -> Path:
    """Get path to daily ECB shocks CSV file."""
    return get_ecb_shocks_dir() / "shocks_ecb_mpd_me_d.csv"

@cache
def get_monthly_shocks_file() -> Path:
    """Get path to monthly ECB shocks CSV file."""
    return get_ecb_shocks_dir() / "shocks_ecb_mpd_me_m.csv"


def _load_ecb_shocks(file_path: Path, date_col_index: int = 0) -> pd.DataFrame:
    """
    Helper function to load and process ECB shocks from CSV.

    Args:
        file_path: Path to the shocks CSV file
        date_col_index: Index of the date column (default: 0)

    Returns:
        DataFrame with parsed dates and numeric shock columns
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"ECB shocks file not found: {file_path}. "
            f"Ensure the file exists at the specified path."
        )

    logger.info(f"[ecb_shocks] Loading shocks from {file_path}")

    try:
        df = pd.read_csv(file_path)

        # Validate date_col_index before using it
        if not isinstance(date_col_index, int):
            raise TypeError(
                f"date_col_index must be an integer, got {type(date_col_index).__name__}"
            )
        if date_col_index < 0 or date_col_index >= len(df.columns):
            raise ValueError(
                f"date_col_index {date_col_index} out of range for "
                f"DataFrame with {len(df.columns)} columns"
            )

        # Parse date column
        date_col = df.columns[date_col_index]

        # Handle pre-existing 'date' column if different from source column
        if 'date' in df.columns and date_col != 'date':
            # Drop the pre-existing date column before assignment to avoid duplication
            df = df.drop(columns=['date'])

        # Parse into df['date']; drop original only if differently named
        df['date'] = pd.to_datetime(df[date_col])
        if date_col != 'date':
            df = df.drop(columns=[date_col])

        # Ensure numeric columns
        numeric_cols = ['pc1', 'STOXX50', 'MP_pm', 'CBI_pm', 'MP_median', 'CBI_median']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort and reset index
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"[ecb_shocks] Loaded {len(df)} shock observations")
        return df

    except Exception as e:
        logger.error(f"[ecb_shocks] Error loading shocks: {e}")
        raise


def load_ecb_shocks_daily() -> pd.DataFrame:
    """
    Load daily ECB monetary policy shocks (by Governing Council meeting).

    Returns:
        DataFrame with columns:
        - date: Meeting date
        - pc1: Surprise in policy indicator (1st principal component of OIS changes)
        - STOXX50: Euro Stoxx 50 changes during event window
        - MP_pm: Monetary Policy shock (Poor Man's sign restrictions)
        - CBI_pm: Central Bank Information shock (Poor Man's sign restrictions)
        - MP_median: Monetary Policy shock (median rotation)
        - CBI_median: Central Bank Information shock (median rotation)

    Example:
        >>> df = load_ecb_shocks_daily()
        >>> print(df.head())
                date      pc1  STOXX50   MP_pm  CBI_pm  MP_median  CBI_median
        0 1999-01-04  0.12345  0.23456  0.1000  0.0234     0.0950      0.0284
    """
    return _load_ecb_shocks(get_daily_shocks_file())


def load_ecb_shocks_monthly() -> pd.DataFrame:
    """
    Load monthly aggregated ECB monetary policy shocks.
    Zero values indicate no shocks in that month.

    Returns:
        DataFrame with same columns as daily data, aggregated to monthly frequency

    Example:
        >>> df = load_ecb_shocks_monthly()
        >>> print(df.head())
                date      pc1  STOXX50   MP_pm  CBI_pm  MP_median  CBI_median
        0 1999-01-01  0.12345  0.23456  0.1000  0.0234     0.0950      0.0284
    """
    return _load_ecb_shocks(get_monthly_shocks_file())


def get_monetary_policy_events(
        start_date: str | None = None,
        end_date: str | None = None,
        frequency: str = "daily",
        shock_threshold: float | None = None
) -> pd.DataFrame:
    """
    Get ECB monetary policy events within a date range, optionally filtered by shock magnitude.

    Args:
        start_date: Start date in "YYYY-MM-DD" format (optional)
        end_date: End date in "YYYY-MM-DD" format (optional)
        frequency: "daily" or "monthly"
        shock_threshold: Optional minimum absolute shock magnitude to include
                        (filters on MP_median absolute value)

    Returns:
        DataFrame with ECB monetary policy shocks

    Example:
        >>> # Get significant policy shocks in 2023
        >>> events = get_monetary_policy_events(
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     shock_threshold=0.05
        ... )
    """
    # Load appropriate frequency
    if frequency.lower() == "daily":
        df = load_ecb_shocks_daily()
    elif frequency.lower() == "monthly":
        df = load_ecb_shocks_monthly()
    else:
        raise ValueError(f"Invalid frequency: {frequency}. Use 'daily' or 'monthly'")

    # Filter by date range
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # Filter by shock magnitude
    if shock_threshold is not None:
        if 'MP_median' in df.columns:
            df = df[abs(df['MP_median']) >= shock_threshold]
        else:
            logger.warning("[ecb_shocks] MP_median column not found, cannot filter by threshold")

    # Reset index after filtering
    df = df.reset_index(drop=True)

    logger.info(f"[ecb_shocks] Retrieved {len(df)} monetary policy events")
    return df


def classify_shock_type(row: pd.Series) -> str:
    """
    Classify the dominant shock type for a given observation.

    Args:
        row: DataFrame row with MP_median and CBI_median columns

    Returns:
        Shock classification: one of "hawkish_MP", "dovish_MP", "neutral_MP",
        "positive_CBI", "negative_CBI", or "neutral_CBI"
    """
    mp = row.get('MP_median', 0)
    cbi = row.get('CBI_median', 0)

    # Thresholds
    threshold = 0.02

    if abs(mp) > abs(cbi):
        # Monetary Policy dominant
        return "hawkish_MP" if mp > threshold else "dovish_MP" if mp < -threshold else "neutral_MP"
    else:
        # Central Bank Information dominant
        return "positive_CBI" if cbi > threshold else "negative_CBI" if cbi < -threshold else "neutral_CBI"


def format_for_cognee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format ECB shocks data for Cognee ingestion with descriptive text.

    Args:
        df: Raw ECB shocks DataFrame

    Returns:
        DataFrame with 'full_text' column for semantic search
    """
    df = df.copy()

    # Add shock classification
    df['shock_type'] = df.apply(classify_shock_type, axis=1)

    def _format_row(row):
        """Format a single row with safe date handling."""
        date_str = "unknown date"
        if pd.notna(row.get('date')):
            try:
                date_str = row['date'].strftime('%Y-%m-%d')
            except (AttributeError, ValueError):
                pass

        return (
            f"ECB Monetary Policy Event on {date_str}: "
            f"Policy indicator surprise (pc1): {row.get('pc1', 0):.4f}, "
            f"Monetary Policy shock: {row.get('MP_median', 0):.4f}, "
            f"Central Bank Information shock: {row.get('CBI_median', 0):.4f}. "
            f"Classification: {row.get('shock_type', 'unknown')}. "
            f"Euro Stoxx 50 change: {row.get('STOXX50', 0):.4f}."
        )

    df['full_text'] = df.apply(_format_row, axis=1)

    return df


def get_shocks_for_forex_pair(
        currency_pair: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily"
) -> pd.DataFrame:
    """
    Get ECB monetary policy shocks relevant to a forex pair involving EUR.

    Args:
        currency_pair: Forex pair (must contain EUR, e.g., "EURUSD", "EURGBP")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format
        frequency: "daily" or "monthly"

    Returns:
        DataFrame with ECB shocks and currency pair context

    Example:
        >>> df = get_shocks_for_forex_pair("EURUSD", "2023-01-01", "2023-12-31")
    """
    pair_upper = currency_pair.upper()

    if "EUR" not in pair_upper:
        logger.warning(f"[ecb_shocks] Currency pair {currency_pair} does not contain EUR")
        return pd.DataFrame()

    # Load shocks
    df = get_monetary_policy_events(start_date, end_date, frequency)

    if df.empty:
        return df

    # Add currency pair context
    df['currency_pair'] = pair_upper
    df['base_currency'] = pair_upper[:3]
    df['quote_currency'] = pair_upper[3:]

    logger.info(f"[ecb_shocks] Retrieved {len(df)} ECB shocks for {currency_pair}")
    return df
