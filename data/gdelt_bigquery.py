"""
GDELT BigQuery Integration for Sentiment Analysis

**STATUS: EXPERIMENTAL - Limited production use**
This module is experimental and only used in prebuild_cognee_graph.py.
For production GDELT data, use data/downloaders/gdelt.py instead.

This module queries GDELT GKG (Global Knowledge Graph) data from Google BigQuery
to extract sentiment scores for financial news. Designed for use in Google Colab
where BigQuery access is free for public datasets.

Features:
- ✅ Query GDELT data by date range (avoids downloading entire archives)
- ✅ Filter by themes (ECON_CURRENCY, TAX_FNCACT, etc.)
- ✅ Extract sentiment from V2Tone field
- ✅ Intelligent caching to avoid redundant queries
- ✅ Compatible with existing sentiment aggregation pipeline

BigQuery Dataset: `gdelt-bq.gdeltv2.gkg`
Free Tier: 1TB/month of queries on public datasets

Example:
    # In Colab
    from google.colab import auth
    auth.authenticate_user()

    from data.gdelt_bigquery import query_gdelt_sentiment

    df = query_gdelt_sentiment(
        start_date="2023-01-01",
        end_date="2023-12-31",
        themes=["ECON_CURRENCY", "TAX_FNCACT"]
    )
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from utils.logger import get_logger

logger = get_logger(__name__)


def _get_bigquery_client():
    """Get authenticated BigQuery client.

    In Colab, authentication is handled automatically after calling:
        from google.colab import auth
        auth.authenticate_user()

    Returns:
        BigQuery Client instance
    """
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project='gdelt-bq')
        return client
    except Exception as e:
        logger.error(f"Failed to create BigQuery client: {e}")
        logger.info("In Colab, run: from google.colab import auth; auth.authenticate_user()")
        raise


def _parse_v2tone(tone_string: str) -> float:
    """Parse GDELT V2Tone field to extract average sentiment.

    V2Tone format: "tone,positive%,negative%,polarity,activity_density,self_density"
    Example: "-2.5,10.5,13.0,0.5,1.2,0.8"

    Args:
        tone_string: Comma-separated V2Tone field from GDELT

    Returns:
        Average tone (sentiment score), or 0.0 if parsing fails
    """
    if not tone_string or pd.isna(tone_string):
        return 0.0

    try:
        parts = tone_string.split(',')
        if len(parts) >= 1:
            return float(parts[0])  # First field is average tone
    except (ValueError, IndexError):
        # Invalid format - return default
        return 0.0


def query_gdelt_sentiment(
        start_date: str,
        end_date: str,
        themes: list[str] | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        limit: int | None = None
) -> pd.DataFrame:
    """Query GDELT GKG data from BigQuery and extract sentiment scores.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        themes: Optional list of GDELT themes to filter by
                (e.g., ['ECON_CURRENCY', 'TAX_FNCACT', 'ECON_FINANCIAL_CRISIS'])
        cache_dir: Directory for caching query results (default: data/gdelt_cache)
        use_cache: Whether to use cached results if available
        limit: Optional limit on number of rows (for testing)

    Returns:
        DataFrame with columns: ['datetime', 'sentiment_score', 'source_url', 'themes']

    Example:
        >>> df = query_gdelt_sentiment(
        ...     start_date="2023-01-01",
        ...     end_date="2023-01-31",
        ...     themes=["ECON_CURRENCY"]
        ... )
        >>> print(df.head())
    """
    # Setup cache directory
    if cache_dir is None:
        cache_dir = ROOT / "data" / "gdelt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache key based on query parameters
    cache_key_input = f"{start_date}_{end_date}_{themes}_{limit}".encode()
    cache_hash = hashlib.sha256(cache_key_input).hexdigest()[:12]
    cache_file = cache_dir / f"gdelt_{cache_hash}.feather"

    # Check cache
    if use_cache and cache_file.exists():
        logger.info(f"[cache] Loading GDELT data from cache: {cache_file}")
        return pd.read_feather(cache_file)

    logger.info(f"[bigquery] Querying GDELT GKG data: {start_date} to {end_date}")

    # Build BigQuery SQL
    query = f"""
    SELECT
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS datetime,
        V2Tone,
        DocumentIdentifier AS source_url,
        V2Themes AS themes
    FROM
        `gdelt-bq.gdeltv2.gkg`
    WHERE
        DATE >= {start_date.replace('-', '')}
        AND DATE <= {end_date.replace('-', '')}
    """

    # Add theme filtering if specified
    if themes:
        theme_conditions = " OR ".join([f"V2Themes LIKE '%{theme}%'" for theme in themes])
        query += f"\n    AND ({theme_conditions})"

    # Add limit if specified (useful for testing)
    if limit:
        query += f"\n    LIMIT {limit}"

    logger.info(f"[bigquery] Query: {query[:200]}...")

    # Execute query
    client = _get_bigquery_client()

    try:
        # Run query and convert to DataFrame
        query_job = client.query(query)
        df = query_job.to_dataframe()

        logger.info(f"[bigquery] Retrieved {len(df):,} rows from GDELT")

        # Parse sentiment from V2Tone
        df['sentiment_score'] = df['V2Tone'].apply(_parse_v2tone)

        # Keep only relevant columns
        df = df[['datetime', 'sentiment_score', 'source_url', 'themes']]

        # Convert datetime to proper timezone-aware format
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

        # Remove rows with null sentiment
        initial_len = len(df)
        df = df.dropna(subset=['sentiment_score'])
        if len(df) < initial_len:
            logger.info(f"[bigquery] Dropped {initial_len - len(df)} rows with null sentiment")

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Cache results
        if use_cache:
            df.to_feather(cache_file)
            logger.info(f"[cache] Saved GDELT data to cache: {cache_file}")

        logger.info(f"[bigquery] Final dataset: {len(df):,} rows with sentiment scores")

        return df

    except Exception as e:
        logger.error(f"[bigquery] Query failed: {e}")
        raise


def query_gdelt_for_date_range(
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
        themes: list[str] | None = None,
        cache_dir: Path | None = None
) -> pd.DataFrame:
    """Query GDELT data matching a specific datetime range (for FX/crypto data alignment).

    This is a convenience wrapper around query_gdelt_sentiment that accepts
    pandas Timestamps instead of string dates.

    Args:
        start_datetime: Start timestamp (pandas Timestamp)
        end_datetime: End timestamp (pandas Timestamp)
        themes: Optional list of GDELT themes to filter by
        cache_dir: Directory for caching query results

    Returns:
        DataFrame with GDELT sentiment data
    """
    start_date = start_datetime.strftime('%Y-%m-%d')
    end_date = end_datetime.strftime('%Y-%m-%d')

    return query_gdelt_sentiment(
        start_date=start_date,
        end_date=end_date,
        themes=themes,
        cache_dir=cache_dir
    )


# Default themes for financial/currency analysis
FINANCIAL_THEMES = [
    "ECON_CURRENCY",  # Currency and exchange rates
    "TAX_FNCACT",  # Financial activities
    "ECON_FINANCIAL_CRISIS",  # Financial crises
    "ECON_INFLATION",  # Inflation
    "ECON_INTERESTRATE",  # Interest rates
    "ECON_STOCKMARKET",  # Stock market
    "ECON_TRADE",  # International trade
]


def query_financial_sentiment(
        start_date: str,
        end_date: str,
        cache_dir: Path | None = None
) -> pd.DataFrame:
    """Query GDELT data filtered for financial/currency themes.

    This is a convenience function that queries GDELT with predefined
    financial themes relevant to FX/crypto trading.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Directory for caching query results

    Returns:
        DataFrame with GDELT sentiment data filtered for financial themes
    """
    return query_gdelt_sentiment(
        start_date=start_date,
        end_date=end_date,
        themes=FINANCIAL_THEMES,
        cache_dir=cache_dir
    )


if __name__ == "__main__":
    # Example usage / testing
    import argparse

    parser = argparse.ArgumentParser(description="Query GDELT sentiment data from BigQuery")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--themes", default=None, help="Comma-separated list of themes")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows (for testing)")
    parser.add_argument("--output", default=None, help="Output CSV file path")

    args = parser.parse_args()

    themes = args.themes.split(',') if args.themes else FINANCIAL_THEMES

    logger.info("=" * 80)
    logger.info("GDELT BIGQUERY SENTIMENT EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Themes: {themes}")
    logger.info("=" * 80)

    df = query_gdelt_sentiment(
        start_date=args.start_date,
        end_date=args.end_date,
        themes=themes,
        limit=args.limit
    )

    logger.info(f"\nRetrieved {len(df):,} sentiment records")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Sentiment range: {df['sentiment_score'].min():.2f} to {df['sentiment_score'].max():.2f}")
    logger.info(f"Mean sentiment: {df['sentiment_score'].mean():.2f}")

    if args.output:
        df.to_csv(args.output, index=False)
        logger.info(f"\nSaved to: {args.output}")

    # Show sample
    logger.info("\nSample rows:")
    print(df.head(10))
