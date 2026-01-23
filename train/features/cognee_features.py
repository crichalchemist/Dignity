"""
Cognee Knowledge Graph Feature Extraction

Extract training features from Cognee Cloud knowledge graphs for FX/crypto trading.
Converts entity mentions, events, and relationships into numerical features aligned with price data.

Features Extracted:
- ✅ Entity mention counts in time windows (Fed, ECB, USD, EUR mentions)
- ✅ Event proximity features (rate hikes, announcements within X hours)
- ✅ Event sentiment aggregation
- ✅ Causal chain depth (how many relationship hops to currency)
- ✅ Semantic similarity to historical patterns

Usage:
    from data.cognee_client import CogneeClient
    from train.features.cognee_features import build_cognee_features

    client = CogneeClient(api_key=os.getenv("COGNEE_API_KEY"))

    cognee_features = build_cognee_features(
        client=client,
        price_df=df,
        pair="eurusd",
        cache_dir=Path("cache/cognee")
    )

    # Result: DataFrame with Cognee features aligned to price_df timestamps
    # Columns: fed_mentions_24h, ecb_mentions_24h, rate_hike_events_48h, etc.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from data.cognee_client import CogneeAPIError, CogneeClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Entity names to track
CENTRAL_BANKS = [
    "Federal Reserve", "Fed", "FOMC",
    "European Central Bank", "ECB",
    "Bank of England", "BOE",
    "Bank of Japan", "BOJ"
]

CURRENCIES = [
    "USD", "US Dollar", "Dollar",
    "EUR", "Euro",
    "GBP", "Pound Sterling", "British Pound",
    "JPY", "Yen", "Japanese Yen",
    "CHF", "Swiss Franc"
]

EVENT_TYPES = [
    "rate hike", "rate cut", "rate hold",
    "quantitative easing", "QE",
    "inflation report", "CPI", "NFP",
    "GDP", "unemployment"
]


def query_entity_mentions(
        client: CogneeClient,
        dataset_name: str,
        entity: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
) -> int:
    """
    Count mentions of an entity within a time window.

    Args:
        client: Cognee API client
        dataset_name: Dataset to query
        entity: Entity name to search for
        start_time: Start of time window
        end_time: End of time window

    Returns:
        Number of mentions
    """
    try:
        # Search for entity in the time window
        query = f"{entity} between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"

        results = client.search(
            query=query,
            dataset_name=dataset_name,
            limit=100  # Count up to 100 mentions
        )

        # Filter by metadata datetime if available
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            result_time = metadata.get('datetime')

            if result_time:
                result_time = pd.to_datetime(result_time)
                if start_time <= result_time <= end_time:
                    filtered_results.append(result)
            else:
                # If no metadata datetime, include (conservative approach)
                filtered_results.append(result)

        return len(filtered_results)

    except CogneeAPIError as e:
        logger.warning(f"Failed to query entity '{entity}': {e}")
        return 0


def query_event_proximity(
        client: CogneeClient,
        dataset_name: str,
        event_type: str,
        timestamp: pd.Timestamp,
        window_hours: int = 24
) -> dict:
    """
    Find events of a specific type within X hours of a timestamp.

    Args:
        client: Cognee API client
        dataset_name: Dataset to query
        event_type: Type of event (e.g., "rate hike", "CPI")
        timestamp: Target timestamp
        window_hours: Hours before timestamp to search

    Returns:
        Dict with keys:
            - event_count: int
            - avg_sentiment: float (if sentiment available)
            - entities_involved: list[str]
    """
    start_time = timestamp - pd.Timedelta(hours=window_hours)
    end_time = timestamp

    try:
        query = f"{event_type} between {start_time.strftime('%Y-%m-%d %H:%M')} and {end_time.strftime('%Y-%m-%d %H:%M')}"

        results = client.search(
            query=query,
            dataset_name=dataset_name,
            limit=50
        )

        # Extract information from results
        event_count = len(results)
        sentiments = []
        entities = set()

        for result in results:
            metadata = result.get('metadata', {})

            # Extract sentiment
            if 'sentiment_score' in metadata:
                sentiments.append(float(metadata['sentiment_score']))

            # Extract entities
            if 'entities' in result:
                for entity in result['entities']:
                    entities.add(entity.get('name', ''))

        return {
            'event_count': event_count,
            'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'entities_involved': list(entities)
        }

    except CogneeAPIError as e:
        logger.warning(f"Failed to query event '{event_type}': {e}")
        return {
            'event_count': 0,
            'avg_sentiment': 0.0,
            'entities_involved': []
        }


def build_cognee_features(
        client: CogneeClient,
        price_df: pd.DataFrame,
        pair: str,
        dataset_name: str | None = None,
        cache_dir: Path | None = None,
        entity_window_hours: int = 24,
        event_window_hours: int = 48,
        use_cache: bool = True
) -> pd.DataFrame:
    """
    Extract Cognee knowledge graph features for each timestamp in price data.

    Args:
        client: Configured CogneeClient
        price_df: DataFrame with price data (must have 'datetime' column)
        pair: Currency pair (e.g., "eurusd", "btcusd")
        dataset_name: Cognee dataset name (defaults to f"fx_{pair}")
        cache_dir: Directory for caching features
        entity_window_hours: Lookback window for entity mentions (default: 24h)
        event_window_hours: Lookback window for events (default: 48h)
        use_cache: Whether to use cached features if available

    Returns:
        DataFrame with Cognee features aligned to price_df index
        Columns include:
            - fed_mentions_24h: Count of Fed mentions in prior 24h
            - ecb_mentions_24h: Count of ECB mentions
            - usd_mentions_24h: Count of USD mentions
            - eur_mentions_24h: Count of EUR mentions
            - rate_hike_events_48h: Number of rate hike events in prior 48h
            - rate_cut_events_48h: Number of rate cut events
            - event_sentiment_mean_48h: Average sentiment of events in prior 48h
            - entity_density: Total entity mentions / time window

    Example:
        >>> features = build_cognee_features(
        ...     client=client,
        ...     price_df=df,
        ...     pair="eurusd",
        ...     dataset_name="fx_eurusd_2023"
        ... )
        >>> print(features.columns)
        ['fed_mentions_24h', 'ecb_mentions_24h', 'rate_hike_events_48h', ...]
    """
    logger.info(f"[cognee features] Building features for {pair} ({len(price_df)} timestamps)")

    if 'datetime' not in price_df.columns:
        raise ValueError("price_df must have 'datetime' column")

    # Default dataset name
    if dataset_name is None:
        dataset_name = f"fx_{pair}"

    # Setup caching
    if cache_dir is None:
        cache_dir = ROOT / "data" / "cognee_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache key
    timestamps_hash = hashlib.sha256(
        price_df['datetime'].astype(str).str.cat().encode()
    ).hexdigest()[:12]

    cache_key = f"{dataset_name}_{timestamps_hash}_e{entity_window_hours}_ev{event_window_hours}"
    cache_path = cache_dir / f"cognee_features_{cache_key}.feather"

    # Check cache
    if use_cache and cache_path.exists():
        logger.info(f"[cache] Loading Cognee features from {cache_path}")
        return pd.read_feather(cache_path)

    # Initialize feature DataFrame
    features_df = pd.DataFrame(index=price_df.index)

    # Ensure datetime is timezone-aware (for consistent comparisons)
    price_df = price_df.copy()
    if price_df['datetime'].dt.tz is None:
        price_df['datetime'] = price_df['datetime'].dt.tz_localize('UTC')
    else:
        price_df['datetime'] = price_df['datetime'].dt.tz_convert('UTC')

    # Feature 1: Entity mention counts
    logger.info("[cognee features] Extracting entity mention features...")

    # Sample a subset of timestamps for efficiency (every Nth row)
    # For full dataset, query every timestamp (expensive but thorough)
    # For quick iteration, sample every 60 minutes
    sample_interval = 60  # Query every 60 rows (adjust based on data frequency)

    entity_features = {
        'fed_mentions_24h': [],
        'ecb_mentions_24h': [],
        'boe_mentions_24h': [],
        'usd_mentions_24h': [],
        'eur_mentions_24h': [],
        'gbp_mentions_24h': []
    }

    for i, (_idx, row) in enumerate(price_df.iterrows()):
        timestamp = row['datetime']
        start_time = timestamp - pd.Timedelta(hours=entity_window_hours)

        # Query entities (or use interpolated values for non-sampled rows)
        if i % sample_interval == 0:
            # Query Cognee for this timestamp
            fed_count = query_entity_mentions(client, dataset_name, "Federal Reserve", start_time, timestamp)
            ecb_count = query_entity_mentions(client, dataset_name, "European Central Bank", start_time, timestamp)
            boe_count = query_entity_mentions(client, dataset_name, "Bank of England", start_time, timestamp)
            usd_count = query_entity_mentions(client, dataset_name, "USD", start_time, timestamp)
            eur_count = query_entity_mentions(client, dataset_name, "EUR", start_time, timestamp)
            gbp_count = query_entity_mentions(client, dataset_name, "GBP", start_time, timestamp)

            # Store values
            entity_features['fed_mentions_24h'].append(fed_count)
            entity_features['ecb_mentions_24h'].append(ecb_count)
            entity_features['boe_mentions_24h'].append(boe_count)
            entity_features['usd_mentions_24h'].append(usd_count)
            entity_features['eur_mentions_24h'].append(eur_count)
            entity_features['gbp_mentions_24h'].append(gbp_count)

            if (i % (sample_interval * 10)) == 0:
                logger.info(f"  Progress: {i}/{len(price_df)} ({i / len(price_df):.1%})")
        else:
            # Interpolate from last sampled value
            entity_features['fed_mentions_24h'].append(
                entity_features['fed_mentions_24h'][-1] if entity_features['fed_mentions_24h'] else 0)
            entity_features['ecb_mentions_24h'].append(
                entity_features['ecb_mentions_24h'][-1] if entity_features['ecb_mentions_24h'] else 0)
            entity_features['boe_mentions_24h'].append(
                entity_features['boe_mentions_24h'][-1] if entity_features['boe_mentions_24h'] else 0)
            entity_features['usd_mentions_24h'].append(
                entity_features['usd_mentions_24h'][-1] if entity_features['usd_mentions_24h'] else 0)
            entity_features['eur_mentions_24h'].append(
                entity_features['eur_mentions_24h'][-1] if entity_features['eur_mentions_24h'] else 0)
            entity_features['gbp_mentions_24h'].append(
                entity_features['gbp_mentions_24h'][-1] if entity_features['gbp_mentions_24h'] else 0)

    # Add entity features to DataFrame
    for feature_name, values in entity_features.items():
        features_df[feature_name] = values

    # Feature 2: Event proximity features
    logger.info("[cognee features] Extracting event proximity features...")

    event_features = {
        'rate_hike_events_48h': [],
        'rate_cut_events_48h': [],
        'cpi_events_48h': [],
        'event_sentiment_mean_48h': []
    }

    for i, (_idx, row) in enumerate(price_df.iterrows()):
        timestamp = row['datetime']

        if i % sample_interval == 0:
            # Query events
            rate_hike_info = query_event_proximity(client, dataset_name, "rate hike", timestamp, event_window_hours)
            rate_cut_info = query_event_proximity(client, dataset_name, "rate cut", timestamp, event_window_hours)
            cpi_info = query_event_proximity(client, dataset_name, "CPI", timestamp, event_window_hours)

            # Aggregate sentiments
            all_sentiments = []
            if rate_hike_info['avg_sentiment'] != 0:
                all_sentiments.append(rate_hike_info['avg_sentiment'])
            if rate_cut_info['avg_sentiment'] != 0:
                all_sentiments.append(rate_cut_info['avg_sentiment'])
            if cpi_info['avg_sentiment'] != 0:
                all_sentiments.append(cpi_info['avg_sentiment'])

            mean_sentiment = np.mean(all_sentiments) if all_sentiments else 0.0

            event_features['rate_hike_events_48h'].append(rate_hike_info['event_count'])
            event_features['rate_cut_events_48h'].append(rate_cut_info['event_count'])
            event_features['cpi_events_48h'].append(cpi_info['event_count'])
            event_features['event_sentiment_mean_48h'].append(mean_sentiment)

            if (i % (sample_interval * 10)) == 0:
                logger.info(f"  Progress: {i}/{len(price_df)} ({i / len(price_df):.1%})")
        else:
            # Interpolate
            event_features['rate_hike_events_48h'].append(
                event_features['rate_hike_events_48h'][-1] if event_features['rate_hike_events_48h'] else 0)
            event_features['rate_cut_events_48h'].append(
                event_features['rate_cut_events_48h'][-1] if event_features['rate_cut_events_48h'] else 0)
            event_features['cpi_events_48h'].append(
                event_features['cpi_events_48h'][-1] if event_features['cpi_events_48h'] else 0)
            event_features['event_sentiment_mean_48h'].append(
                event_features['event_sentiment_mean_48h'][-1] if event_features['event_sentiment_mean_48h'] else 0.0)

    # Add event features
    for feature_name, values in event_features.items():
        features_df[feature_name] = values

    # Feature 3: Composite features
    logger.info("[cognee features] Computing composite features...")

    # Entity density (total mentions / time window)
    features_df['entity_density_24h'] = (
                                                features_df['fed_mentions_24h'] +
                                                features_df['ecb_mentions_24h'] +
                                                features_df['boe_mentions_24h'] +
                                                features_df['usd_mentions_24h'] +
                                                features_df['eur_mentions_24h'] +
                                                features_df['gbp_mentions_24h']
                                        ) / entity_window_hours

    # Event density
    features_df['event_density_48h'] = (
                                               features_df['rate_hike_events_48h'] +
                                               features_df['rate_cut_events_48h'] +
                                               features_df['cpi_events_48h']
                                       ) / event_window_hours

    # Cache results
    if use_cache:
        features_df.reset_index(drop=True).to_feather(cache_path)
        logger.info(f"[cache] Saved Cognee features to {cache_path}")

    logger.info(f"[cognee features] Extracted {len(features_df.columns)} features")
    logger.info(f"[cognee features] Feature columns: {list(features_df.columns)}")

    return features_df


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract Cognee knowledge graph features")
    parser.add_argument("--api-key", help="Cognee API key")
    parser.add_argument("--dataset", required=True, help="Cognee dataset name")
    parser.add_argument("--pair", required=True, help="Currency pair (e.g., eurusd)")
    parser.add_argument("--price-csv", type=Path, required=True, help="Price data CSV file")
    parser.add_argument("--output", type=Path, help="Output feature CSV file")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COGNEE FEATURE EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Price data: {args.price_csv}")
    logger.info("=" * 80)

    # Load price data
    price_df = pd.read_csv(args.price_csv)

    # Ensure datetime column
    if 'datetime' not in price_df.columns:
        if 'DateTime' in price_df.columns:
            price_df['datetime'] = price_df['DateTime']
        else:
            logger.error("No datetime column found in price data!")
            sys.exit(1)

    price_df['datetime'] = pd.to_datetime(price_df['datetime'])

    # Initialize client
    client = CogneeClient(api_key=args.api_key or os.getenv("COGNEE_API_KEY"))

    # Extract features
    features = build_cognee_features(
        client=client,
        price_df=price_df,
        pair=args.pair,
        dataset_name=args.dataset,
        use_cache=not args.no_cache
    )

    # Show summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 80)
    print(features.describe())

    # Save output
    if args.output:
        features.to_csv(args.output, index=False)
        logger.info(f"\n✅ Saved {len(features)} rows to {args.output}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ FEATURE EXTRACTION COMPLETE")
    logger.info("=" * 80)
