#!/usr/bin/env python3
"""
Pre-Build Cognee Knowledge Graphs for Historical FX/Crypto Data

This script builds knowledge graphs for all currency pairs across multiple years,
caching the results to avoid repeated API calls during training.

Run this once to pre-process historical data (2010-2024), then use the cached
features during training for fast, cost-effective iterations.

Usage:
    # Pre-build for EUR/USD 2020-2024
    python scripts/prebuild_cognee_graph.py \\
        --pairs eurusd \\
        --start-year 2020 \\
        --end-year 2024 \\
        --include-gdelt \\
        --include-economic-indicators \\
        --include-price-narratives

    # Pre-build for all major pairs (full historical data)
    python scripts/prebuild_cognee_graph.py \\
        --pairs eurusd,gbpusd,usdjpy \\
        --start-year 2010 \\
        --end-year 2024 \\
        --all-features

Estimated Runtime: ~2-4 hours for full historical data (can run in Colab overnight)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from data.cognee_client import CogneeClient
from data.cognee_processor import CogneeDataProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def prebuild_for_pair_year(
        pair: str,
        year: int,
        client: CogneeClient,
        processor: CogneeDataProcessor,
        include_gdelt: bool = True,
        include_economic_indicators: bool = True,
        include_price_narratives: bool = True,
        data_root: Path = Path("data"),
        cache_dir: Path = Path("data/cognee_cache")
) -> bool:
    """
    Pre-build knowledge graph for a specific currency pair and year.

    Args:
        pair: Currency pair (e.g., "eurusd")
        year: Year to process
        client: Cognee API client
        processor: Cognee data processor
        include_gdelt: Whether to ingest GDELT news
        include_economic_indicators: Whether to ingest economic data
        include_price_narratives: Whether to generate price narratives
        data_root: Root directory for data files
        cache_dir: Directory for cached results

    Returns:
        True if successful, False otherwise
    """
    dataset_name = f"{pair}_{year}"
    logger.info("=" * 80)
    logger.info(f"BUILDING KNOWLEDGE GRAPH: {dataset_name}")
    logger.info("=" * 80)

    # Step 1: Load price data for this pair/year
    logger.info(f"[{dataset_name}] Loading price data...")
    price_file = data_root / pair / f"{pair}_{year}.csv"

    if not price_file.exists():
        # Try .zip format
        zip_file = data_root / pair / f"DAT_ASCII_{pair.upper()}_M1_{year}.zip"
        if not zip_file.exists():
            logger.warning(f"[{dataset_name}] No price data found at {price_file} or {zip_file}")
            return False

        # Load from zip
        import zipfile
        with zipfile.ZipFile(zip_file) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                price_df = pd.read_csv(
                    f,
                    sep=';',
                    header=None,
                    names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
                )
    else:
        price_df = pd.read_csv(price_file)

    # Normalize columns
    if 'DateTime' in price_df.columns:
        price_df['datetime'] = pd.to_datetime(price_df['DateTime'])
    else:
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])

    logger.info(f"[{dataset_name}] Loaded {len(price_df):,} price bars")

    # Step 2: Ingest GDELT news
    if include_gdelt:
        logger.info(f"[{dataset_name}] Downloading GDELT news for {year}...")
        try:
            from data.gdelt_bigquery import FINANCIAL_THEMES, query_gdelt_sentiment

            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            gdelt_df = query_gdelt_sentiment(
                start_date=start_date,
                end_date=end_date,
                themes=FINANCIAL_THEMES
            )

            if len(gdelt_df) > 0:
                logger.info(f"[{dataset_name}] Ingesting {len(gdelt_df):,} GDELT records...")
                processor.ingest_gdelt_news(gdelt_df, dataset_name)
            else:
                logger.warning(f"[{dataset_name}] No GDELT data for {year}")

        except Exception as e:
            logger.error(f"[{dataset_name}] Failed to download/ingest GDELT: {e}")

    # Step 3: Ingest economic indicators
    if include_economic_indicators:
        logger.info(f"[{dataset_name}] Downloading economic indicators for {year}...")
        try:
            from data.downloaders.economic_indicators import download_forex_fred_bundle

            indicators_df = download_forex_fred_bundle(
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                api_key=os.getenv("FRED_API_KEY")
            )

            if len(indicators_df) > 0:
                logger.info(f"[{dataset_name}] Ingesting {len(indicators_df):,} economic indicators...")
                processor.ingest_economic_indicators(indicators_df, dataset_name)
            else:
                logger.warning(f"[{dataset_name}] No economic indicators for {year}")

        except Exception as e:
            logger.error(f"[{dataset_name}] Failed to download/ingest economic indicators: {e}")

    # Step 4: Generate and ingest price pattern narratives
    if include_price_narratives:
        logger.info(f"[{dataset_name}] Generating price pattern narratives...")
        try:
            from data.price_pattern_narrator import generate_pattern_text

            pattern_df = generate_pattern_text(
                df=price_df,
                pair=pair.upper()
            )

            logger.info(f"[{dataset_name}] Ingesting {len(pattern_df):,} price patterns...")
            processor.ingest_price_patterns(pattern_df, dataset_name, pair.upper())

        except Exception as e:
            logger.error(f"[{dataset_name}] Failed to generate/ingest price narratives: {e}")

    # Step 5: Trigger cognify and wait for completion
    logger.info(f"[{dataset_name}] Triggering knowledge graph building...")
    try:
        job_id = client.cognify(dataset_name)
        success = processor.wait_for_cognify(job_id, timeout=600)

        if not success:
            logger.error(f"[{dataset_name}] Knowledge graph building failed")
            return False

        logger.info(f"[{dataset_name}] ✅ Knowledge graph built successfully")

    except Exception as e:
        logger.error(f"[{dataset_name}] Cognify failed: {e}")
        return False

    # Step 6: Export entity cache
    logger.info(f"[{dataset_name}] Exporting entity cache...")
    try:
        cache_files = processor.export_entity_cache(
            dataset_name=dataset_name,
            output_dir=cache_dir
        )

        logger.info(f"[{dataset_name}] ✅ Exported {len(cache_files)} cache files")

    except Exception as e:
        logger.error(f"[{dataset_name}] Failed to export cache: {e}")

    logger.info(f"[{dataset_name}] ✅ Pre-build complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build Cognee knowledge graphs for historical FX/crypto data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Comma-separated currency pairs (e.g., eurusd,gbpusd,usdjpy)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year for pre-building (default: 2020)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year for pre-building (default: current year)"
    )
    parser.add_argument(
        "--cognee-api-key",
        help="Cognee API key (or set COGNEE_API_KEY env var)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for price data (default: data/)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cognee_cache"),
        help="Directory for cached results (default: data/cognee_cache)"
    )

    # Feature options
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Include all features (GDELT + economic indicators + price narratives)"
    )
    parser.add_argument(
        "--include-gdelt",
        action="store_true",
        help="Include GDELT news sentiment"
    )
    parser.add_argument(
        "--include-economic-indicators",
        action="store_true",
        help="Include economic indicators (FRED)"
    )
    parser.add_argument(
        "--include-price-narratives",
        action="store_true",
        help="Include price pattern narratives"
    )

    # Optional year chunking
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Process data in chunks of N years (default: 1 = process year-by-year)"
    )

    args = parser.parse_args()

    # Parse pairs
    pairs = [p.strip().lower() for p in args.pairs.split(',') if p.strip()]

    # Determine which features to include
    include_gdelt = args.all_features or args.include_gdelt
    include_economic_indicators = args.all_features or args.include_economic_indicators
    include_price_narratives = args.all_features or args.include_price_narratives

    logger.info("=" * 80)
    logger.info("COGNEE KNOWLEDGE GRAPH PRE-BUILD")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Years: {args.start_year}-{args.end_year}")
    logger.info("Features:")
    logger.info(f"  - GDELT News: {include_gdelt}")
    logger.info(f"  - Economic Indicators: {include_economic_indicators}")
    logger.info(f"  - Price Narratives: {include_price_narratives}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info("=" * 80)

    # Initialize Cognee client
    api_key = args.cognee_api_key or os.getenv("COGNEE_API_KEY")
    if not api_key:
        logger.error("Cognee API key not provided!")
        logger.error("Set COGNEE_API_KEY environment variable or use --cognee-api-key")
        return 1

    client = CogneeClient(api_key=api_key)
    processor = CogneeDataProcessor(client)

    # Ensure cache directory exists
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair and year
    total_datasets = len(pairs) * (args.end_year - args.start_year + 1)
    current_dataset = 0
    successful_datasets = 0
    failed_datasets = []

    for pair in pairs:
        for year in range(args.start_year, args.end_year + 1):
            current_dataset += 1

            logger.info("")
            logger.info(f"Processing dataset {current_dataset}/{total_datasets}")

            success = prebuild_for_pair_year(
                pair=pair,
                year=year,
                client=client,
                processor=processor,
                include_gdelt=include_gdelt,
                include_economic_indicators=include_economic_indicators,
                include_price_narratives=include_price_narratives,
                data_root=args.data_root,
                cache_dir=args.cache_dir
            )

            if success:
                successful_datasets += 1
            else:
                failed_datasets.append(f"{pair}_{year}")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PRE-BUILD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total datasets: {total_datasets}")
    logger.info(f"Successful: {successful_datasets}")
    logger.info(f"Failed: {len(failed_datasets)}")

    if failed_datasets:
        logger.warning("\nFailed datasets:")
        for dataset in failed_datasets:
            logger.warning(f"  - {dataset}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ PRE-BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Cached entities and features saved to: {args.cache_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run data preparation with --use-cognee flag")
    logger.info("2. Features will be loaded from cache (fast and free!)")
    logger.info("3. Optional: Run with --cognee-rebuild-graph to rebuild specific datasets")

    return 0 if len(failed_datasets) == 0 else 1


if __name__ == "__main__":
    exit(main())
