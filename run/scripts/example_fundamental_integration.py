"""
Example: Collect and integrate fundamental data for forex training.

This script demonstrates how to:
1. Collect fundamental economic data
2. Merge with existing price data
3. Save enhanced dataset for training

Usage:
    python run/scripts/example_fundamental_integration.py --pair EURUSD --start 2023-01-01 --end 2023-12-31
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os

import pandas as pd

from data.extended_data_collection import (
    collect_all_forex_fundamentals,
    merge_with_price_data,
    save_fundamental_data,
)
from data.pipeline_controller import controller
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect fundamental data for forex trading")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair (e.g., EURUSD)")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/fundamentals", help="Output directory")
    parser.add_argument("--price-data", help="Optional: Path to price data to merge with")
    parser.add_argument("--date-column", default="datetime", help="Name of date column in price data (default: datetime)")
    parser.add_argument("--sources", nargs="+", default=None,
                       help="Data sources to collect (space-separated list: trade economic shocks). Example: --sources trade economic")
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("Fundamental Data Collection Example")
    logger.info("="*70)
    logger.info(f"Currency Pair: {args.pair}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("")

    # Step 1: Collect fundamental data
    logger.info("Step 1: Collecting fundamental data...")
    logger.info("-" * 70)

    try:
        fundamentals = collect_all_forex_fundamentals(
            currency_pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            comtrade_api_key=os.getenv("COMTRADE_API_KEY"),
            fred_api_key=os.getenv("FRED_API_KEY"),
            include_sources=args.sources
        )

        # Display summary
        logger.info("\nData Collection Summary:")
        for source, df in fundamentals.items():
            if not df.empty and 'date' in df.columns:
                logger.info(f"  ✅ {source:12s}: {len(df):6d} records, "
                          f"{df['date'].min()} to {df['date'].max()}")
            elif not df.empty:
                logger.info(f"  ✅ {source:12s}: {len(df):6d} records")
            else:
                logger.info(f"  ⚠️  {source:12s}: No data")

    except Exception as e:
        logger.error(f"Error collecting fundamental data: {e}")
        return 1

    # Step 2: Save fundamental data
    logger.info("\nStep 2: Saving fundamental data...")
    logger.info("-" * 70)

    try:
        paths = save_fundamental_data(
            data=fundamentals,
            output_dir=args.output_dir,
            currency_pair=args.pair,
            file_format="parquet"
        )

        logger.info("\nSaved Files:")
        for source, path in paths.items():
            file_size = path.stat().st_size / 1024  # KB
            logger.info(f"  📁 {path} ({file_size:.1f} KB)")

    except Exception as e:
        logger.error(f"Error saving fundamental data: {e}")
        return 1

    # Step 3: Optionally merge with price data
    if args.price_data:
        logger.info("\nStep 3: Merging with price data...")
        logger.info("-" * 70)

        try:
            price_file = Path(args.price_data)
            if not price_file.exists():
                logger.error(f"Price data file not found: {price_file}")
                return 1

            # Load price data
            if price_file.suffix == '.parquet':
                price_df = pd.read_parquet(price_file)
            elif price_file.suffix == '.csv':
                price_df = pd.read_csv(price_file, parse_dates=[args.date_column])
            else:
                logger.error(f"Unsupported price data format: {price_file.suffix}")
                return 1

            logger.info(f"  Loaded price data: {len(price_df)} rows, {len(price_df.columns)} columns")

            # Merge
            merged_df = merge_with_price_data(fundamentals, price_df, date_column=args.date_column)

            # Save merged dataset
            output_file = Path(args.output_dir) / f"{args.pair}_merged.parquet"
            merged_df.to_parquet(output_file)

            logger.info(f"  ✅ Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            logger.info(f"  📁 Saved to: {output_file}")

            # Display sample columns
            fundamental_cols = [col for col in merged_df.columns
                              if col.startswith(('trade_', 'economic_', 'shocks_'))]
            logger.info(f"\n  Added {len(fundamental_cols)} fundamental features:")
            for i, col in enumerate(fundamental_cols[:10], 1):
                logger.info(f"    {i}. {col}")
            if len(fundamental_cols) > 10:
                logger.info(f"    ... and {len(fundamental_cols) - 10} more")

        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return 1

    # Step 4: Record in pipeline database
    logger.info("\nStep 4: Recording in pipeline database...")
    logger.info("-" * 70)

    try:
        # Use pipeline controller to record collection
        controller.collect_fundamental_data(
            currency_pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            comtrade_api_key=os.getenv("COMTRADE_API_KEY"),
            fred_api_key=os.getenv("FRED_API_KEY"),
            include_sources=args.sources
        )

        # Get pipeline status
        status = controller.get_pipeline_status()
        logger.info(f"  Pipeline Status: {status.get('pipeline_status', 'UNKNOWN')}")
        logger.info(f"  Total Collections: {status.get('collections_completed', 0)}")

    except Exception as e:
        logger.warning(f"Could not record in pipeline database: {e}")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ Fundamental Data Collection Complete!")
    logger.info("="*70)
    logger.info("\nNext Steps:")
    logger.info("  1. Review collected data in: " + args.output_dir)
    logger.info("  2. Use merged dataset for training if price data was provided")
    logger.info("  3. Configure feature engineering in your training pipeline")
    logger.info("  4. Adjust data/prepare_dataset.py to include fundamentals")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
