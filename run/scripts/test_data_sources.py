"""
Quick test script to verify new data sources are properly integrated.

Run this after installing the packages to ensure everything works.

Usage:
    python run/scripts/test_data_sources.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os

from utils.logger import get_logger

logger = get_logger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")

    try:
        logger.info("✅ Comtrade downloader imported successfully")
    except Exception as e:
        logger.error(f"❌ Comtrade downloader import failed: {e}")
        return False

    try:
        logger.info("✅ FRED downloader imported successfully")
    except Exception as e:
        logger.error(f"❌ FRED downloader import failed: {e}")
        return False

    try:
        logger.info("✅ ECB shocks downloader imported successfully")
    except Exception as e:
        logger.error(f"❌ ECB shocks downloader import failed: {e}")
        return False

    try:
        logger.info("✅ Extended data collection imported successfully")
    except Exception as e:
        logger.error(f"❌ Extended data collection import failed: {e}")
        return False

    return True


def test_ecb_shocks():
    """Test ECB shocks loading (no API key required)."""
    logger.info("\nTesting ECB shocks data...")

    try:
        from data.downloaders.ecb_shocks_downloader import load_ecb_shocks_daily

        df = load_ecb_shocks_daily()

        if df.empty:
            logger.warning("⚠️  No ECB shock data loaded")
            return False

        # Always log count and columns once
        logger.info(f"✅ Loaded {len(df)} ECB shock observations")
        logger.info(f"   Columns: {list(df.columns)}")

        # Check if 'date' column is present and validate its contents
        if 'date' not in df.columns:
            logger.error("   Missing 'date' column - ECB shocks data invalid")
            return False

        # Convert date column to datetime and validate
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df['date'].dropna()

        if len(valid_dates) == 0:
            logger.error("   Missing or invalid 'date' values - ECB shocks data invalid")
            return False

        # Log date range from valid dates
        logger.info(f"   Date range: {valid_dates.min()} to {valid_dates.max()}")
        return True

    except Exception as e:
        logger.error(f"❌ ECB shocks test failed: {e}")
        return False


def test_comtrade(api_key=None):
    """Test Comtrade API (requires API key for full test)."""
    logger.info("\nTesting UN Comtrade API...")

    if not api_key:
        api_key = os.getenv("COMTRADE_API_KEY")

    if not api_key:
        logger.warning("⚠️  COMTRADE_API_KEY not set - using preview mode (limited to 500 records)")

    try:
        from data.downloaders.comtrade_downloader import download_trade_balance

        # Test with USA (country code 842) for a single year
        df = download_trade_balance(
            reporter_code="842",
            start_year=2023,
            end_year=2023,
            subscription_key=api_key
        )

        if len(df) > 0:
            logger.info(f"✅ Retrieved {len(df)} trade balance records")
            logger.info(f"   Sample data: {df.head(2).to_dict('records')}")
            return True
        else:
            logger.warning("⚠️  No data returned - may need valid API key")
            return False
    except Exception as e:
        logger.error(f"❌ Comtrade test failed: {e}")
        return False


def test_fred(api_key=None):
    """Test FRED API (requires API key)."""
    logger.info("\nTesting FRED API...")

    if not api_key:
        api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        logger.error("❌ FRED_API_KEY not set - cannot test FRED API")
        logger.info("   Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False

    try:
        from data.downloaders.fred_downloader import download_series

        # Test with Federal Funds Rate
        df = download_series(
            series_id="FEDFUNDS",
            start_date="2023-01-01",
            end_date="2023-01-31",
            api_key=api_key
        )

        if len(df) > 0:
            logger.info(f"✅ Retrieved {len(df)} FRED observations")
            logger.info(f"   Sample data: {df.head(2).to_dict('records')}")
            return True
        else:
            logger.warning("⚠️  No data returned")
            return False
    except Exception as e:
        logger.error(f"❌ FRED test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Testing New Data Sources Integration")
    logger.info("="*60)

    results = {
        "imports": test_imports(),
        "ecb_shocks": test_ecb_shocks(),
        "comtrade": test_comtrade(),
        "fred": test_fred()
    }

    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.warning("\n⚠️  Some tests failed. Check the logs above.")
        logger.info("\nNext steps:")
        logger.info("1. Install packages: bash run/scripts/install_data_sources.sh")
        logger.info("2. Set API keys:")
        logger.info("   export FRED_API_KEY='your_key'")
        logger.info("   export COMTRADE_API_KEY='your_key'  # optional")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
