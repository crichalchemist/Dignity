"""
TwelveData API downloader for missing FX pairs.

Downloads historical OHLCV data from TwelveData API to supplement HistData coverage.
Optimized for free tier: 800 API calls, 5000 datapoints per call.

Usage:
    python data/downloaders/twelvedata_downloader.py \
        --pairs usdjpy,usdcad,audusd \
        --start-date 2010-01-01 \
        --end-date 2024-12-31 \
        --interval 1day \
        --output-dir data/twelvedata \
        --max-calls 800

Features:
    - Auto-detects missing pairs from pairs.csv
    - Converts prices to pips (multiply by 10,000 for FX)
    - Rate limiting & call budget tracking
    - Output format compatible with HistData (semicolon-delimited CSV)
    - SQLite tracking via PipelineController
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline_controller import PipelineController


class TwelveDataClient:
    """Client for TwelveData API with rate limiting and budget tracking."""

    BASE_URL = "https://api.twelvedata.com"

    def __init__(
        self,
        api_key: str,
        max_calls: int = 800,
        calls_per_minute: int = 8,  # Free tier limit
    ):
        """
        Initialize TwelveData client.

        Args:
            api_key: TwelveData API key
            max_calls: Maximum total API calls (budget)
            calls_per_minute: Rate limit for API calls
        """
        self.api_key = api_key
        self.max_calls = max_calls
        self.calls_per_minute = calls_per_minute
        self.calls_made = 0
        self.last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        if self.calls_per_minute > 0:
            min_interval = 60.0 / self.calls_per_minute
            elapsed = time.time() - self.last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def _check_budget(self):
        """Check if we've exceeded call budget."""
        if self.calls_made >= self.max_calls:
            raise RuntimeError(
                f"API call budget exceeded: {self.calls_made}/{self.max_calls} calls used"
            )

    def get_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        start_date: str | None = None,
        end_date: str | None = None,
        outputsize: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch time series data from TwelveData.

        Args:
            symbol: Trading pair symbol (e.g., "EUR/USD", "BTC/USD")
            interval: Time interval (1min, 5min, 1h, 1day, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            outputsize: Number of data points (max 5000 for free tier)

        Returns:
            DataFrame with datetime, open, high, low, close, volume columns
        """
        self._check_budget()
        self._rate_limit()

        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": outputsize,
            "format": "JSON",
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        endpoint = f"{self.BASE_URL}/time_series"

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            self.calls_made += 1
            self.last_call_time = time.time()

            response.raise_for_status()
            data = response.json()

            if "status" in data and data["status"] == "error":
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

            if "values" not in data:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(data["values"])

            # Rename columns to match HistData schema
            df = df.rename(
                columns={
                    "datetime": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Parse datetime
            df["datetime"] = pd.to_datetime(df["datetime"])

            # Sort by datetime (ascending)
            df = df.sort_values("datetime").reset_index(drop=True)

            print(
                f"✓ Downloaded {len(df)} bars for {symbol} "
                f"({df['datetime'].min()} to {df['datetime'].max()})"
            )
            print(f"  API calls: {self.calls_made}/{self.max_calls}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"✗ HTTP error fetching {symbol}: {e}")
            raise
        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")
            raise


def convert_to_pips(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Convert FX prices to pips (multiply by 10,000).

    Args:
        df: DataFrame with OHLCV columns
        pair: Trading pair code

    Returns:
        DataFrame with prices in pips
    """
    # Only convert FX pairs (not crypto, commodities)
    crypto_keywords = ["btc", "eth", "ada", "sol", "xrp", "doge", "ltc", "link", "matic", "avax"]
    commodity_keywords = ["xau", "xag"]

    pair_lower = pair.lower()

    is_crypto = any(kw in pair_lower for kw in crypto_keywords)
    is_commodity = any(kw in pair_lower for kw in commodity_keywords)

    if is_crypto or is_commodity:
        print(f"  Skipping pips conversion for {pair} (crypto/commodity)")
        return df

    # Convert FX prices to pips
    df_pips = df.copy()
    for col in ["open", "high", "low", "close"]:
        df_pips[col] = df_pips[col] * 10000

    print(f"  Converted {pair} prices to pips (×10,000)")
    return df_pips


def save_to_histdata_format(
    df: pd.DataFrame,
    output_dir: Path,
    pair: str,
    year: int,
    interval: str,
):
    """
    Save DataFrame in HistData-compatible format.

    Format: YYYYMMDD HHMMSS;open;high;low;close;volume

    Args:
        df: DataFrame with datetime, OHLCV columns
        output_dir: Output directory path
        pair: Trading pair code
        year: Year for filename
        interval: Data interval (for filename)
    """
    pair_dir = output_dir / pair.lower()
    pair_dir.mkdir(parents=True, exist_ok=True)

    filename = f"TwelveData_{pair.upper()}_{interval}_{year}.csv"
    output_path = pair_dir / filename

    # Filter for specific year
    df_year = df[df["datetime"].dt.year == year].copy()

    if df_year.empty:
        print(f"  No data for year {year}, skipping file")
        return

    # Format datetime as "YYYYMMDD HHMMSS"
    df_year["datetime_str"] = df_year["datetime"].dt.strftime("%Y%m%d %H%M%S")

    # Write semicolon-delimited CSV (no header, matches HistData format)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for _, row in df_year.iterrows():
            writer.writerow(
                [
                    row["datetime_str"],
                    f"{row['open']:.5f}",
                    f"{row['high']:.5f}",
                    f"{row['low']:.5f}",
                    f"{row['close']:.5f}",
                    int(row["volume"]) if pd.notna(row["volume"]) else 0,
                ]
            )

    print(f"  Saved {len(df_year)} bars to {output_path}")


def find_missing_pairs(pairs_csv_path: Path, histdata_dir: Path) -> list[str]:
    """
    Find pairs in pairs.csv that don't have HistData coverage.

    Args:
        pairs_csv_path: Path to pairs.csv
        histdata_dir: Path to histdata directory

    Returns:
        List of missing pair codes
    """
    # Read pairs.csv
    pairs_df = pd.read_csv(pairs_csv_path)

    # Get all pair codes
    all_pairs = pairs_df["pair"].str.lower().tolist()

    # Check which have data in histdata directory
    histdata_pairs = []
    for pair in all_pairs:
        pair_dir = histdata_dir / pair
        if pair_dir.exists() and pair_dir.is_dir():
            # Check if directory has any files
            files = list(pair_dir.glob("*.zip")) + list(pair_dir.glob("*.csv"))
            if files:
                histdata_pairs.append(pair)

    # Missing pairs = all pairs - histdata pairs
    missing_pairs = [p for p in all_pairs if p not in histdata_pairs]

    return missing_pairs


def normalize_pair_symbol(pair: str) -> str:
    """
    Normalize pair code to TwelveData format.

    TwelveData uses "/" separator: EUR/USD, BTC/USD, etc.

    Args:
        pair: Pair code (e.g., "eurusd", "btcusd")

    Returns:
        Normalized symbol (e.g., "EUR/USD", "BTC/USD")
    """
    pair_upper = pair.upper()

    # Special handling for crypto (already has dash)
    if "-" in pair_upper:
        return pair_upper.replace("-", "/")

    # Special handling for commodities
    if pair_upper.startswith("XAU"):
        return "XAU/USD"
    if pair_upper.startswith("XAG"):
        return "XAG/USD"

    # Standard FX pairs (6 characters)
    if len(pair_upper) == 6:
        return f"{pair_upper[:3]}/{pair_upper[3:]}"

    # Default: return as-is
    return pair_upper


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from TwelveData API"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        help="Comma-separated list of pairs to download (or 'missing' to auto-detect)",
        default="missing",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1day",
        help="Time interval (1min, 5min, 1h, 1day, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/twelvedata",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=800,
        help="Maximum API calls (budget limit)",
    )
    parser.add_argument(
        "--convert-to-pips",
        action="store_true",
        help="Convert FX prices to pips (×10,000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="TwelveData API key (or set TWELVEDATA_API_KEY env var)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("Error: TwelveData API key required. Set TWELVEDATA_API_KEY env var or use --api-key")
        sys.exit(1)

    # Setup paths
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    pairs_csv_path = ROOT / "pairs.csv"
    histdata_dir = ROOT / "data" / "histdata"

    # Determine which pairs to download
    if args.pairs.lower() == "missing":
        print("Auto-detecting missing pairs...")
        pairs = find_missing_pairs(pairs_csv_path, histdata_dir)
        print(f"Found {len(pairs)} missing pairs: {', '.join(pairs)}")
    else:
        pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    if not pairs:
        print("No pairs to download")
        sys.exit(0)

    # Initialize client
    client = TwelveDataClient(api_key=api_key, max_calls=args.max_calls)

    # Calculate year range
    start_year = datetime.strptime(args.start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(args.end_date, "%Y-%m-%d").year

    # Initialize pipeline controller
    controller = PipelineController()

    # Download each pair
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"Downloading {pair.upper()}")
        print(f"{'='*60}")

        try:
            # Normalize pair for TwelveData API
            symbol = normalize_pair_symbol(pair)

            # Download data
            df = client.get_time_series(
                symbol=symbol,
                interval=args.interval,
                start_date=args.start_date,
                end_date=args.end_date,
                outputsize=5000,
            )

            # Convert to pips if requested
            if args.convert_to_pips:
                df = convert_to_pips(df, pair)

            # Save by year (split into separate files like HistData)
            for year in range(start_year, end_year + 1):
                save_to_histdata_format(
                    df, output_dir, pair, year, args.interval
                )

            # Log to pipeline controller
            controller.log_data_collection(
                source="twelvedata",
                symbol=pair,
                start_date=args.start_date,
                end_date=args.end_date,
                rows_collected=len(df),
                status="success",
            )

            print(f"✓ Completed {pair}")

        except Exception as e:
            print(f"✗ Failed to download {pair}: {e}")
            controller.log_data_collection(
                source="twelvedata",
                symbol=pair,
                start_date=args.start_date,
                end_date=args.end_date,
                rows_collected=0,
                status="failed",
            )
            continue

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"API calls used: {client.calls_made}/{client.max_calls}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
