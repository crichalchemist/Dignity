"""
Powerball Lottery data downloader and converter.

Downloads Powerball winning numbers from Kaggle and converts to time-series format
compatible with the Sequence pipeline. Used for control experiments to validate
that models don't find spurious patterns in pure randomness.

Kaggle Dataset: https://www.kaggle.com/datasets/ulrikthygepedersen/lottery-powerball-winning-numbers/data

Usage:
    # Manual download (recommended for first run):
    # 1. Download CSV from Kaggle manually
    # 2. Place in data/lottery/raw/powerball.csv
    # 3. Run this script

    python data/downloaders/lottery_downloader.py \
        --input-file data/lottery/raw/powerball.csv \
        --output-dir data/lottery \
        --create-separate-balls

    # Or use Kaggle API (requires kaggle credentials):
    python data/downloaders/lottery_downloader.py \
        --use-kaggle-api \
        --output-dir data/lottery
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline_controller import PipelineController


def download_from_kaggle(output_dir: Path) -> Path:
    """
    Download Powerball dataset using Kaggle API.

    Requires:
        - KAGGLE_USERNAME and KAGGLE_KEY environment variables
        - or ~/.kaggle/kaggle.json credentials file

    Args:
        output_dir: Directory to save downloaded file

    Returns:
        Path to downloaded CSV file
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Run: pip install kaggle\n"
            "Or download dataset manually from: "
            "https://www.kaggle.com/datasets/ulrikthygepedersen/lottery-powerball-winning-numbers/data"
        )

    # Check for credentials
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            raise ValueError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY env vars "
                "or create ~/.kaggle/kaggle.json"
            )

    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Powerball dataset from Kaggle...")
    api.dataset_download_files(
        "ulrikthygepedersen/lottery-powerball-winning-numbers",
        path=str(raw_dir),
        unzip=True,
    )

    # Find the downloaded CSV file
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir} after download")

    print(f"✓ Downloaded to {csv_files[0]}")
    return csv_files[0]


def convert_to_timeseries_combined(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """
    Convert lottery draws to single "powerball" pair.

    Maps:
        - Ball 1 → open
        - Ball 2 → high
        - Ball 3 → low
        - Ball 4 → close
        - Ball 5 → volume
        - Powerball → auxiliary feature (not used in OHLCV)

    Args:
        df: DataFrame with lottery draw data
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    pair_dir = output_dir / "powerball"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Extract columns (column names may vary)
    # Common formats: Draw Date, Ball 1-5, Powerball
    # Try to find the right columns
    date_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "draw_date", "draw date"]):
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Could not find date column. Available columns: {df.columns.tolist()}")

    # Find ball columns
    ball_cols = []
    for i in range(1, 6):
        found = False
        for col in df.columns:
            if f"ball {i}" in col.lower() or f"ball{i}" in col.lower() or f"white_{i}" in col.lower():
                ball_cols.append(col)
                found = True
                break
        if not found:
            # Try generic number columns
            if f"{i}" in df.columns or f"Number {i}" in df.columns:
                ball_cols.append(f"{i}" if f"{i}" in df.columns else f"Number {i}")
            else:
                raise ValueError(f"Could not find ball {i} column")

    # Find powerball column
    powerball_col = None
    for col in df.columns:
        if "powerball" in col.lower() or "power ball" in col.lower() or "bonus" in col.lower():
            powerball_col = col
            break

    print("Using columns:")
    print(f"  Date: {date_col}")
    print(f"  Balls 1-5: {ball_cols}")
    print(f"  Powerball: {powerball_col}")

    # Create OHLCV DataFrame
    output_df = pd.DataFrame()
    output_df["datetime"] = pd.to_datetime(df[date_col])
    output_df["open"] = df[ball_cols[0]].astype(float)
    output_df["high"] = df[ball_cols[1]].astype(float)
    output_df["low"] = df[ball_cols[2]].astype(float)
    output_df["close"] = df[ball_cols[3]].astype(float)
    output_df["volume"] = df[ball_cols[4]].astype(float)

    # Sort by datetime
    output_df = output_df.sort_values("datetime").reset_index(drop=True)

    # Save year-by-year (like HistData format)
    years = output_df["datetime"].dt.year.unique()

    for year in years:
        df_year = output_df[output_df["datetime"].dt.year == year].copy()

        if df_year.empty:
            continue

        filename = f"Lottery_POWERBALL_1day_{year}.csv"
        output_path = pair_dir / filename

        # Format datetime as "YYYYMMDD HHMMSS"
        df_year["datetime_str"] = df_year["datetime"].dt.strftime("%Y%m%d %H%M%S")

        # Write semicolon-delimited CSV (no header, matches HistData format)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for _, row in df_year.iterrows():
                writer.writerow(
                    [
                        row["datetime_str"],
                        f"{row['open']:.1f}",
                        f"{row['high']:.1f}",
                        f"{row['low']:.1f}",
                        f"{row['close']:.1f}",
                        int(row["volume"]),
                    ]
                )

        print(f"  Saved {len(df_year)} draws to {output_path}")

    return pair_dir


def convert_to_timeseries_separate(
    df: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """
    Convert lottery draws to 6 separate "pairs" (one per ball position).

    Creates:
        - powerball_ball1: Ball 1 time series
        - powerball_ball2: Ball 2 time series
        - ... (balls 3-5)
        - powerball_bonus: Powerball time series

    Each treated as a "price" with close=high=low=open=ball value

    Args:
        df: DataFrame with lottery draw data
        output_dir: Output directory

    Returns:
        List of paths to saved directories
    """
    # Find date column
    date_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "draw_date", "draw date"]):
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Could not find date column. Available columns: {df.columns.tolist()}")

    # Find ball columns
    ball_cols = []
    for i in range(1, 6):
        found = False
        for col in df.columns:
            if f"ball {i}" in col.lower() or f"ball{i}" in col.lower() or f"white_{i}" in col.lower():
                ball_cols.append(col)
                found = True
                break
        if not found:
            if f"{i}" in df.columns or f"Number {i}" in df.columns:
                ball_cols.append(f"{i}" if f"{i}" in df.columns else f"Number {i}")

    # Find powerball
    powerball_col = None
    for col in df.columns:
        if "powerball" in col.lower() or "power ball" in col.lower() or "bonus" in col.lower():
            powerball_col = col
            break

    # Parse datetime
    datetime_series = pd.to_datetime(df[date_col])

    saved_dirs = []

    # Create separate pair for each ball position
    ball_names = ["ball1", "ball2", "ball3", "ball4", "ball5"]
    for i, (ball_name, ball_col) in enumerate(zip(ball_names, ball_cols)):
        pair_name = f"powerball_{ball_name}"
        pair_dir = output_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Create OHLCV (all same since it's a single value)
        output_df = pd.DataFrame()
        output_df["datetime"] = datetime_series
        output_df["open"] = df[ball_col].astype(float)
        output_df["high"] = df[ball_col].astype(float)
        output_df["low"] = df[ball_col].astype(float)
        output_df["close"] = df[ball_col].astype(float)
        output_df["volume"] = 1  # Dummy volume

        output_df = output_df.sort_values("datetime").reset_index(drop=True)

        # Save year-by-year
        years = output_df["datetime"].dt.year.unique()

        for year in years:
            df_year = output_df[output_df["datetime"].dt.year == year].copy()

            if df_year.empty:
                continue

            filename = f"Lottery_{pair_name.upper()}_1day_{year}.csv"
            output_path = pair_dir / filename

            df_year["datetime_str"] = df_year["datetime"].dt.strftime("%Y%m%d %H%M%S")

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                for _, row in df_year.iterrows():
                    writer.writerow(
                        [
                            row["datetime_str"],
                            f"{row['open']:.1f}",
                            f"{row['high']:.1f}",
                            f"{row['low']:.1f}",
                            f"{row['close']:.1f}",
                            int(row["volume"]),
                        ]
                    )

        print(f"  Created pair: {pair_name} ({len(output_df)} draws)")
        saved_dirs.append(pair_dir)

    # Also create powerball bonus
    if powerball_col:
        pair_name = "powerball_bonus"
        pair_dir = output_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        output_df = pd.DataFrame()
        output_df["datetime"] = datetime_series
        output_df["open"] = df[powerball_col].astype(float)
        output_df["high"] = df[powerball_col].astype(float)
        output_df["low"] = df[powerball_col].astype(float)
        output_df["close"] = df[powerball_col].astype(float)
        output_df["volume"] = 1

        output_df = output_df.sort_values("datetime").reset_index(drop=True)

        years = output_df["datetime"].dt.year.unique()

        for year in years:
            df_year = output_df[output_df["datetime"].dt.year == year].copy()

            if df_year.empty:
                continue

            filename = f"Lottery_{pair_name.upper()}_1day_{year}.csv"
            output_path = pair_dir / filename

            df_year["datetime_str"] = df_year["datetime"].dt.strftime("%Y%m%d %H%M%S")

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                for _, row in df_year.iterrows():
                    writer.writerow(
                        [
                            row["datetime_str"],
                            f"{row['open']:.1f}",
                            f"{row['high']:.1f}",
                            f"{row['low']:.1f}",
                            f"{row['close']:.1f}",
                            int(row["volume"]),
                        ]
                    )

        print(f"  Created pair: {pair_name} ({len(output_df)} draws)")
        saved_dirs.append(pair_dir)

    return saved_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Powerball lottery data"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input CSV file (if already downloaded)",
    )
    parser.add_argument(
        "--use-kaggle-api",
        action="store_true",
        help="Download from Kaggle using API (requires credentials)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lottery",
        help="Output directory for lottery data",
    )
    parser.add_argument(
        "--create-separate-balls",
        action="store_true",
        help="Create 6 separate pairs (one per ball) instead of combined",
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input file
    if args.use_kaggle_api:
        input_file = download_from_kaggle(output_dir)
    elif args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            print("\nTo download manually:")
            print("1. Visit: https://www.kaggle.com/datasets/ulrikthygepedersen/lottery-powerball-winning-numbers/data")
            print("2. Download the CSV file")
            print(f"3. Place it at: {output_dir / 'raw' / 'powerball.csv'}")
            print(f"4. Run: python {Path(__file__).name} --input-file {output_dir / 'raw' / 'powerball.csv'}")
            sys.exit(1)
    else:
        # Check default location
        default_path = output_dir / "raw" / "powerball.csv"
        if default_path.exists():
            input_file = default_path
        else:
            print("Error: No input file specified and no default file found")
            print(f"\nExpected location: {default_path}")
            print("\nOptions:")
            print("  1. Use --input-file to specify CSV path")
            print("  2. Use --use-kaggle-api to download automatically")
            print("  3. Download manually and place at default location")
            sys.exit(1)

    # Load lottery data
    print(f"Loading lottery data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} lottery draws")
    print(f"  Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")

    # Convert to time series
    if args.create_separate_balls:
        print("\nCreating separate pairs for each ball position...")
        saved_dirs = convert_to_timeseries_separate(df, output_dir)
        print(f"\n✓ Created {len(saved_dirs)} lottery pairs")
    else:
        print("\nCreating combined powerball pair...")
        saved_dir = convert_to_timeseries_combined(df, output_dir)
        print(f"\n✓ Created lottery pair at {saved_dir}")

    # Log to pipeline controller
    controller = PipelineController()
    controller.log_data_collection(
        source="lottery",
        symbol="powerball",
        start_date=str(df.iloc[0, 0]),
        end_date=str(df.iloc[-1, 0]),
        rows_collected=len(df),
        status="success",
    )

    print(f"\n{'='*60}")
    print("Lottery data conversion complete!")
    print(f"Output directory: {output_dir}")
    print("\nUsage with prepare_dataset.py:")
    if args.create_separate_balls:
        print("  python data/prepare_dataset.py --pairs powerball_ball1 --input-root data/lottery")
    else:
        print("  python data/prepare_dataset.py --pairs powerball --input-root data/lottery")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
