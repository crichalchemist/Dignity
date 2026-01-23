"""
Fetch FX data from Yahoo Finance to complement HistData coverage.

Usage:
  PYTHONPATH=.venv/bin/python data/download_yfinance_fx.py \\
    --output-root yfinance_output --start 2020-01-01 --interval 1h

Options:
  --pairs-csv: path to pairs.csv (defaults to repo-root pairs.csv if present, else data/pairs.csv)
  --pairs: comma list to override pairs file (e.g., eurusd,usdjpy)
  --start/--end: date range (YYYY-MM-DD)
  --interval: Yahoo interval (1d,1h,30m,15m,5m where allowed)
  --skip-existing: skip download when the target CSV already exists
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def resolve_pairs_file(pairs_csv_arg: str) -> Path:
    if pairs_csv_arg:
        return Path(pairs_csv_arg).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "pairs.csv"
    if default_root.exists():
        return default_root
    return (repo_root / "data" / "pairs.csv")


def load_pairs(pairs_csv: Path) -> list[str]:
    rows = pd.read_csv(pairs_csv)
    if "pair" not in rows.columns:
        raise ValueError(f"'pair' column not found in {pairs_csv}")
    return rows["pair"].str.lower().tolist()


def pair_to_ticker(pair: str) -> str:
    """Convert internal pair name to Yahoo Finance ticker.

    - FX pairs: EURUSD=X
    - Crypto pairs: BTC-USD
    - Commodities: GC=F (gold futures), SI=F (silver futures)
    """
    pair_upper = pair.upper()

    # Crypto pairs - Yahoo uses {CRYPTO}-USD format
    crypto_bases = {
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "SOLUSD": "SOL-USD",
        "BNBUSD": "BNB-USD",
        "ADAUSD": "ADA-USD",
        "XRPUSD": "XRP-USD",
        "DOGEUSD": "DOGE-USD",
        "AVAXUSD": "AVAX-USD",
        "MATICUSD": "MATIC-USD",
        "LINKUSD": "LINK-USD",
        "DOTUSD": "DOT-USD",
    }
    if pair_upper in crypto_bases:
        return crypto_bases[pair_upper]

    # Commodities - use spot FX rates
    commodity_map = {
        "XAUUSD": "GC=F",  # Gold futures (more liquid than XAUUSD=X)
        "XAGUSD": "SI=F",  # Silver futures
    }
    if pair_upper in commodity_map:
        return commodity_map[pair_upper]

    # Standard FX pairs
    return pair_upper + "=X"


def _cap_intraday_range(start: str | None, end: str | None, interval: str) -> tuple[str, str]:
    start_dt = pd.Timestamp(start) if start else None
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.utcnow()
    if start_dt is not None and start_dt.tzinfo:
        start_dt = start_dt.tz_convert(None)
    if end_dt.tzinfo:
        end_dt = end_dt.tz_convert(None)
    if start_dt is None:
        start_dt = end_dt - pd.Timedelta(days=365)

    if interval.endswith("m"):
        max_days = 60  # Yahoo intraday minute bars are limited to ~60 days
    elif interval.endswith("h"):
        max_days = 730  # hourly bars limited to ~2 years
    else:
        max_days = None

    if max_days:
        min_start = end_dt - pd.Timedelta(days=max_days - 1)
        if start_dt < min_start:
            start_dt = min_start
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(out_path)


def download_pair(
        pair: str, start: str | None, end: str | None, interval: str, output_root: Path, skip_existing: bool
) -> tuple[str, Path | None, bool]:
    ticker = pair_to_ticker(pair)
    adj_start, adj_end = _cap_intraday_range(start, end, interval)
    out_dir = output_root / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_path = out_dir / f"{pair}_{interval}.csv"
    fallback_path = out_dir / f"{pair}_1d.csv" if interval != "1d" else primary_path

    if skip_existing and primary_path.exists():
        return pair, primary_path, True
    if skip_existing and interval != "1d" and fallback_path.exists():
        return pair, fallback_path, True

    df = yf.download(ticker, start=adj_start, end=adj_end, interval=interval, progress=False, group_by="ticker")
    if df.empty and interval != "1d":
        # Fallback to daily if intraday not available.
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, group_by="ticker")
        interval_used = "1d"
    else:
        interval_used = interval

    if df.empty:
        return pair, None

    df = df.reset_index()

    # Flatten MultiIndex columns from yfinance
    # With group_by='ticker': columns are like ('GBPUSD=X', 'Open') or ('Datetime', '')
    # We want the price name (Open, High, etc) not the ticker
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for c in df.columns:
            if c[1] == '' or c[1] is None:
                # Datetime column: ('Datetime', '')
                new_cols.append(c[0])
            else:
                # Price columns: ('GBPUSD=X', 'Open') -> 'Open'
                new_cols.append(c[1])
        df.columns = new_cols
    elif all(isinstance(c, tuple) for c in df.columns):
        # Tuple columns but not MultiIndex
        new_cols = []
        for c in df.columns:
            if c[1] == '' or c[1] is None:
                new_cols.append(c[0])
            else:
                new_cols.append(c[1])
        df.columns = new_cols

    # Normalize to HistData-compatible format
    # Expected columns: datetime,open,high,low,close,volume
    # Expected datetime format: YYYYMMDD HHMMSS

    # Rename columns to lowercase
    col_map = {
        'Datetime': 'datetime', 'Date': 'datetime',
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
        'Volume': 'volume', 'Adj Close': 'adj_close'
    }
    df = df.rename(columns=col_map)

    # Ensure datetime column exists and is properly formatted
    if 'datetime' in df.columns:
        # Convert to datetime if string
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Remove timezone info if present
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        # Convert to HistData format: YYYYMMDD HHMMSS
        df['datetime'] = df['datetime'].dt.strftime('%Y%m%d %H%M%S')
    else:
        raise ValueError(f"No datetime column found in downloaded data. Columns: {list(df.columns)}")

    # Ensure volume is integer (yfinance returns 0 for FX)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0).astype(int)
    else:
        df['volume'] = 0

    # Select only required columns in correct order
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required_cols if c in df.columns]]

    # Write atomically to avoid corrupted partial files if interrupted.
    out_path = out_dir / f"{pair}_{interval_used}.csv"
    _atomic_write_csv(df, out_path)
    return pair, out_path, False


def main():
    parser = argparse.ArgumentParser(description="Download FX data from yfinance.")
    parser.add_argument("--pairs-csv", default="", help="Path to pairs.csv (default: repo root or data/pairs.csv)")
    parser.add_argument("--pairs", default="",
                        help="Comma-separated pairs to override pairs file (e.g., eurusd,usdjpy)")
    parser.add_argument("--output-root", default="yfinance_output", help="Output directory")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", default="1h", help="Interval (e.g., 1d,1h,30m,15m,5m)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloads when output CSV already exists",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]
    else:
        pairs_csv = resolve_pairs_file(args.pairs_csv)
        pairs = load_pairs(pairs_csv)

    missing = []
    for pair in pairs:
        pair, path, skipped = download_pair(
            pair, args.start, args.end, args.interval, output_root, args.skip_existing
        )
        if path is None:
            missing.append(pair)
            print(f"[warn] {pair} returned no data")
        else:
            status = "[skip]" if skipped else "[ok]"
            print(f"{status} {pair} -> {path}")

    if missing:
        print(f"No data for pairs: {', '.join(missing)}")


if __name__ == "__main__":
    main()
