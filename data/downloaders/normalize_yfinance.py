"""
Normalize Yahoo Finance FX data to synthetic 1-minute bars by forward-filling closes.

This up-samples higher-interval OHLC data (e.g., 1h) to 1-minute bars by:
  - forward-filling Close values
  - setting Open/High/Low to the filled close
  - setting Volume to 0 (intraday volume not available from yfinance FX)

Output columns match HistData schema: datetime, open, high, low, close, volume.
"""

import argparse
from pathlib import Path

import pandas as pd


def normalize_file(csv_path: Path, output_root: Path) -> Path | None:
    df = pd.read_csv(csv_path)

    # Identify datetime column: yfinance writes the index with no name (Unnamed: 0).
    dt_col = None
    for candidate in df.columns:
        if candidate.lower() in {"datetime", "date"} or candidate.startswith("Unnamed"):
            dt_col = candidate
            break
    if dt_col is None:
        print(f"[warn] No datetime column found in {csv_path}, skipping")
        return None

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).drop_duplicates(subset=dt_col)
    if df.empty:
        print(f"[warn] Empty dataframe for {csv_path}, skipping")
        return None

    df = df.set_index(dt_col)
    close = df["Close"].ffill()
    if close.isna().all():
        print(f"[warn] No Close data in {csv_path}, skipping")
        return None

    idx = pd.date_range(df.index[0], df.index[-1], freq="1min")
    close_1m = close.reindex(idx).ffill()

    resampled = pd.DataFrame(index=idx)
    resampled["open"] = close_1m
    resampled["high"] = close_1m
    resampled["low"] = close_1m
    resampled["close"] = close_1m
    resampled["volume"] = 0

    resampled = resampled.reset_index().rename(columns={"index": "datetime"})

    pair = csv_path.parent.name
    out_dir = output_root / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_m1.csv"
    resampled.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Upsample yfinance FX data to synthetic 1-minute bars.")
    parser.add_argument("--input-root", default="yfinance_output", help="Root directory of yfinance CSVs")
    parser.add_argument("--output-root", default="yfinance_output_m1", help="Root directory for normalized CSVs")
    parser.add_argument("--pairs", default="", help="Comma-separated pair filter (default: all pairs found)")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    pair_filter: list[str] = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    if not input_root.exists():
        raise FileNotFoundError(f"Input root {input_root} not found")

    csv_files = sorted(input_root.rglob("*.csv"))
    if pair_filter:
        csv_files = [p for p in csv_files if p.parent.name.lower() in pair_filter]

    if not csv_files:
        print("No CSV files found to normalize.")
        return

    for csv_path in csv_files:
        out = normalize_file(csv_path, output_root)
        if out:
            print(f"[ok] {csv_path} -> {out}")


if __name__ == "__main__":
    main()
