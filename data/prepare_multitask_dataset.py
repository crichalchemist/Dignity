"""
Utility to load HistData CSVs (Central time), build features, and create train/val/test datasets
for the multi-task model.

This script prepares data for multi-task learning with auxiliary objectives including:
- Primary task: Price movement prediction (up/down/flat)
- Auxiliary tasks: Top-K future returns, sell-now decisions, volatility direction

Features:
- ✅ Data validation (OHLC integrity, duplicate removal, NaN handling)
- ✅ Feature caching for faster iteration
- ✅ Support for both ZIP and CSV file inputs
- ✅ Intrinsic time bars via directional-change events
- ✅ Optional GDELT sentiment integration
- ✅ Configurable feature groups and technical indicators

Example:
  python data/prepare_multitask_dataset.py --pairs gbpusd,eurusd --t-in 120 --t-out 10 --top-k 3
  python data/prepare_multitask_dataset.py --pairs btcusd --intrinsic-time --dc-threshold-up 0.002
"""

import argparse
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from utils.datetime_utils import convert_to_utc_and_dedup

# Local utilities
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import FeatureConfig, MultiTaskDataConfig
from data.agents.multitask_agent import MultiTaskDataAgent
from data.gdelt_ingest import load_gdelt_gkg
from train.features.agent_features import build_feature_frame
from train.features.agent_sentiment import aggregate_sentiment, attach_sentiment_features
from train.features.intrinsic_time import build_intrinsic_time_bars


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Validate and sanitize raw price data."""
    logger.info(f"Validating DataFrame with {len(df)} rows and columns: {list(df.columns)}")

    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check and convert datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        logger.warning("Converting datetime column to datetime64[ns]")
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Remove NaN rows for critical columns
    initial_len = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} NaN rows from OHLC data")

    # Detect and remove duplicate timestamps
    dups = df.duplicated(subset=["datetime"], keep="first")
    if dups.any():
        logger.warning(f"Removing {dups.sum()} duplicate timestamps")
        df = df[~dups].reset_index(drop=True)

    # Validate OHLC relationships
    invalid_ohlc = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"]) | (
            df["low"] > df["open"]) | (df["low"] > df["close"])
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        logger.warning(f"Removing {invalid_count} rows with invalid OHLC relationships")
        df = df[~invalid_ohlc].reset_index(drop=True)

    # Sort by datetime and reset index
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Validation complete. DataFrame reduced to {len(df)} valid rows")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for multi-task training.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes (e.g., gbpusd,eurusd)")
    parser.add_argument("--input-root", default="output_central",
                        help="Root directory containing pair subfolders with zips")
    parser.add_argument("--years", default=None,
                        help="Comma-separated list of years to include (e.g., 2018,2019). Default: all")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--t-out", type=int, default=10, help="Forecast horizon in minutes")
    parser.add_argument("--lookahead-window", type=int, default=None, help="Lookahead window for auxiliary tasks")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K future returns/prices to supervise")
    parser.add_argument("--predict-sell-now", action="store_true", help="Whether to supervise a sell-now decision")
    parser.add_argument("--flat-threshold", type=float, default=0.0001, help="Flat class threshold for log returns")
    parser.add_argument("--vol-min-change", type=float, default=0.0,
                        help="Volatility delta threshold for vol direction label")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train fraction (time-ordered)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (time-ordered)")
    parser.add_argument(
        "--intrinsic-time",
        action="store_true",
        help="Convert minute bars to intrinsic-time bars via directional-change events.",
    )
    parser.add_argument(
        "--dc-threshold-up",
        type=float,
        default=0.001,
        help="Fractional increase needed to flag an upward directional change (e.g., 0.001=0.1%).",
    )
    parser.add_argument(
        "--dc-threshold-down",
        type=float,
        default=None,
        help="Fractional decrease needed to flag a downward directional change. Defaults to dc-threshold-up.",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Include news sentiment features (requires GDELT data). Adds sentiment_score, sentiment_5min, sentiment_15min, sentiment_60min columns.",
    )
    parser.add_argument(
        "--use-bigquery-gdelt",
        action="store_true",
        help="Use BigQuery to fetch GDELT data (recommended for Colab). Requires Google Cloud authentication.",
    )
    parser.add_argument(
        "--gdelt-themes",
        default=None,
        help="Comma-separated list of GDELT themes to filter (e.g., ECON_CURRENCY,TAX_FNCACT). Default: all financial themes.",
    )
    parser.add_argument("--gdelt-path", default=None,
                        help="Path to GDELT GKG file or directory of .zip files (only used if --use-bigquery-gdelt is NOT set)")
    parser.add_argument("--gdelt-tz", default="UTC", help="Timezone to convert GDELT timestamps to (default: UTC)")

    # Cognee Cloud knowledge graph integration
    parser.add_argument("--use-cognee", action="store_true", help="Enable Cognee Cloud knowledge graph features")
    parser.add_argument("--cognee-api-key", default=None, help="Cognee API key (or set COGNEE_API_KEY env var)")
    parser.add_argument("--cognee-dataset", default=None, help="Cognee dataset name (default: fx_{pair})")
    parser.add_argument("--cognee-rebuild-graph", action="store_true",
                        help="Force rebuild of knowledge graph (default: use cache)")
    parser.add_argument("--include-economic-indicators", action="store_true",
                        help="Download and ingest economic indicator data into Cognee")
    parser.add_argument("--include-price-narratives", action="store_true",
                        help="Generate price pattern text descriptions for Cognee")
    parser.add_argument("--cognee-entity-window", type=int, default=24,
                        help="Entity mention lookback window in hours (default: 24)")
    parser.add_argument("--cognee-event-window", type=int, default=48,
                        help="Event proximity lookback window in hours (default: 48)")

    parser.add_argument("--feature-groups", default="all", help="Comma-separated feature groups to include or 'all'")
    parser.add_argument("--exclude-feature-groups", default=None, help="Comma-separated feature groups to drop")
    parser.add_argument("--sma-windows", default="10,20,50", help="Comma-separated SMA window lengths")
    parser.add_argument("--ema-windows", default="10,20,50", help="Comma-separated EMA spans")
    parser.add_argument("--rsi-window", type=int, default=14, help="Window length for RSI")
    parser.add_argument("--bollinger-window", type=int, default=20, help="Window length for Bollinger bands")
    parser.add_argument("--bollinger-num-std", type=float, default=2.0, help="Std dev multiplier for Bollinger bands")
    parser.add_argument("--atr-window", type=int, default=14, help="Window length for ATR")
    parser.add_argument("--short-vol-window", type=int, default=10, help="Short window for volatility clustering")
    parser.add_argument("--long-vol-window", type=int, default=50, help="Long window for volatility clustering")
    parser.add_argument("--spread-windows", default="20", help="Comma-separated windows for normalized spread stats")
    parser.add_argument("--imbalance-smoothing", type=int, default=5,
                        help="Rolling mean window for wick/body imbalance")
    return parser.parse_args()


def _load_pair_data(pair: str, input_root: Path, years: list[str] | None) -> pd.DataFrame:
    pair_dir = input_root / pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"No data folder for pair {pair} under {input_root}")

    zips = sorted(pair_dir.glob("*.zip"))
    csvs = sorted(pair_dir.glob("*.csv"))
    if years:
        zips = [z for z in zips if any(y in z.name for y in years)]
        csvs = [c for c in csvs if any(y in c.name for y in years)]
    if not zips and not csvs:
        raise FileNotFoundError(f"No data files found for pair {pair} with years={years}")

    frames = []
    for zpath in zips:
        with ZipFile(zpath) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                continue
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["datetime", "open", "high", "low", "close", "volume"],
                )
                df["source_file"] = zpath.name
                frames.append(df)
    for cpath in csvs:
        try:
            df = pd.read_csv(cpath)
            if {"datetime", "open", "high", "low", "close", "volume"}.issubset(df.columns):
                df = df[["datetime", "open", "high", "low", "close", "volume"]]
            else:
                # Fallback to legacy format without header; assume comma-delimited.
                df = pd.read_csv(
                    cpath,
                    header=None,
                    names=["datetime", "open", "high", "low", "close", "volume"],
                )
            df["source_file"] = cpath.name
            frames.append(df)
        except Exception as exc:
            logger.warning("[warn] failed to load CSV %s: %s", cpath, exc)
    if not frames:
        raise RuntimeError("No CSV data loaded; check zip contents.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["datetime"], format="%Y%m%d %H%M%S")
    full_df = full_df.sort_values("datetime").reset_index(drop=True)

    # Validate the data after loading
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    full_df = validate_dataframe(full_df, required_cols)

    return full_df


def _compute_time_ranges(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    train_start_ts = df["datetime"].iloc[0]
    val_start_ts = df["datetime"].iloc[train_end] if train_end < n else df["datetime"].iloc[-1]
    test_start_ts = df["datetime"].iloc[val_end] if val_end < n else df["datetime"].iloc[-1]
    train_range = (train_start_ts.isoformat(), df["datetime"].iloc[train_end - 1].isoformat())
    val_range = (val_start_ts.isoformat(),
                 df["datetime"].iloc[val_end - 1].isoformat() if val_end < n else df["datetime"].iloc[-1].isoformat())
    test_range = (test_start_ts.isoformat(), df["datetime"].iloc[-1].isoformat())
    return train_range, val_range, test_range


def process_pair(pair: str, args):
    years = args.years.split(",") if args.years else None
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (ROOT / input_root).resolve()

    raw_df = _load_pair_data(pair, input_root, years)
    # ---------------------------------------------------------------------
    # 1️⃣  Ensure timestamps are UTC and deduplicate any overlapping rows.
    # ---------------------------------------------------------------------
    # HistData timestamps are in Central Time (America/Chicago). Converting
    # them to UTC provides a single source of truth across the pipeline.
    # ---------------------------------------------------------------------
    raw_df = convert_to_utc_and_dedup(raw_df, datetime_col="datetime")

    include_groups = None if args.feature_groups.lower() == "all" else [g.strip() for g in
                                                                        args.feature_groups.split(",") if g.strip()]
    exclude_groups = (
        [g.strip() for g in args.exclude_feature_groups.split(",") if g.strip()]
        if args.exclude_feature_groups
        else None
    )
    feature_cfg = FeatureConfig(
        sma_windows=[int(x) for x in args.sma_windows.split(",") if x.strip()],
        ema_windows=[int(x) for x in args.ema_windows.split(",") if x.strip()],
        rsi_window=args.rsi_window,
        bollinger_window=args.bollinger_window,
        bollinger_num_std=args.bollinger_num_std,
        atr_window=args.atr_window,
        short_vol_window=args.short_vol_window,
        long_vol_window=args.long_vol_window,
        spread_windows=[int(x) for x in args.spread_windows.split(",") if x.strip()],
        imbalance_smoothing=args.imbalance_smoothing,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )

    # ---------------------------------------------------------------------
    # 2️⃣  Apply intrinsic time bars if requested
    # ---------------------------------------------------------------------
    df_for_features = raw_df
    if args.intrinsic_time:
        df_for_features = build_intrinsic_time_bars(
            raw_df,
            up_threshold=args.dc_threshold_up,
            down_threshold=args.dc_threshold_down or args.dc_threshold_up,
        )
        logger.info(
            "[intrinsic] reduced %d -> %d bars using DC thresholds up=%s, down=%s",
            len(raw_df),
            len(df_for_features),
            args.dc_threshold_up,
            args.dc_threshold_down or args.dc_threshold_up,
        )

    # ---------------------------------------------------------------------
    # 3️⃣  Cache heavy feature computation to avoid re‑running on every call.
    # ---------------------------------------------------------------------
    cache_dir = ROOT / "output_central" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import hashlib
    # Hash the raw data values (including index) – ``hash_pandas_object`` returns a
    # Series of uint64 hashes; converting to bytes yields a deterministic digest.
    raw_hash_bytes = pd.util.hash_pandas_object(df_for_features, index=True).values.tobytes()
    hash_input = (
            raw_hash_bytes
            + str(feature_cfg).encode()
            + (b"intrinsic" if args.intrinsic_time else b"regular")
    )
    cache_hash = hashlib.sha256(hash_input).hexdigest()[:12]
    cache_path = cache_dir / f"{pair}_multitask_features_{cache_hash}.feather"

    if cache_path.exists():
        feature_df = pd.read_feather(cache_path)
        logger.info("[cache] loaded pre‑computed features from %s", cache_path)
    else:
        feature_df = build_feature_frame(df_for_features, config=feature_cfg)
        feature_df.to_feather(cache_path)
        logger.info("[cache] saved computed features to %s", cache_path)

    feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])

    # ---------------------------------------------------------------------
    # 4️⃣  Attach sentiment features if requested
    # ---------------------------------------------------------------------
    if args.include_sentiment:
        if args.use_bigquery_gdelt:
            # Use BigQuery to fetch GDELT data (recommended for Colab)
            try:
                from data.gdelt_bigquery import FINANCIAL_THEMES, query_gdelt_for_date_range

                logger.info("[sentiment] querying GDELT data from BigQuery...")

                # Parse themes if provided
                themes = None
                if args.gdelt_themes:
                    themes = [t.strip() for t in args.gdelt_themes.split(',') if t.strip()]
                else:
                    themes = FINANCIAL_THEMES  # Use default financial themes

                # Query GDELT for the same date range as our feature data
                start_dt = feature_df['datetime'].min()
                end_dt = feature_df['datetime'].max()

                logger.info(f"[sentiment] date range: {start_dt} to {end_dt}")
                logger.info(f"[sentiment] themes: {themes}")

                gdelt_df = query_gdelt_for_date_range(
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    themes=themes
                )
                logger.info("[sentiment] retrieved %d GDELT records from BigQuery", len(gdelt_df))

                logger.info("[sentiment] aggregating sentiment to bar frequency...")
                sent_feats = aggregate_sentiment(
                    gdelt_df,
                    feature_df,
                    time_col="datetime",
                    score_col="sentiment_score"
                )
                logger.info("[sentiment] created %d sentiment features", len(sent_feats.columns))

                feature_df = attach_sentiment_features(feature_df, sent_feats)
                logger.info("[sentiment] successfully merged %d sentiment features to feature frame",
                            len(sent_feats.columns))
            except Exception as e:
                logger.warning(f"[sentiment] failed to query GDELT from BigQuery: {e}. Continuing without sentiment.")
        else:
            # Use local GDELT files (legacy approach)
            if not args.gdelt_path:
                logger.warning(
                    "[sentiment] --include-sentiment specified but --gdelt-path not provided. Skipping sentiment features.")
            else:
                try:
                    logger.info("[sentiment] loading GDELT data from %s", args.gdelt_path)
                    gdelt_df = load_gdelt_gkg(Path(args.gdelt_path), target_tz=args.gdelt_tz)
                    logger.info("[sentiment] loaded %d GDELT records", len(gdelt_df))

                    logger.info("[sentiment] aggregating sentiment to bar frequency...")
                    sent_feats = aggregate_sentiment(
                        gdelt_df,
                        feature_df,
                        time_col="datetime",
                        score_col="sentiment_score"
                    )
                    logger.info("[sentiment] created %d sentiment features", len(sent_feats.columns))

                    feature_df = attach_sentiment_features(feature_df, sent_feats)
                    logger.info("[sentiment] successfully merged %d sentiment features to feature frame",
                                len(sent_feats.columns))
                except Exception as e:
                    logger.warning(
                        f"[sentiment] failed to attach sentiment features: {e}. Continuing without sentiment.")

    # Cognee Cloud knowledge graph integration
    if args.use_cognee:
        logger.info("[cognee] Cognee Cloud integration enabled")

        try:
            import os

            from data.cognee_client import CogneeClient
            from data.cognee_processor import CogneeDataProcessor
            from features.cognee_features import build_cognee_features

            # Initialize Cognee client
            api_key = args.cognee_api_key or os.getenv("COGNEE_API_KEY")
            if not api_key:
                logger.error("[cognee] API key not provided. Set COGNEE_API_KEY env var or use --cognee-api-key")
                logger.warning("[cognee] Skipping Cognee features")
            else:
                client = CogneeClient(api_key=api_key)
                processor = CogneeDataProcessor(client)

                # Determine dataset name
                dataset_name = args.cognee_dataset or f"fx_{pair}"
                logger.info(f"[cognee] Using dataset: {dataset_name}")

                # Rebuild graph if requested
                if args.cognee_rebuild_graph:
                    logger.info("[cognee] Rebuilding knowledge graph...")

                    # Ingest GDELT news (if we have it from sentiment integration)
                    if args.include_sentiment and 'gdelt_df' in locals():
                        logger.info("[cognee] Ingesting GDELT news into knowledge graph...")
                        processor.ingest_gdelt_news(gdelt_df, dataset_name)

                    # Ingest economic indicators if requested
                    if args.include_economic_indicators:
                        logger.info("[cognee] Downloading and ingesting economic indicators...")
                        try:
                            from data.downloaders.economic_indicators import (
                                download_forex_fred_bundle,
                            )

                            start_dt = feature_df['datetime'].min()
                            end_dt = feature_df['datetime'].max()

                            indicators_df = download_forex_fred_bundle(
                                start_date=start_dt.strftime('%Y-%m-%d'),
                                end_date=end_dt.strftime('%Y-%m-%d'),
                                api_key=os.getenv("FRED_API_KEY")
                            )

                            if len(indicators_df) > 0:
                                processor.ingest_economic_indicators(indicators_df, dataset_name)
                            else:
                                logger.warning("[cognee] No economic indicators data downloaded")

                        except Exception as e:
                            logger.warning(f"[cognee] Failed to download economic indicators: {e}")

                    # Ingest price pattern narratives if requested
                    if args.include_price_narratives:
                        logger.info("[cognee] Generating and ingesting price pattern narratives...")
                        try:
                            from data.price_pattern_narrator import generate_pattern_text

                            pattern_df = generate_pattern_text(
                                df=feature_df,
                                pair=pair.upper()
                            )

                            processor.ingest_price_patterns(pattern_df, dataset_name, pair.upper())

                        except Exception as e:
                            logger.warning(f"[cognee] Failed to generate price narratives: {e}")

                    # Trigger cognify
                    logger.info("[cognee] Triggering knowledge graph building...")
                    job_id = client.cognify(dataset_name)
                    success = processor.wait_for_cognify(job_id, timeout=600)

                    if not success:
                        logger.error("[cognee] Knowledge graph building failed or timed out")
                        logger.warning("[cognee] Skipping Cognee features")
                    else:
                        logger.info("[cognee] ✅ Knowledge graph built successfully")

                # Extract features from knowledge graph
                if not args.cognee_rebuild_graph or success:
                    logger.info("[cognee] Extracting features from knowledge graph...")

                    cognee_features = build_cognee_features(
                        client=client,
                        price_df=feature_df,
                        pair=pair,
                        dataset_name=dataset_name,
                        cache_dir=cache_dir / "cognee",
                        entity_window_hours=args.cognee_entity_window,
                        event_window_hours=args.cognee_event_window
                    )

                    # Merge Cognee features with existing features
                    feature_df = pd.concat([feature_df, cognee_features], axis=1)
                    logger.info(f"[cognee] ✅ Added {len(cognee_features.columns)} Cognee features")

        except ImportError as e:
            logger.error(f"[cognee] Failed to import Cognee modules: {e}")
            logger.warning("[cognee] Skipping Cognee features")
        except Exception as e:
            logger.error(f"[cognee] Cognee integration failed: {e}")
            logger.warning("[cognee] Continuing without Cognee features")

    train_range, val_range, test_range = _compute_time_ranges(
        feature_df, args.train_ratio, args.val_ratio
    )

    feature_cols = [c for c in feature_df.columns if c not in {"datetime", "source_file"}]

    data_cfg = MultiTaskDataConfig(
        csv_path="",
        datetime_column="datetime",
        feature_columns=feature_cols,
        t_in=args.t_in,
        t_out=args.t_out,
        lookahead_window=args.lookahead_window,
        top_k_predictions=args.top_k,
        predict_sell_now=args.predict_sell_now,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        flat_threshold=args.flat_threshold,
        vol_min_change=args.vol_min_change,
    )

    agent = MultiTaskDataAgent(data_cfg)
    datasets = agent.build_datasets(feature_df)
    loaders = MultiTaskDataAgent.build_dataloaders(datasets, batch_size=64)

    logger.info("Pair: %s", pair)
    logger.info("Rows after features: %s", f"{len(feature_df):,}")
    logger.info("Features: %s -> %s", len(feature_cols), feature_cols)
    for split, ds in datasets.items():
        logger.info("%s: %s windows", split, f"{len(ds):,}")
    logger.info("Dataloaders ready (train/val/test).")
    return pair, loaders


def main():
    # Ensure reproducibility.
    set_seed()
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]
    results = {}
    for pair in pairs:
        try:
            _, loaders = process_pair(pair, args)
            results[pair] = loaders
        except Exception as e:
            logger.error("[error] Failed to process %s: %s", pair, e)
    return results


if __name__ == "__main__":
    main()
