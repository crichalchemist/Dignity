"""Validate and prepare FX data for RL training.

This script checks data availability, validates format, and prepares
data for RL policy training with proper telemetry integration.

Usage:
    python scripts/validate_training_data.py --pair EURUSD --check-only
    python scripts/validate_training_data.py --pair EURUSD --prepare --output data/prepared/
"""

import argparse
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

from train.features.alpha_factors import build_alpha_feature_frame
from train.features.fx_patterns import build_fx_feature_frame
from train.features.multi_timeframe import add_daily_patterns, add_multi_timeframe_features
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate and prepare FX data for RL training."""

    def __init__(self, data_root: Path = Path("data")):
        """
        Args:
            data_root: Root directory for data files
        """
        self.data_root = data_root
        self.histdata_dir = data_root / "histdata"
        self.output_dir = data_root / "output"  # Common HistData download location

    def find_data_files(self, pair: str) -> list[Path]:
        """Find available data files for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'GBPUSD')

        Returns:
            List of data file paths found
        """
        pair_lower = pair.lower()
        pair_upper = pair.upper()

        search_paths = [
            self.histdata_dir / pair_lower,
            self.output_dir / pair_lower,
            self.data_root / pair_lower,
            self.data_root / pair_upper,
        ]

        found_files = []
        for search_path in search_paths:
            if search_path.exists():
                # Look for CSV and ZIP files
                csv_files = list(search_path.glob("*.csv"))
                zip_files = list(search_path.glob("*.zip"))
                found_files.extend(csv_files + zip_files)

        return found_files

    def validate_csv_format(self, csv_path: Path) -> dict[str, any]:
        """Validate CSV file format and content.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dict with validation results
        """
        try:
            # HistData format: semicolon-delimited, no header
            # Format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
            df = pd.read_csv(
                csv_path,
                sep=';',
                header=None,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                nrows=100
            )

            # Validate we got the expected columns
            if len(df.columns) != 6:
                return {
                    "valid": False,
                    "error": f"Expected 6 columns, got {len(df.columns)}",
                    "num_rows": 0,
                }

            # Read full file to get statistics
            df_full = pd.read_csv(
                csv_path,
                sep=';',
                header=None,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )

            return {
                "valid": True,
                "num_rows": len(df_full),
                "columns": ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                "date_range": (str(df_full.iloc[0]["DateTime"]), str(df_full.iloc[-1]["DateTime"])),
                "file_size_mb": csv_path.stat().st_size / (1024 * 1024),
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "num_rows": 0,
            }

    def prepare_for_training(
            self,
            csv_path: Path,
            output_path: Path,
            sequence_length: int = 390,
    ) -> dict[str, any]:
        """Prepare CSV data for RL training.

        Args:
            csv_path: Path to source CSV file
            output_path: Path to save prepared numpy array
            sequence_length: Length of each training episode in steps

        Returns:
            Dict with preparation statistics
        """
        try:
            logger.info(f"Loading data from {csv_path}")
            # HistData format: semicolon-delimited, no header
            df = pd.read_csv(
                csv_path,
                sep=';',
                header=None,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )

            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Parse datetime (format: YYYYMMDD HHMMSS)
            df["datetime"] = pd.to_datetime(df["datetime"], format='%Y%m%d %H%M%S')

            # Sort by time (HistData should already be sorted, but be safe)
            df = df.sort_values("datetime")

            # Calculate basic features (OHLC + returns)
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            df["mid_price"] = (df["high"] + df["low"]) / 2

            logger.info("Adding FX-specific features (sessions, S/R, patterns)...")
            # Add FX-specific features
            df = build_fx_feature_frame(
                df,
                include_sessions=True,
                include_support_resistance=True,
                support_resistance_lookback=100,
                include_trend_strength=True,
                trend_strength_window=14,
                include_patterns=True,
            )

            logger.info("Adding session-aware alpha factors...")
            # Add alpha factors
            df = build_alpha_feature_frame(
                df,
                include_session_momentum=True,
                session_momentum_window=20,
                include_cross_session_gap=True,
                include_volatility_regime=True,
                volatility_window=20,
                include_mean_reversion=True,
                mean_reversion_window=20,
            )

            logger.info("Adding multi-timeframe features (15min, 1h, 4h, daily)...")
            # Add multi-timeframe context features
            df = add_multi_timeframe_features(df, timeframes=['15min', '1h', '4h', '1D'])
            df = add_daily_patterns(df)
            logger.info("  Added 21 multi-timeframe features (16 TF + 5 daily patterns)")

            # Drop NaN rows (from rolling calculations)
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            logger.info(f"Dropped {dropped_rows} rows with NaN values (from feature calculations)")

            # Select feature columns for model input
            # Exclude datetime and intermediate calculation columns
            exclude_cols = [
                "datetime",
                # Intermediate S/R columns (we keep distance and flags instead)
                "resistance_100", "support_100",
                # Intermediate BB columns (we keep bb_pct and flags instead)
                "sma", "bb_std", "bb_upper", "bb_lower",
            ]

            feature_cols = [col for col in df.columns if col not in exclude_cols]
            logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}... (showing first 10)")

            # Create feature matrix
            features = df[feature_cols].values

            # Split into sequences of sequence_length
            num_sequences = len(features) // sequence_length
            trimmed_length = num_sequences * sequence_length

            features_trimmed = features[:trimmed_length]
            sequences = features_trimmed.reshape(num_sequences, sequence_length, -1)

            # Save to numpy array
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, sequences.astype(np.float32))

            stats = {
                "success": True,
                "num_rows": len(df),
                "num_sequences": num_sequences,
                "sequence_length": sequence_length,
                "num_features": sequences.shape[2],
                "output_file": str(output_path),
                "output_size_mb": output_path.stat().st_size / (1024 * 1024),
            }

            logger.info(f"Prepared {num_sequences} sequences of length {sequence_length}")
            logger.info(f"Saved to {output_path} ({stats['output_size_mb']:.2f} MB)")

            return stats

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare FX data for RL training")
    parser.add_argument(
        "--pair",
        type=str,
        default="EURUSD",
        help="Currency pair to validate (e.g., EURUSD, GBPUSD)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for data files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check data availability, don't prepare",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare data for training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared"),
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=390,
        help="Sequence length for training episodes (390 = ~6.5 hour trading day)",
    )

    args = parser.parse_args()

    validator = DataValidator(data_root=args.data_root)

    logger.info("=" * 80)
    logger.info("FX DATA VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Data root: {args.data_root}")
    logger.info("=" * 80)

    # Find data files
    logger.info(f"\nSearching for {args.pair} data files...")
    data_files = validator.find_data_files(args.pair)

    if not data_files:
        logger.error(f"❌ No data files found for {args.pair}")
        logger.info("\nTo download data:")
        logger.info("  1. Use HistData.com manual download")
        logger.info(f"  2. Place files in data/histdata/{args.pair.lower()}/")
        logger.info("  3. Or use: python data/download_all_fx_data.py")
        return 1

    logger.info(f"✅ Found {len(data_files)} data file(s):")
    for f in data_files:
        logger.info(f"  - {f}")

    # Validate each file
    logger.info("\nValidating file formats...")
    valid_files = []

    for data_file in data_files:
        if data_file.suffix == ".csv":
            result = validator.validate_csv_format(data_file)

            if result["valid"]:
                logger.info(f"✅ {data_file.name}")
                logger.info(f"     Rows: {result['num_rows']:,}")
                logger.info(f"     Size: {result['file_size_mb']:.2f} MB")
                logger.info(f"     Columns: {result['columns']}")
                valid_files.append(data_file)
            else:
                logger.error(f"❌ {data_file.name}: {result['error']}")
        else:
            logger.info(f"⚠️  {data_file.name} (ZIP file - extract first)")

    if not valid_files:
        logger.error("\n❌ No valid CSV files found")
        return 1

    logger.info(f"\n✅ {len(valid_files)} valid CSV file(s) ready for training")

    if args.check_only:
        logger.info("\n✅ Data validation complete (check-only mode)")
        return 0

    # Prepare data for training
    if args.prepare:
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING DATA FOR TRAINING")
        logger.info("=" * 80)

        for csv_file in valid_files:
            output_file = args.output / f"{args.pair.lower()}_{csv_file.stem}_prepared.npy"

            logger.info(f"\nPreparing {csv_file.name}...")
            result = validator.prepare_for_training(
                csv_path=csv_file,
                output_path=output_file,
                sequence_length=args.sequence_length,
            )

            if result["success"]:
                logger.info(f"✅ Prepared {result['num_sequences']} sequences")
                logger.info(f"     Output: {result['output_file']}")
                logger.info(f"     Size: {result['output_size_mb']:.2f} MB")
            else:
                logger.error(f"❌ Failed: {result['error']}")

        logger.info("\n✅ Data preparation complete!")
        logger.info(f"\nPrepared data saved to: {args.output}")
        logger.info("Next step: Run training with prepared data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
