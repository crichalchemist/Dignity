"""
Utilities for loading and processing GDELT GKG files into sentiment DataFrames.
"""
from pathlib import Path

import pandas as pd
from data.gdelt.parser import GDELTParser


def load_gdelt_gkg(
        gdelt_path: Path,
        target_tz: str = "UTC",
        score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Load GDELT GKG file(s) and extract sentiment tone scores.

    Args:
        gdelt_path: Path to a single GDELT .zip file or a directory of .zip files
        target_tz: Timezone to convert GDELT timestamps to (default: UTC)
        score_col: Name of the sentiment score column to create

    Returns:
        DataFrame with columns: datetime, sentiment_score

    Example:
        >>> gdelt_df = load_gdelt_gkg(Path("data/gdelt"), target_tz="UTC")
        >>> print(gdelt_df.head())
               datetime  sentiment_score
        0  2016-01-01 00:00:00          -2.5
        1  2016-01-01 00:15:00           1.2
    """
    parser = GDELTParser()

    # Handle both single file and directory
    if gdelt_path.is_file():
        paths = [gdelt_path]
    elif gdelt_path.is_dir():
        # Find all .zip files in directory
        paths = sorted(gdelt_path.glob("*.zip"))
        if not paths:
            raise ValueError(f"No .zip files found in {gdelt_path}")
    else:
        raise ValueError(f"GDELT path does not exist: {gdelt_path}")

    # Parse all records and extract tone scores
    records = []
    for record in parser.parse_files(paths):
        records.append({
            "datetime": record.datetime,
            score_col: record.tone.tone,  # GDELT tone score (-inf to +inf, typically -10 to +10)
        })

    if not records:
        raise ValueError(f"No GDELT records found in {gdelt_path}")

    # Create DataFrame and convert timezone
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(target_tz).dt.tz_localize(None)

    return df
