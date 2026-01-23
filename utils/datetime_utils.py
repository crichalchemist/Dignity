"""Timezone handling and duplicate‑row utilities.

Historical price data (HistData) ships timestamps in US Central Time.  The
pipeline must operate in UTC to avoid daylight‑saving surprises and to provide a
single reference across all agents.  This module offers a small helper that
converts a ``DataFrame`` containing a ``datetime`` column to UTC‑aware timestamps
and removes any duplicate rows that share the same timestamp.

Both operations are pure – they return a new ``DataFrame`` and never mutate the
input.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

__all__ = ["convert_to_utc_and_dedup"]


def convert_to_utc_and_dedup(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Return a UTC‑aware ``DataFrame`` with duplicate timestamps removed.

    Parameters
    ----------
    df:
        Input ``DataFrame`` that must contain a column named ``datetime_col``.
    datetime_col:
        Name of the column holding the original timestamps.  The column is
        expected to be a string in the ``%Y%m%d %H%M%S`` format used by HistData.

    The function performs three steps:
    1. Parse strings to ``datetime`` objects.
    2. Localise to the ``America/Chicago`` (Central) timezone and convert to UTC.
    3. Drop duplicate timestamps, keeping the first occurrence.

    Returns a new ``DataFrame`` with the same columns (including ``datetime``) but
    with the ``datetime`` column as a timezone‑aware ``datetime64[ns, UTC]``.
    """
    # 1️⃣ Parse.
    parsed = pd.to_datetime(df[datetime_col], format="%Y%m%d %H%M%S", errors="raise")
    # 2️⃣ Localise to Central and convert to UTC.
    central = parsed.dt.tz_localize(ZoneInfo("America/Chicago"), ambiguous="NaT")
    utc = central.dt.tz_convert(ZoneInfo("UTC"))
    df_copy = df.copy()
    df_copy[datetime_col] = utc
    # 3️⃣ Remove exact duplicate timestamps (keep first).
    deduped = df_copy.drop_duplicates(subset=datetime_col, keep="first")
    # Preserve original order after deduplication.
    deduped = deduped.sort_values(datetime_col).reset_index(drop=True)
    return deduped
