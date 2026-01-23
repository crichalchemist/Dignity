"""Utilities for aligning candle times to GDELT regime buckets."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import get_gdelt_bucket_minutes


def align_candle_to_regime(
        candle_time: datetime, bucket_minutes: int | None = None
) -> datetime:
    """Floor candle_time to the previous full regime bucket using the configured default.

    When ``bucket_minutes`` is not provided, the interval from ``config.py``
    (``GDELT_TIME_DELTA_MINUTES``, currently 15 minutes) is used so aligners
    and iterators remain consistent.
    """
    if bucket_minutes is None:
        bucket_minutes = get_gdelt_bucket_minutes()
    candle_time = candle_time.astimezone(timezone.utc)
    minutes = (candle_time.minute // bucket_minutes) * bucket_minutes
    aligned = candle_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    if aligned > candle_time:
        aligned -= timedelta(minutes=bucket_minutes)
    return aligned


def iter_gdelt_buckets(
        start: datetime, end: datetime, bucket_minutes: int | None = None
) -> list[datetime]:
    """Generate regime bucket start times between start and end inclusive.

    The bucket size defaults to :func:`get_gdelt_bucket_minutes` to match the configured
    ingestion cadence.
    """
    bucket_minutes = bucket_minutes or get_gdelt_bucket_minutes()
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end < start:
        raise ValueError("end must be after start")
    buckets = []
    current = align_candle_to_regime(start, bucket_minutes=bucket_minutes)
    while current <= end:
        buckets.append(current)
        current += timedelta(minutes=bucket_minutes)
    return buckets
