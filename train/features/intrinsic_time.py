"""Directional-change and intrinsic-time bar utilities.

The functions here are pure, stateless helpers to compute directional-change
(DC) events and overshoots from a price series, then convert those events into
intrinsic-time bars that can replace fixed clock-time bars during dataset
preparation.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from config.constants import DEFAULT_DC_THRESHOLD, MAX_THRESHOLD_VALUE


def _validate_thresholds(up_threshold: float, down_threshold: float) -> None:
    if up_threshold <= 0 or down_threshold <= 0:
        raise ValueError("Directional-change thresholds must be positive.")
    if up_threshold > MAX_THRESHOLD_VALUE or down_threshold > MAX_THRESHOLD_VALUE:
        raise ValueError(
            f"Directional-change thresholds should be fractional "
            f"(e.g., 0.001 for 0.1%, not {MAX_THRESHOLD_VALUE} or greater)"
        )


def detect_directional_changes(
        prices: pd.Series,
        up_threshold: float,
        down_threshold: float | None = None,
        timestamps: Sequence | None = None,
) -> pd.DataFrame:
    """
    Detect directional-change events and overshoots using relative price moves.

    Args:
        prices: Series of prices ordered in time.
        up_threshold: Fractional increase required to flag an upward directional
            change (e.g., DEFAULT_DC_THRESHOLD == 0.1%).
        down_threshold: Fractional decrease required to flag a downward
            directional change. Defaults to ``up_threshold`` if omitted.
        timestamps: Optional timestamps aligned with ``prices``. If omitted, the
            price index is used.

    Returns:
        DataFrame with columns ``["idx", "timestamp", "price", "direction",
        "overshoot"]``. The first DC starts when either threshold is breached;
        overshoot tracks any additional favorable move until the next reversal
        triggers.
    """

    if prices.empty:
        raise ValueError("Price series must contain at least one observation.")
    if prices.isna().any():
        raise ValueError("Price series contains NaN values. Please clean the data first.")
    if (prices <= 0).any():
        raise ValueError("Price series contains non-positive values. All prices must be positive.")
    if down_threshold is None:
        down_threshold = up_threshold
    _validate_thresholds(up_threshold, down_threshold)

    ts = pd.Series(timestamps) if timestamps is not None else prices.index

    extreme_price = float(prices.iloc[0])
    current_direction: str | None = None
    event_price = extreme_price

    events = []

    def append_event(idx: int, price: float, direction: str, overshoot: float = 0.0) -> None:
        events.append(
            {
                "idx": idx,
                "timestamp": ts.iloc[idx] if hasattr(ts, "iloc") else ts[idx],
                "price": price,
                "direction": direction,
                "overshoot": overshoot,
            }
        )

    for i in range(1, len(prices)):
        price = float(prices.iloc[i])

        if current_direction is None:
            change = (price - extreme_price) / extreme_price
            if change >= up_threshold:
                current_direction = "up"
                event_price = price
                extreme_price = price
                append_event(i, price, current_direction)
            elif change <= -down_threshold:
                current_direction = "down"
                event_price = price
                extreme_price = price
                append_event(i, price, current_direction)
            continue

        if current_direction == "up":
            if price > extreme_price:
                extreme_price = price
                overshoot = (extreme_price - event_price) / event_price
                events[-1]["overshoot"] = overshoot
            drawdown = (price - extreme_price) / extreme_price
            if drawdown <= -down_threshold:
                current_direction = "down"
                event_price = price
                extreme_price = price
                append_event(i, price, current_direction)
        else:
            if price < extreme_price:
                extreme_price = price
                overshoot = (event_price - extreme_price) / event_price
                events[-1]["overshoot"] = overshoot
            rebound = (price - extreme_price) / extreme_price
            if rebound >= up_threshold:
                current_direction = "up"
                event_price = price
                extreme_price = price
                append_event(i, price, current_direction)

    return pd.DataFrame(events)


def build_intrinsic_time_bars(
        df: pd.DataFrame,
        price_col: str = "close",
        datetime_col: str = "datetime",
        up_threshold: float = DEFAULT_DC_THRESHOLD,
        down_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Convert a clock-time price frame into intrinsic-time bars using DC events.

    Args:
        df: DataFrame with at least ``price_col`` and ``datetime_col`` columns,
            ordered by time.
        price_col: Column containing prices.
        datetime_col: Timestamp column aligned to prices.
        up_threshold: Fractional increase needed for an upward DC.
        down_threshold: Fractional decrease needed for a downward DC. Defaults to
            ``up_threshold`` if omitted.

    Returns:
        A copy of ``df`` filtered to directional-change event rows. Extra columns
        ``direction`` and ``overshoot`` annotate the DC direction and overshoot
        magnitude observed before reversal.
    """

    events = detect_directional_changes(
        df[price_col].reset_index(drop=True),
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        timestamps=df[datetime_col].reset_index(drop=True),
    )
    if events.empty:
        raise ValueError("No directional-change events detected; consider lowering thresholds.")

    bars = df.reset_index(drop=True).iloc[events["idx"].to_list()].copy()
    bars["direction"] = events["direction"].to_list()
    bars["overshoot"] = events["overshoot"].to_list()
    return bars.reset_index(drop=True)


def add_intrinsic_time_features(
        df: pd.DataFrame,
        price_col: str = "close",
        up_threshold: float = DEFAULT_DC_THRESHOLD,
        down_threshold: float | None = None,
        timestamp_col: str = "datetime",
) -> pd.DataFrame:
    """
    Add intrinsic time features to existing dataframe without subsampling.

    This differs from build_intrinsic_time_bars() which filters rows.
    Instead, this annotates ALL rows with DC-based features:
      - dc_direction: Binary (1=up, 0=down, NaN=no DC yet)
      - dc_overshoot: Overshoot magnitude at last DC
      - dc_bars_since: Number of bars since last DC event
      - dc_event_flag: Binary flag marking DC event rows

    Args:
        df: DataFrame with OHLC data
        price_col: Column to use for DC detection (default: "close")
        up_threshold: Upward DC threshold (default: 0.001 = 0.1%)
        down_threshold: Downward DC threshold (default: same as up_threshold)
        timestamp_col: Timestamp column name

    Returns:
        DataFrame with added intrinsic time feature columns
    """
    if down_threshold is None:
        down_threshold = up_threshold

    _validate_thresholds(up_threshold, down_threshold)

    # Detect DC events
    events = detect_directional_changes(
        df[price_col].reset_index(drop=True),
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        timestamps=df[timestamp_col] if timestamp_col in df.columns else None,
    )

    if events.empty:
        # No DC events - add NaN columns
        df = df.copy()
        df["dc_direction"] = np.nan
        df["dc_overshoot"] = np.nan
        df["dc_bars_since"] = np.nan
        df["dc_event_flag"] = 0
        return df

    # Create feature columns
    df = df.copy()

    # Initialize all rows
    df["dc_direction"] = np.nan
    df["dc_overshoot"] = np.nan
    df["dc_bars_since"] = np.nan
    df["dc_event_flag"] = 0

    # Mark DC event rows and propagate features forward
    event_indices = events["idx"].to_list()

    for i, (_idx, row) in enumerate(events.iterrows()):
        event_idx = row["idx"]

        # Mark this row as DC event
        df.loc[event_idx, "dc_event_flag"] = 1
        df.loc[event_idx, "dc_direction"] = 1 if row["direction"] == "up" else 0
        df.loc[event_idx, "dc_overshoot"] = row["overshoot"]

        # Propagate to subsequent rows until next event
        next_event_idx = event_indices[i + 1] if i + 1 < len(event_indices) else len(df)

        for j in range(event_idx + 1, next_event_idx):
            if j < len(df):
                df.loc[j, "dc_direction"] = 1 if row["direction"] == "up" else 0
                df.loc[j, "dc_overshoot"] = row["overshoot"]
                df.loc[j, "dc_bars_since"] = j - event_idx

    # Fill bars_since for event rows
    df.loc[df["dc_event_flag"] == 1, "dc_bars_since"] = 0

    return df
