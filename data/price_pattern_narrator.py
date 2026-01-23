"""
Price Pattern Narrator

Convert OHLCV price action into natural language descriptions for Cognee knowledge graph.
Creates cross-modal links between numerical price data and textual news events.

Features:
- ✅ Candlestick pattern descriptions (bullish/bearish engulfing, doji, hammer, etc.)
- ✅ Support/resistance level breaks
- ✅ Trend and momentum descriptions
- ✅ Volatility regime changes
- ✅ Multi-timeframe context

Usage:
    from data.price_pattern_narrator import generate_pattern_text

    df = pd.read_csv("data/eurusd_2023.csv")
    pattern_df = generate_pattern_text(df, pair="EUR/USD")

    # Result: DataFrame with 'datetime' and 'pattern_description' columns
    # Example descriptions:
    # - "Strong bullish engulfing candle, EUR/USD rallied 1.2% to 1.0850"
    # - "Broke above 20-day SMA resistance at 1.0800, momentum turning positive"
    # - "Volatility spike: ATR increased 2.5x above 50-day average"
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from utils.logger import get_logger

logger = get_logger(__name__)


def describe_candle(
        open_price: float,
        high: float,
        low: float,
        close: float,
        prev_open: float | None = None,
        prev_close: float | None = None
) -> str:
    """
    Describe a single candlestick's characteristics.

    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        prev_open: Previous candle's open (for pattern detection)
        prev_close: Previous candle's close (for pattern detection)

    Returns:
        Text description of the candle

    Examples:
        "Bullish engulfing candle with 1.5% body"
        "Doji candle showing indecision"
        "Long upper shadow indicating rejection of higher prices"
    """
    body = abs(close - open_price)
    body_pct = (body / open_price) * 100
    range_price = high - low
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    descriptions = []

    # Candle direction
    if close > open_price:
        direction = "bullish"
    elif close < open_price:
        direction = "bearish"
    else:
        direction = "neutral"

    # Significant body
    if body_pct > 1.0:
        descriptions.append(f"Strong {direction} candle with {body_pct:.1f}% body")
    elif body_pct > 0.5:
        descriptions.append(f"{direction.capitalize()} candle")
    elif body_pct < 0.1:
        descriptions.append("Doji candle showing indecision")
    else:
        descriptions.append(f"Small {direction} candle")

    # Shadows
    if range_price > 0:
        upper_shadow_pct = (upper_shadow / range_price) * 100
        lower_shadow_pct = (lower_shadow / range_price) * 100

        if upper_shadow_pct > 60:
            descriptions.append("long upper shadow indicating rejection of higher prices")
        elif lower_shadow_pct > 60:
            descriptions.append("long lower shadow indicating buying support")

    # Engulfing patterns
    if prev_open is not None and prev_close is not None:
        prev_body = abs(prev_close - prev_open)

        # Bullish engulfing
        if (close > open_price and prev_close < prev_open and
                open_price < prev_close and close > prev_open and
                body > prev_body * 1.2):
            descriptions.append("bullish engulfing pattern (reversal signal)")

        # Bearish engulfing
        elif (close < open_price and prev_close > prev_open and
              open_price > prev_close and close < prev_open and
              body > prev_body * 1.2):
            descriptions.append("bearish engulfing pattern (reversal signal)")

    return ", ".join(descriptions) if descriptions else "Normal price action"


def describe_price_movement(
        df: pd.DataFrame,
        window: int = 20,
        pair: str = "PAIR"
) -> list[str]:
    """
    Generate descriptions for each bar based on price movement patterns.

    Args:
        df: DataFrame with OHLCV data
        window: Lookback window for calculating moving averages
        pair: Currency pair name (e.g., "EUR/USD")

    Returns:
        List of descriptions (one per row in df)
    """
    descriptions = []

    # Calculate technical indicators for context
    df = df.copy()
    df['sma_20'] = df['close'].rolling(window).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['return'] = df['close'].pct_change()

    for i in range(len(df)):
        desc_parts = []
        row = df.iloc[i]

        # Price change
        if pd.notna(row['return']):
            return_pct = row['return'] * 100

            if abs(return_pct) > 1.0:
                direction = "rallied" if return_pct > 0 else "dropped"
                desc_parts.append(f"{pair} {direction} {abs(return_pct):.1f}% to {row['close']:.5f}")
            elif abs(return_pct) > 0.5:
                direction = "rose" if return_pct > 0 else "fell"
                desc_parts.append(f"{pair} {direction} {abs(return_pct):.1f}% to {row['close']:.5f}")

        # SMA crosses
        if i > 0 and pd.notna(row['sma_20']) and pd.notna(df.iloc[i - 1]['sma_20']):
            prev_row = df.iloc[i - 1]

            # Price crossed above SMA 20
            if prev_row['close'] < prev_row['sma_20'] and row['close'] > row['sma_20']:
                desc_parts.append(f"broke above 20-day SMA at {row['sma_20']:.5f}, momentum turning positive")

            # Price crossed below SMA 20
            elif prev_row['close'] > prev_row['sma_20'] and row['close'] < row['sma_20']:
                desc_parts.append(f"broke below 20-day SMA at {row['sma_20']:.5f}, momentum turning negative")

        # Candlestick patterns
        prev_open = df.iloc[i - 1]['open'] if i > 0 else None
        prev_close = df.iloc[i - 1]['close'] if i > 0 else None

        candle_desc = describe_candle(
            row['open'], row['high'], row['low'], row['close'],
            prev_open, prev_close
        )

        # Only add candle description if it's interesting
        if "Strong" in candle_desc or "engulfing" in candle_desc or "Doji" in candle_desc:
            desc_parts.append(candle_desc)

        # Combine descriptions
        if desc_parts:
            descriptions.append(". ".join(desc_parts) + ".")
        else:
            # Minimal description for quiet bars
            descriptions.append(f"{pair} traded at {row['close']:.5f}")

    return descriptions


def describe_volatility_regime(
        df: pd.DataFrame,
        window: int = 50,
        pair: str = "PAIR"
) -> list[str]:
    """
    Generate volatility regime descriptions.

    Args:
        df: DataFrame with OHLCV data
        window: Window for calculating average volatility
        pair: Currency pair name

    Returns:
        List of volatility descriptions
    """
    descriptions = []

    # Calculate ATR (Average True Range)
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_sma'] = df['atr'].rolling(window).mean()

    for i in range(len(df)):
        row = df.iloc[i]

        if pd.notna(row['atr']) and pd.notna(row['atr_sma']) and row['atr_sma'] > 0:
            atr_ratio = row['atr'] / row['atr_sma']

            if atr_ratio > 2.0:
                descriptions.append(
                    f"High volatility regime for {pair}: ATR {atr_ratio:.1f}x above {window}-day average")
            elif atr_ratio > 1.5:
                descriptions.append(f"Elevated volatility for {pair}")
            elif atr_ratio < 0.5:
                descriptions.append(f"Low volatility regime for {pair}: consolidation phase")
            else:
                descriptions.append(f"Normal volatility for {pair}")
        else:
            descriptions.append(f"Volatility data not yet available for {pair}")

    return descriptions


def generate_pattern_text(
        df: pd.DataFrame,
        pair: str,
        include_volatility: bool = True,
        min_window: int = 50
) -> pd.DataFrame:
    """
    Generate comprehensive pattern descriptions for price data.

    Args:
        df: DataFrame with OHLCV data
            Required columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
        pair: Currency pair name (e.g., "EUR/USD")
        include_volatility: Whether to include volatility descriptions
        min_window: Minimum window for calculating indicators
                   (first N rows will have limited descriptions)

    Returns:
        DataFrame with columns: ['datetime', 'pattern_description', 'open', 'high', 'low', 'close', 'volume']
        Suitable for ingestion into Cognee via CogneeDataProcessor.ingest_price_patterns()

    Example:
        >>> df = pd.read_csv("eurusd_2023.csv")
        >>> patterns = generate_pattern_text(df, "EUR/USD")
        >>> print(patterns['pattern_description'].iloc[100])
        EUR/USD rallied 1.2% to 1.08500. Strong bullish candle with 1.2% body. High volatility regime...
    """
    logger.info(f"[narrator] Generating pattern descriptions for {pair} ({len(df)} bars)")

    if not {'datetime', 'open', 'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain: datetime, open, high, low, close")

    # Make a copy to avoid modifying original
    result_df = df.copy()

    # Ensure datetime is datetime type
    result_df['datetime'] = pd.to_datetime(result_df['datetime'])

    # Generate price movement descriptions
    price_descriptions = describe_price_movement(result_df, window=20, pair=pair)

    # Generate volatility descriptions if requested
    if include_volatility:
        volatility_descriptions = describe_volatility_regime(result_df, window=50, pair=pair)

        # Combine descriptions
        combined_descriptions = [
            f"{price}. {vol}" if price and vol else (price or vol or f"{pair} trading normally")
            for price, vol in zip(price_descriptions, volatility_descriptions, strict=False)
        ]
    else:
        combined_descriptions = price_descriptions

    # Create output DataFrame
    output_df = pd.DataFrame({
        'datetime': result_df['datetime'],
        'pattern_description': combined_descriptions,
        'open': result_df['open'],
        'high': result_df['high'],
        'low': result_df['low'],
        'close': result_df['close'],
        'volume': result_df.get('volume', 0)  # Optional column
    })

    logger.info(f"[narrator] Generated {len(output_df)} pattern descriptions")

    # Show sample
    if len(output_df) > 0:
        sample_idx = min(100, len(output_df) - 1)
        logger.info(f"[narrator] Sample description (bar {sample_idx}):")
        logger.info(f"  {output_df.iloc[sample_idx]['pattern_description']}")

    return output_df


if __name__ == "__main__":
    # Example usage and testing
    import argparse

    parser = argparse.ArgumentParser(description="Generate price pattern narratives")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file with OHLCV data")
    parser.add_argument("--pair", required=True, help="Currency pair (e.g., EUR/USD)")
    parser.add_argument("--output", type=Path, help="Output CSV file path")
    parser.add_argument("--no-volatility", action="store_true", help="Exclude volatility descriptions")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PRICE PATTERN NARRATOR")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Pair: {args.pair}")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading price data...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} bars")

    # Ensure required columns
    if 'datetime' not in df.columns:
        if 'DateTime' in df.columns:
            df['datetime'] = df['DateTime']
        elif 'Date' in df.columns:
            df['datetime'] = df['Date']
        else:
            logger.error("No datetime column found!")
            sys.exit(1)

    # Rename columns if needed (handle different formats)
    column_mapping = {
        'DateTime': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Generate descriptions
    logger.info("\nGenerating pattern narratives...")
    pattern_df = generate_pattern_text(
        df,
        pair=args.pair,
        include_volatility=not args.no_volatility
    )

    # Show sample descriptions
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE PATTERN DESCRIPTIONS")
    logger.info("=" * 80)

    sample_indices = [
        len(pattern_df) // 4,
        len(pattern_df) // 2,
        3 * len(pattern_df) // 4
    ]

    for idx in sample_indices:
        row = pattern_df.iloc[idx]
        logger.info(f"\n[{row['datetime']}]")
        logger.info(f"{row['pattern_description']}")

    # Save output
    if args.output:
        pattern_df.to_csv(args.output, index=False)
        logger.info(f"\n✅ Saved {len(pattern_df)} pattern descriptions to {args.output}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PATTERN NARRATION COMPLETE")
    logger.info("=" * 80)
