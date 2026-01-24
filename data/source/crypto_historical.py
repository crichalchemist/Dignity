"""CCXT-based cryptocurrency historical data loader with caching."""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd


class CCXTHistoricalLoader:
    """
    Load historical OHLCV data from cryptocurrency exchanges using CCXT.
    
    Features:
        - Multiple exchange support (Binance, Coinbase, Kraken)
        - Automatic caching to parquet files
        - Rate limiting to respect API limits
        - Batch fetching for large date ranges
        - Retry logic for failed requests
    
    Example:
        >>> loader = CCXTHistoricalLoader(exchange='binance', cache_dir='data/cache')
        >>> df = loader.fetch_ohlcv('BTC/USDT', start_date='2024-01-01', end_date='2024-01-07')
        >>> print(df.head())
    """
    
    def __init__(
        self,
        exchange: str = 'binance',
        cache_dir: str = 'data/cache',
        enable_cache: bool = True,
    ):
        """
        Initialize CCXT historical data loader.
        
        Args:
            exchange: Exchange name (binance, coinbase, kraken)
            cache_dir: Directory for caching parquet files
            enable_cache: Whether to use file caching
        """
        self.exchange_name = exchange.lower()
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        self._init_exchange()
    
    def _init_exchange(self):
        """Initialize CCXT exchange object."""
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,  # Respect API rate limits
            'options': {
                'defaultType': 'spot',  # Spot market
            }
        })
    
    def _get_cache_path(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        # Sanitize symbol for filename (BTC/USDT -> BTC_USDT)
        safe_symbol = symbol.replace('/', '_')
        filename = f"{self.exchange_name}_{safe_symbol}_{timeframe}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        if not self.enable_cache:
            return None
        
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save data to cache."""
        if not self.enable_cache:
            return
        
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        start_date: str = '2024-01-01',
        end_date: Optional[str] = None,
        limit_per_request: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            limit_per_request: Max candles per API request
        
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
            - timestamp: Unix timestamp in milliseconds
            - open, high, low, close: Price data
            - volume: Trading volume
        
        Example:
            >>> loader = CCXTHistoricalLoader('binance')
            >>> df = loader.fetch_ohlcv('BTC/USDT', '1m', '2024-01-01', '2024-01-02')
            >>> print(f"Loaded {len(df)} candles")
        """
        # Set default end date to today
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        cached_df = self._load_from_cache(cache_path)
        if cached_df is not None:
            print(f"✓ Loaded {len(cached_df)} candles from cache: {cache_path.name}")
            return cached_df
        
        # Convert dates to millisecond timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch data in batches
        print(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}...")
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                # Fetch batch
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=limit_per_request
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update timestamp for next batch
                current_ts = candles[-1][0] + 1
                
                # Rate limiting (handled by CCXT enableRateLimit)
                print(f"  Fetched {len(candles)} candles (total: {len(all_candles)})")
                
                # Stop if we've reached the end date
                if candles[-1][0] >= end_ts:
                    break
                
            except ccxt.BaseError as e:
                print(f"Warning: CCXT error: {e}")
                time.sleep(5)  # Wait before retry
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Filter to exact date range
        df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Fetched {len(df)} candles from {self.exchange_name}")
        
        # Save to cache
        self._save_to_cache(df, cache_path)
        
        return df
    
    def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = '1m',
        start_date: str = '2024-01-01',
        end_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary mapping symbol -> DataFrame
        
        Example:
            >>> loader = CCXTHistoricalLoader('binance')
            >>> data = loader.fetch_multiple(['BTC/USDT', 'ETH/USDT'], start_date='2024-01-01')
            >>> print(f"Loaded {len(data)} symbols")
        """
        result = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)
            result[symbol] = df
            time.sleep(0.5)  # Small delay between symbols
        return result


def generate_synthetic_crypto_data(
    symbol: str = 'BTC/USDT',
    start_date: str = '2024-01-01',
    end_date: str = '2024-01-07',
    timeframe: str = '1m',
    initial_price: float = 40000.0,
) -> pd.DataFrame:
    """
    Generate synthetic cryptocurrency OHLCV data for testing.
    
    Args:
        symbol: Trading pair name (for metadata only)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframe: Candle timeframe ('1m', '5m', '1h', '1d')
        initial_price: Starting price
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
    
    Example:
        >>> df = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        >>> print(f"Generated {len(df)} synthetic candles")
    """
    import numpy as np

    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Determine time delta based on timeframe
    timeframe_deltas = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '1h': timedelta(hours=1),
        '1d': timedelta(days=1),
    }
    delta = timeframe_deltas.get(timeframe, timedelta(minutes=1))
    
    # Generate timestamps
    timestamps = []
    current_dt = start_dt
    while current_dt < end_dt:
        timestamps.append(int(current_dt.timestamp() * 1000))
        current_dt += delta
    
    n_candles = len(timestamps)
    
    # Generate synthetic price data with random walk
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(0.0001, 0.01, n_candles)  # Small upward drift
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    volatility = 0.002
    opens = prices
    closes = prices * (1 + np.random.normal(0, volatility, n_candles))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, volatility/2, n_candles)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, volatility/2, n_candles)))
    
    # Generate volume (random around mean)
    volumes = np.random.lognormal(10, 1, n_candles)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
    
    return df
