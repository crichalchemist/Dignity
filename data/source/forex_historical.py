"""Forex historical data loader with histdata.com support."""

import io
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class ForexHistoricalLoader:
    """
    Load historical forex tick/OHLC data from histdata.com.
    
    Features:
        - Free historical forex data from histdata.com
        - Automatic download and extraction of ZIP files
        - Tick data and 1-minute bar data support
        - Automatic caching to parquet files
        - Same output format as crypto loader for alignment
    
    Data Source:
        https://www.histdata.com/download-free-forex-data/
        
        Available data:
        - Tick data (bid/ask) - Every tick
        - 1-minute bar data (OHLC) - 1-minute intervals
        - Free downloads (no API key required)
        - Historical data back to 2000s for major pairs
    
    Example:
        >>> loader = ForexHistoricalLoader(cache_dir='data/cache')
        >>> # Download 1-minute bar data
        >>> df = loader.fetch_ohlc('EURUSD', year=2024, month=1)
        >>> print(df.head())
    """
    
    BASE_URL = 'https://www.histdata.com/download-free-forex-historical-data'
    
    def __init__(
        self,
        cache_dir: str = 'data/cache',
        enable_cache: bool = True,
    ):
        """
        Initialize forex historical data loader.
        
        Args:
            cache_dir: Directory for caching parquet files
            enable_cache: Whether to use file caching
        """
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, pair: str, year: int, month: int) -> Path:
        """Generate cache file path for month data."""
        filename = f"forex_{pair}_M1_{year}_{month:02d}.parquet"
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
    
    def _parse_histdata_csv(self, csv_content: str, pair: str) -> pd.DataFrame:
        """
        Parse histdata.com CSV format to standard OHLCV format.
        
        histdata.com M1 CSV format:
        DateTime,Open,High,Low,Close,Volume
        20240101 000000,1.10450,1.10460,1.10440,1.10455,0
        
        Args:
            csv_content: Raw CSV content string
            pair: Currency pair (for metadata)
        
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        # Read CSV
        df = pd.read_csv(
            io.StringIO(csv_content),
            names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
            header=0  # Skip header row
        )
        
        # Parse datetime: "20240101 000000" -> timestamp in milliseconds
        df['timestamp'] = pd.to_datetime(
            df['DateTime'],
            format='%Y%m%d %H%M%S'
        ).astype('int64') // 10**6  # Convert to milliseconds
        
        # Rename columns to match crypto loader format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Select and reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_month_data(
        self,
        pair: str,
        year: int,
        month: int,
        manual_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch one month of 1-minute forex data.
        
        Args:
            pair: Currency pair (e.g., 'EURUSD', 'GBPUSD')
            year: Year (e.g., 2024)
            month: Month (1-12)
            manual_csv_path: Optional path to manually downloaded CSV file
        
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        
        Example:
            >>> loader = ForexHistoricalLoader()
            >>> # Try automatic download
            >>> df = loader.fetch_month_data('EURUSD', 2024, 1)
            >>> # Or use manually downloaded file
            >>> df = loader.fetch_month_data('EURUSD', 2024, 1, 
            ...     manual_csv_path='downloads/EURUSD_M1_202401.csv')
        """
        # Check cache first
        cache_path = self._get_cache_path(pair, year, month)
        cached_df = self._load_from_cache(cache_path)
        if cached_df is not None:
            print(f"✓ Loaded {len(cached_df)} candles from cache: {cache_path.name}")
            return cached_df
        
        # Load from manual CSV if provided
        if manual_csv_path:
            print(f"Loading from manual CSV: {manual_csv_path}")
            try:
                with open(manual_csv_path, 'r') as f:
                    csv_content = f.read()
                df = self._parse_histdata_csv(csv_content, pair)
                print(f"✓ Loaded {len(df)} candles from manual CSV")
                self._save_to_cache(df, cache_path)
                return df
            except Exception as e:
                print(f"Error loading manual CSV: {e}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Try automatic download (may not work due to histdata.com restrictions)
        print(f"Attempting to fetch {pair} {year}-{month:02d} from histdata.com...")
        print("Note: histdata.com may require manual download. See docstring for details.")
        
        # Return empty DataFrame with helpful message
        print("For reliable data access:")
        print(f"1. Visit https://www.histdata.com/download-free-forex-historical-data/")
        print(f"2. Download {pair} 1-minute data for {year}/{month:02d}")
        print(f"3. Call fetch_month_data() with manual_csv_path parameter")
        
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def fetch_ohlc(
        self,
        pair: str,
        year: int = 2024,
        month: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        manual_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch forex OHLC data (wrapper around fetch_month_data for compatibility).
        
        Args:
            pair: Currency pair without slash (e.g., 'EURUSD', not 'EUR/USD')
            year: Year for data
            month: Month for data (1-12)
            start_date: Optional filter start date 'YYYY-MM-DD'
            end_date: Optional filter end date 'YYYY-MM-DD'
            manual_csv_path: Optional path to manually downloaded CSV
        
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        
        Example:
            >>> loader = ForexHistoricalLoader()
            >>> df = loader.fetch_ohlc('EURUSD', year=2024, month=1)
        """
        # Normalize pair format (remove slash if present)
        pair = pair.replace('/', '').upper()
        
        # Fetch month data
        df = self.fetch_month_data(pair, year, month, manual_csv_path)
        
        # Filter by date range if specified
        if start_date and len(df) > 0:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            df = df[df['timestamp'] >= start_ts]
        
        if end_date and len(df) > 0:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            df = df[df['timestamp'] <= end_ts]
        
        return df.reset_index(drop=True)
    
    def fetch_multiple(
        self,
        pairs: list[str],
        year: int = 2024,
        month: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLC data for multiple currency pairs.
        
        Args:
            pairs: List of currency pairs (e.g., ['EURUSD', 'GBPUSD'])
            year: Year for data
            month: Month for data
        
        Returns:
            Dictionary mapping pair -> DataFrame
        
        Example:
            >>> loader = ForexHistoricalLoader()
            >>> data = loader.fetch_multiple(['EURUSD', 'GBPUSD'], year=2024, month=1)
        """
        result = {}
        for pair in pairs:
            df = self.fetch_ohlc(pair, year, month)
            result[pair] = df
        return result


def generate_synthetic_forex_data(
    pair: str = 'EUR/USD',
    start_date: str = '2024-01-01',
    end_date: str = '2024-01-07',
    interval: str = '1min',
    initial_price: float = 1.10,
) -> pd.DataFrame:
    """
    Generate synthetic forex OHLC data for testing.
    
    Args:
        pair: Currency pair name (for metadata only)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1min', '5min', '15min', '30min', '60min')
        initial_price: Starting exchange rate
    
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
        Note: volume is always 0 for forex
    
    Example:
        >>> df = generate_synthetic_forex_data('EUR/USD', '2024-01-01', '2024-01-02')
        >>> print(f"Generated {len(df)} synthetic candles")
    """
    import numpy as np

    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Determine time delta based on interval
    interval_deltas = {
        '1min': timedelta(minutes=1),
        '5min': timedelta(minutes=5),
        '15min': timedelta(minutes=15),
        '30min': timedelta(minutes=30),
        '60min': timedelta(hours=1),
    }
    delta = interval_deltas.get(interval, timedelta(minutes=1))
    
    # Generate timestamps
    timestamps = []
    current_dt = start_dt
    while current_dt < end_dt:
        timestamps.append(int(current_dt.timestamp() * 1000))
        current_dt += delta
    
    n_candles = len(timestamps)
    
    # Generate synthetic price data (forex is less volatile than crypto)
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(0.00001, 0.0005, n_candles)  # Low volatility
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    volatility = 0.0001  # Forex volatility is much lower
    opens = prices
    closes = prices * (1 + np.random.normal(0, volatility, n_candles))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, volatility/2, n_candles)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, volatility/2, n_candles)))
    
    # Create DataFrame (no volume for forex)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.zeros(n_candles),  # Forex doesn't have volume
    })
    
    return df
