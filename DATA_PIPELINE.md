# Data Pipeline Specification

## Overview

Multi-source data loading and preprocessing for Dignity Foundation Model pre-training.

## Architecture

```
Data Sources → Raw Loaders → Timestamp Alignment → Feature Engineering → Sequence Windowing → Training Batches
```

## 1. Raw Data Loaders

### 1.1 Cryptocurrency Loader (CCXT)

**File:** `data/source/crypto_historical.py`

```python
"""Historical cryptocurrency data via CCXT."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time


class CCXTHistoricalLoader:
    """Load historical OHLCV data from cryptocurrency exchanges."""
    
    def __init__(
        self,
        exchange: str = "binance",
        cache_dir: str = "data/cache/crypto"
    ):
        """
        Args:
            exchange: Exchange name (binance, coinbase, kraken, etc.)
            cache_dir: Directory for caching downloaded data
        """
        self.exchange = getattr(ccxt, exchange)()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_date: str = "2020-01-01",
        end_date: str = "2026-01-01"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check cache first
        cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.parquet"
        
        if cache_file.exists():
            print(f"Loading {symbol} from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Convert dates to timestamps (milliseconds)
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_candles = []
        current = since
        
        print(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}...")
        
        while current < end:
            try:
                # Fetch batch (exchanges typically limit to 1000 candles per request)
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update cursor to last candle timestamp
                current = candles[-1][0] + 1
                
                # Rate limiting (respect exchange limits)
                time.sleep(self.exchange.rateLimit / 1000)
                
                # Progress indicator
                progress = (current - since) / (end - since) * 100
                print(f"  Progress: {progress:.1f}% ({len(all_candles)} candles)", end="\r")
                
            except Exception as e:
                print(f"\nError fetching data: {e}")
                time.sleep(5)  # Back off on error
        
        print(f"\n  Fetched {len(all_candles)} total candles")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Ensure timestamp is int64 milliseconds
        df["timestamp"] = df["timestamp"].astype("int64")
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Save to cache
        df.to_parquet(cache_file, index=False)
        print(f"  Cached to {cache_file}")
        
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols: list[str],
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            **kwargs: Passed to fetch_ohlcv
        
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.fetch_ohlcv(symbol, **kwargs)
        return data


# Usage example
if __name__ == "__main__":
    loader = CCXTHistoricalLoader(exchange="binance")
    
    # Fetch major crypto assets
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    
    crypto_data = loader.fetch_multiple_symbols(
        symbols,
        timeframe="1m",
        start_date="2024-01-01",
        end_date="2024-02-01"  # Start small for testing
    )
    
    for symbol, df in crypto_data.items():
        print(f"{symbol}: {len(df)} candles, {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### 1.2 Forex Loader (OANDA / Alpha Vantage)

**File:** `data/source/forex_historical.py`

```python
"""Historical forex data via OANDA or Alpha Vantage."""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time


class ForexHistoricalLoader:
    """Load historical forex OHLC data."""
    
    def __init__(
        self,
        provider: str = "alphavantage",
        api_key: str = None,
        cache_dir: str = "data/cache/forex"
    ):
        """
        Args:
            provider: 'alphavantage' or 'oanda'
            api_key: API key for the provider
            cache_dir: Directory for caching downloaded data
        """
        self.provider = provider
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_ohlc(
        self,
        pair: str,
        interval: str = "1min",
        start_date: str = "2020-01-01",
        end_date: str = "2026-01-01"
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for a forex pair.
        
        Args:
            pair: Forex pair (e.g., 'EUR/USD', 'GBP/USD')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check cache
        cache_file = self.cache_dir / f"{pair.replace('/', '')}_{interval}_{start_date}_{end_date}.parquet"
        
        if cache_file.exists():
            print(f"Loading {pair} from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        if self.provider == "alphavantage":
            df = self._fetch_alpha_vantage(pair, interval, start_date, end_date)
        elif self.provider == "oanda":
            df = self._fetch_oanda(pair, interval, start_date, end_date)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Save to cache
        df.to_parquet(cache_file, index=False)
        print(f"  Cached to {cache_file}")
        
        return df
    
    def _fetch_alpha_vantage(
        self,
        pair: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch from Alpha Vantage API."""
        # Alpha Vantage uses format like "EUR/USD" -> "EURUSD"
        symbol = pair.replace("/", "")
        
        # Alpha Vantage interval mapping
        av_interval = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min"
        }[interval]
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol": symbol[3:],
            "interval": av_interval,
            "apikey": self.api_key,
            "outputsize": "full",
            "datatype": "json"
        }
        
        print(f"Fetching {pair} from Alpha Vantage...")
        response = requests.get(url, params=params)
        data = response.json()
        
        # Parse response
        time_series_key = f"Time Series FX ({av_interval})"
        if time_series_key not in data:
            raise ValueError(f"API error: {data}")
        
        records = []
        for timestamp_str, values in data[time_series_key].items():
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            records.append({
                "timestamp": timestamp_ms,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": 0  # Forex doesn't have volume (use tick count if available)
            })
        
        df = pd.DataFrame(records)
        
        # Filter date range
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        print(f"  Fetched {len(df)} candles")
        return df
    
    def _fetch_oanda(
        self,
        pair: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch from OANDA API (requires account)."""
        # OANDA implementation
        # See https://developer.oanda.com/rest-live-v20/instrument-ep/
        raise NotImplementedError("OANDA implementation requires account setup")
    
    def fetch_multiple_pairs(
        self,
        pairs: list[str],
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple forex pairs."""
        data = {}
        for pair in pairs:
            data[pair] = self.fetch_ohlc(pair, **kwargs)
            time.sleep(12)  # Alpha Vantage: 5 requests/min limit
        return data


# Usage example
if __name__ == "__main__":
    # Get API key from environment or config
    import os
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
    
    loader = ForexHistoricalLoader(provider="alphavantage", api_key=api_key)
    
    # Fetch major forex pairs
    pairs = ["EUR/USD", "GBP/USD", "USD/JPY"]
    
    forex_data = loader.fetch_multiple_pairs(
        pairs,
        interval="1min",
        start_date="2024-01-01",
        end_date="2024-01-02"  # Small test
    )
    
    for pair, df in forex_data.items():
        print(f"{pair}: {len(df)} candles")
```

## 2. Timestamp Alignment

**File:** `data/pipeline/alignment.py`

```python
"""Align multi-source data to common timestamps."""

import pandas as pd
import numpy as np


class TimestampAligner:
    """Align multiple data sources to a common timestamp grid."""
    
    def __init__(self, reference_interval_ms: int = 60000):
        """
        Args:
            reference_interval_ms: Reference interval in milliseconds (default: 1 min)
        """
        self.reference_interval_ms = reference_interval_ms
    
    def align(
        self,
        data_sources: dict[str, pd.DataFrame],
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Align multiple data sources to common timestamp grid.
        
        Args:
            data_sources: Dict mapping source name -> DataFrame (must have 'timestamp' column)
            method: Fill method for missing values ('ffill', 'bfill', 'interpolate')
        
        Returns:
            Aligned DataFrame with multi-level columns [source][feature]
        """
        # Find global timestamp range
        min_ts = min(df["timestamp"].min() for df in data_sources.values())
        max_ts = max(df["timestamp"].max() for df in data_sources.values())
        
        # Create reference grid
        reference_grid = np.arange(
            min_ts,
            max_ts + self.reference_interval_ms,
            self.reference_interval_ms
        )
        
        print(f"Aligning {len(data_sources)} sources to {len(reference_grid)} timestamps")
        
        aligned_dfs = {}
        
        for source_name, df in data_sources.items():
            # Set timestamp as index
            df = df.set_index("timestamp")
            
            # Reindex to reference grid
            df_aligned = df.reindex(reference_grid, method=method)
            
            # Add source name prefix to columns
            df_aligned.columns = pd.MultiIndex.from_product(
                [[source_name], df_aligned.columns]
            )
            
            aligned_dfs[source_name] = df_aligned
        
        # Concatenate all sources
        aligned = pd.concat(aligned_dfs.values(), axis=1)
        aligned.index.name = "timestamp"
        
        return aligned.reset_index()


# Usage example
if __name__ == "__main__":
    # Simulate multi-source data
    crypto_btc = pd.DataFrame({
        "timestamp": [1000, 2000, 3000, 4000],
        "close": [50000, 50100, 50200, 50300],
        "volume": [100, 110, 105, 108]
    })
    
    forex_eur = pd.DataFrame({
        "timestamp": [1500, 2500, 3500],  # Different timestamps
        "close": [1.12, 1.13, 1.14],
        "volume": [0, 0, 0]
    })
    
    aligner = TimestampAligner(reference_interval_ms=1000)
    
    aligned = aligner.align({
        "BTC": crypto_btc,
        "EURUSD": forex_eur
    })
    
    print(aligned)
    # Output:
    #    timestamp    BTC              EURUSD
    #                close  volume    close  volume
    # 0      1000  50000.0   100.0      NaN     NaN
    # 1      2000  50100.0   110.0   1.12      0.0
    # 2      3000  50200.0   105.0   1.13      0.0
    # 3      4000  50300.0   108.0   1.14      0.0
```

## 3. Feature Engineering

**File:** `data/pipeline/features.py`

```python
"""Feature engineering for financial time series."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Compute technical indicators and derived features."""
    
    @staticmethod
    def compute_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Compute log returns."""
        df = df.copy()
        df[f"{price_col}_returns"] = np.log(df[price_col] / df[price_col].shift(1))
        df[f"{price_col}_returns"] = df[f"{price_col}_returns"].fillna(0)
        return df
    
    @staticmethod
    def compute_volatility(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 60
    ) -> pd.DataFrame:
        """Compute rolling volatility."""
        df = df.copy()
        returns = np.log(df[price_col] / df[price_col].shift(1))
        df[f"{price_col}_volatility"] = returns.rolling(window).std()
        df[f"{price_col}_volatility"] = df[f"{price_col}_volatility"].fillna(0)
        return df
    
    @staticmethod
    def compute_rsi(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 14
    ) -> pd.DataFrame:
        """Compute Relative Strength Index."""
        df = df.copy()
        delta = df[price_col].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        
        rs = gain / loss
        df[f"{price_col}_rsi"] = 100 - (100 / (1 + rs))
        df[f"{price_col}_rsi"] = df[f"{price_col}_rsi"].fillna(50)  # Neutral
        return df
    
    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Compute MACD indicator."""
        df = df.copy()
        
        ema_fast = df[price_col].ewm(span=fast).mean()
        ema_slow = df[price_col].ewm(span=slow).mean()
        
        df[f"{price_col}_macd"] = ema_fast - ema_slow
        df[f"{price_col}_macd_signal"] = df[f"{price_col}_macd"].ewm(span=signal).mean()
        df[f"{price_col}_macd_hist"] = df[f"{price_col}_macd"] - df[f"{price_col}_macd_signal"]
        
        df[f"{price_col}_macd"] = df[f"{price_col}_macd"].fillna(0)
        df[f"{price_col}_macd_signal"] = df[f"{price_col}_macd_signal"].fillna(0)
        df[f"{price_col}_macd_hist"] = df[f"{price_col}_macd_hist"].fillna(0)
        
        return df
    
    @staticmethod
    def compute_volume_ma(
        df: pd.DataFrame,
        volume_col: str = "volume",
        window: int = 20
    ) -> pd.DataFrame:
        """Compute volume moving average."""
        df = df.copy()
        df[f"{volume_col}_ma"] = df[volume_col].rolling(window).mean()
        df[f"{volume_col}_ma"] = df[f"{volume_col}_ma"].fillna(df[volume_col])
        return df
    
    def compute_all_features(
        self,
        df: pd.DataFrame,
        price_cols: list[str] = None,
        volume_cols: list[str] = None
    ) -> pd.DataFrame:
        """Compute all technical features."""
        df = df.copy()
        
        if price_cols is None:
            # Auto-detect price columns (contain 'close')
            price_cols = [col for col in df.columns if 'close' in col.lower()]
        
        if volume_cols is None:
            # Auto-detect volume columns
            volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        # Price-based features
        for col in price_cols:
            df = self.compute_returns(df, col)
            df = self.compute_volatility(df, col, window=60)
            df = self.compute_rsi(df, col)
            df = self.compute_macd(df, col)
        
        # Volume-based features
        for col in volume_cols:
            df = self.compute_volume_ma(df, col)
        
        return df


# Usage example
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        "timestamp": range(1000, 2000),
        "BTC_close": np.random.randn(1000).cumsum() + 50000,
        "BTC_volume": np.random.randint(100, 200, 1000)
    })
    
    engineer = FeatureEngineer()
    df_features = engineer.compute_all_features(df)
    
    print(df_features.columns.tolist())
    # ['timestamp', 'BTC_close', 'BTC_volume', 'BTC_close_returns',
    #  'BTC_close_volatility', 'BTC_close_rsi', 'BTC_close_macd', ...]
```

## 4. Sequence Windowing

**File:** `data/pipeline/windowing.py`

```python
"""Create training sequences from continuous time series."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """Sliding window dataset for sequence modeling."""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10000,
        stride: int = 1000,
        feature_columns: list[str] = None
    ):
        """
        Args:
            data: 2D array [timesteps, features]
            sequence_length: Length of each sequence window
            stride: Stride between consecutive windows
            feature_columns: Names of features (for debugging)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_columns = feature_columns
        
        # Compute number of windows
        self.num_windows = (len(data) - sequence_length) // stride + 1
        
        print(f"Created {self.num_windows} windows from {len(data)} timesteps")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stride: {stride}")
        print(f"  Features: {data.shape[1]}")
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sequence_length
        
        sequence = self.data[start:end]
        
        # Convert to tensor
        sequence_tensor = torch.from_numpy(sequence).float()
        
        return {
            "sequence": sequence_tensor,  # [seq_len, features]
            "start_idx": start,
            "end_idx": end
        }


# Usage example
if __name__ == "__main__":
    # Simulate continuous time series
    data = np.random.randn(100000, 13)  # 100K timesteps, 13 features
    
    dataset = SequenceWindowDataset(
        data,
        sequence_length=10000,
        stride=1000
    )
    
    # Test iteration
    sample = dataset[0]
    print(f"Sample shape: {sample['sequence'].shape}")  # [10000, 13]
    
    # DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch shape: {batch['sequence'].shape}")  # [8, 10000, 13]
```

## 5. Complete Data Pipeline

**File:** `data/pipeline/foundation_data.py`

```python
"""Complete data pipeline for foundation model pre-training."""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.source.crypto_historical import CCXTHistoricalLoader
from data.source.forex_historical import ForexHistoricalLoader
from data.pipeline.alignment import TimestampAligner
from data.pipeline.features import FeatureEngineer
from data.pipeline.windowing import SequenceWindowDataset


class FoundationDataPipeline:
    """End-to-end data pipeline for Dignity-FM pre-training."""
    
    def __init__(
        self,
        crypto_symbols: list[str] = None,
        forex_pairs: list[str] = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        cache_dir: str = "data/cache"
    ):
        """
        Args:
            crypto_symbols: List of crypto symbols (e.g., ['BTC/USDT'])
            forex_pairs: List of forex pairs (e.g., ['EUR/USD'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cache_dir: Directory for caching processed data
        """
        self.crypto_symbols = crypto_symbols or ["BTC/USDT", "ETH/USDT"]
        self.forex_pairs = forex_pairs or ["EUR/USD"]
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        
        # Initialize loaders
        self.crypto_loader = CCXTHistoricalLoader()
        # self.forex_loader = ForexHistoricalLoader(api_key="YOUR_KEY")
        
        # Initialize processors
        self.aligner = TimestampAligner(reference_interval_ms=60000)  # 1 min
        self.feature_engineer = FeatureEngineer()
    
    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw data from all sources."""
        data = {}
        
        # Load crypto
        print("Loading cryptocurrency data...")
        for symbol in self.crypto_symbols:
            key = f"crypto_{symbol.replace('/', '_')}"
            data[key] = self.crypto_loader.fetch_ohlcv(
                symbol,
                timeframe="1m",
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        # Load forex
        print("Loading forex data...")
        # Uncomment when API key is configured
        # for pair in self.forex_pairs:
        #     key = f"forex_{pair.replace('/', '')}"
        #     data[key] = self.forex_loader.fetch_ohlc(
        #         pair,
        #         interval="1min",
        #         start_date=self.start_date,
        #         end_date=self.end_date
        #     )
        
        return data
    
    def process(self) -> pd.DataFrame:
        """Full processing pipeline."""
        # Step 1: Load raw data
        raw_data = self.load_raw_data()
        
        # Step 2: Align timestamps
        print("Aligning timestamps...")
        aligned = self.aligner.align(raw_data, method="ffill")
        
        # Step 3: Feature engineering
        print("Computing features...")
        with_features = self.feature_engineer.compute_all_features(aligned)
        
        # Step 4: Handle missing values (after alignment)
        with_features = with_features.fillna(0)
        
        print(f"Final shape: {with_features.shape}")
        print(f"Columns: {with_features.columns.tolist()}")
        
        return with_features
    
    def create_dataset(
        self,
        sequence_length: int = 10000,
        stride: int = 1000,
        train_split: float = 0.8
    ) -> tuple[SequenceWindowDataset, SequenceWindowDataset]:
        """Create train/val datasets."""
        # Process data
        df = self.process()
        
        # Convert to numpy
        # Drop timestamp column, keep only features
        features = df.drop(columns=["timestamp"]).values
        
        # Train/val split (chronological)
        split_idx = int(len(features) * train_split)
        train_data = features[:split_idx]
        val_data = features[split_idx:]
        
        # Create datasets
        train_dataset = SequenceWindowDataset(
            train_data,
            sequence_length=sequence_length,
            stride=stride
        )
        
        val_dataset = SequenceWindowDataset(
            val_data,
            sequence_length=sequence_length,
            stride=stride
        )
        
        return train_dataset, val_dataset


# Usage example
if __name__ == "__main__":
    pipeline = FoundationDataPipeline(
        crypto_symbols=["BTC/USDT", "ETH/USDT"],
        # forex_pairs=["EUR/USD"],  # Add when API configured
        start_date="2024-01-01",
        end_date="2024-01-07"  # Small test
    )
    
    train_ds, val_ds = pipeline.create_dataset(
        sequence_length=10000,
        stride=1000
    )
    
    print(f"Train: {len(train_ds)} windows")
    print(f"Val: {len(val_ds)} windows")
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['sequence'].shape}")
```

## Next Steps

1. **Install dependencies:**
   ```bash
   conda install -y -c conda-forge ccxt requests
   ```

2. **Test crypto loader:**
   ```bash
   cd .worktree/Dignity-Model
   python data/source/crypto_historical.py
   ```

3. **Get API keys:**
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key (free tier: 5 req/min)
   - Alternative: OANDA (requires account)

4. **Run full pipeline:**
   ```bash
   python data/pipeline/foundation_data.py
   ```

5. **Monitor data quality:**
   - Check for NaN values
   - Verify timestamp alignment
   - Inspect feature distributions

## Configuration

**Environment variables:**
```bash
export ALPHAVANTAGE_API_KEY="your_key_here"
export CCXT_EXCHANGE="binance"  # or coinbase, kraken, etc.
```

**Config file:** `config/data_pipeline.yaml`
```yaml
crypto:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - BNB/USDT
    - SOL/USDT
    - ADA/USDT
  exchange: binance
  timeframe: 1m

forex:
  pairs:
    - EUR/USD
    - GBP/USD
    - USD/JPY
  provider: alphavantage
  interval: 1min

alignment:
  reference_interval_ms: 60000  # 1 minute
  fill_method: ffill

features:
  volatility_window: 60
  rsi_window: 14
  macd_fast: 12
  macd_slow: 26
  volume_ma_window: 20

sequences:
  length: 10000
  stride: 1000
  train_split: 0.8

date_range:
  start: "2020-01-01"
  end: "2026-01-01"
```
