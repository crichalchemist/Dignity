"""Cryptocurrency data source interface."""

import numpy as np
import pandas as pd
from typing import Optional, List


class CryptoSource:
    """
    Interface for cryptocurrency transaction data.
    
    Can pull from:
    - CCXT (exchange data)
    - On-chain APIs (Blockchair, Etherscan, etc.)
    - Local CSV files
    """
    
    def __init__(self, pair: str = 'BTC/USD'):
        """
        Args:
            pair: Trading pair (e.g., 'BTC/USD', 'XMR/USD')
        """
        self.pair = pair
    
    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load cryptocurrency data from CSV file.
        
        Expected columns:
        - timestamp
        - open, high, low, close, volume
        - Optional: tx_count, fee_rate
        
        Args:
            path: Path to CSV file
            
        Returns:
            DataFrame with cryptocurrency data
        """
        df = pd.DataFrame(pd.read_csv(path))
        
        # Ensure required columns
        required = ['timestamp', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw cryptocurrency data.
        
        Args:
            df: Raw dataframe with price/volume data
            
        Returns:
            DataFrame with engineered features
        """
        prepared = df.copy()
        
        # Price-based features
        if 'close' in df.columns:
            prepared['price'] = df['close']
            prepared['price_change'] = df['close'].pct_change().fillna(0)
            prepared['volatility'] = df['close'].rolling(20).std().fillna(0)
        
        # Volume features
        if 'volume' in df.columns:
            prepared['volume_ma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
            prepared['volume_std'] = df['volume'].rolling(20).std().fillna(0)
        
        # High-low spread (if available)
        if 'high' in df.columns and 'low' in df.columns:
            prepared['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return prepared
    
    def resample_to_blocks(self,
                          df: pd.DataFrame,
                          block_time: str = '10min') -> pd.DataFrame:
        """
        Resample exchange data to block-like intervals.
        
        Args:
            df: Dataframe with timestamp index
            block_time: Block time interval (e.g., '10min', '1h')
            
        Returns:
            Resampled DataFrame
        """
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Resample
        resampled = df.resample(block_time).agg({
            'close': 'last',
            'volume': 'sum',
            'high': 'max',
            'low': 'min'
        }).fillna(method='ffill')
        
        return resampled.reset_index()
