"""
Regime detection using Gaussian Mixture Models (GMM).

Detects market regimes (trend, consolidation, volatile, crash) from price/volatility patterns.
Useful for:
  - Adapting model confidence by regime
  - Adjusting position sizing in different states
  - Risk management based on regime-specific volatility

Example:
  detector = RegimeDetector(n_regimes=4)
  regimes = detector.fit_predict(price_data)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeConfig:
    """Regime detection configuration."""
    n_regimes: int = 4  # Number of regimes: trend_up, trend_down, consolidate, volatile
    lookback: int = 50  # Window for computing features
    max_iter: int = 100  # EM iterations
    random_state: int = 42
    min_samples: int = 100  # Minimum samples to fit model


class RegimeDetector:
    """
    Gaussian Mixture Model-based regime detector.

    Identifies 4 market regimes:
      0: UPTREND - positive drift, low volatility
      1: DOWNTREND - negative drift, low volatility
      2: CONSOLIDATION - low drift, low volatility
      3: VOLATILE - any drift, high volatility or large moves
    """

    REGIME_NAMES = ["UPTREND", "DOWNTREND", "CONSOLIDATION", "VOLATILE"]

    def __init__(self, cfg: RegimeConfig = None):
        self.cfg = cfg or RegimeConfig()
        self.gmm: GaussianMixture | None = None
        self.scaler = StandardScaler()
        self.regime_history = None

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features for regime classification.

        Features:
          1. returns: log returns
          2. volatility: rolling std of returns
          3. high_low_ratio: (high - low) / close (range as % of close)
          4. close_position: (close - low) / (high - low) (0=bottom, 1=top of range)
          5. volume_profile: volume / MA(volume) - volume surge indicator
        """
        df = df.copy()

        # 1. Log returns
        df["returns"] = np.log(df["close"] / df["close"].shift(1))

        # 2. Volatility (rolling std of returns)
        df["volatility"] = df["returns"].rolling(self.cfg.lookback).std()

        # 3. High-low range (as % of close)
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]

        # 4. Close position in candle (0=bottom, 1=top)
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # 5. Volume surge (volume relative to moving average)
        df["volume_ma"] = df["volume"].rolling(self.cfg.lookback).mean()
        df["volume_surge"] = df["volume"] / (df["volume_ma"] + 1e-8)

        # Return selected features
        features = df[["returns", "volatility", "range_pct", "close_position", "volume_surge"]].values

        # Fill NaN with forward fill then back fill
        features = pd.DataFrame(features).bfill().ffill().values

        return features

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit GMM to historical price data.

        Args:
            df: DataFrame with columns [datetime, open, high, low, close, volume]
        """
        if len(df) < self.cfg.min_samples:
            raise ValueError(
                f"Insufficient data: {len(df)} < {self.cfg.min_samples}"
            )

        features = self._compute_features(df)
        features_scaled = self.scaler.fit_transform(features)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.cfg.n_regimes,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            n_init=10,
        )
        self.gmm.fit(features_scaled)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for each row.

        Args:
            df: DataFrame with price data

        Returns:
            regimes: array of regime labels [0, 1, 2, 3]
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        features = self._compute_features(df)
        features_scaled = self.scaler.transform(features)

        regimes = self.gmm.predict(features_scaled)
        self.regime_history = regimes

        return regimes

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and predict in one call."""
        self.fit(df)
        return self.predict(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get soft regime probabilities (confidence for each regime).

        Returns:
            proba: shape (n_samples, n_regimes), probabilities for each regime
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        features = self._compute_features(df)
        features_scaled = self.scaler.transform(features)

        return self.gmm.predict_proba(features_scaled)

    def attach_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach regime labels and one-hot encoded regime features to dataframe.

        Returns:
            df: DataFrame with added columns:
              - regime: regime label
              - regime_is_uptrend, regime_is_downtrend, regime_is_consolidate, regime_is_volatile
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        regimes = self.predict(df)

        df = df.copy()
        df["regime"] = regimes

        # One-hot encode regimes
        for i, regime_name in enumerate(self.REGIME_NAMES):
            df[f"regime_is_{regime_name.lower()}"] = (regimes == i).astype(int)

        return df


def integrate_regime_features(feature_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to attach regime features to a feature dataframe.

    Args:
        feature_df: Feature dataframe with technical indicators
        price_df: Original price dataframe used for regime detection

    Returns:
        feature_df with regime columns appended
    """
    detector = RegimeDetector()
    detector.fit(price_df)

    # Align indices
    feature_df_copy = feature_df.copy()
    regimes = detector.predict(price_df)

    if len(regimes) != len(feature_df):
        # Take last len(feature_df) regimes
        regimes = regimes[-len(feature_df):]

    feature_df_copy["regime"] = regimes

    # Add one-hot regime indicators
    for i, regime_name in enumerate(RegimeDetector.REGIME_NAMES):
        feature_df_copy[f"regime_is_{regime_name.lower()}"] = (regimes == i).astype(int)

    return feature_df_copy
