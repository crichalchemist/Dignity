"""Data pipeline for preprocessing transaction sequences."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.config import PrivacyConfig
from core.privacy import PrivacyBudget, PrivacyManager
from core.signals import ASSET_CONFIGS, AssetConfig, SignalProcessor


class TransactionPipeline:
    """
    Unified preprocessing pipeline for transaction data.

    Handles:
    - Feature selection
    - Scaling/normalization
    - Signal computation
    - Sequence windowing
    """

    def __init__(
        self,
        seq_len: int = 100,
        features: list = None,
        scaler_type: str = "robust",
        privacy: PrivacyConfig | None = None,
        privacy_rng=None,
    ):
        """
        Args:
            seq_len: Sequence length for windowing
            features: List of features to use
            scaler_type: Scaler type ('robust', 'standard', 'minmax')
            privacy: Optional privacy stage config. None => no stage runs.
            privacy_rng: Test seam forwarded to PrivacyManager(rng=...). Leave None.
        """
        self.seq_len = seq_len
        self.features = features or [
            "volume",
            "price",
            "fee_rate",
            "tx_count",
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bollinger_pct_b",
            "bollinger_width",
            "atr",
            "stoch_k",
            "stoch_d",
            "adx",
            "obv",
            "vwap",
            "roc_5",
            "roc_20",
            "momentum_10",
            "momentum_20",
            "volatility_5",
            "volatility_20",
            "vol_ratio",
            "order_flow_imbalance",
            "dc_direction",
            "dc_overshoot",
            "dc_bars_since_event",
            "volume_volatility",
            "volume_entropy",
            "price_change",
            "directional_change",
        ]

        # Initialize scaler
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.signal_processor = SignalProcessor()
        self.fitted = False

        self.privacy = privacy
        self.privacy_manager = None
        if privacy is not None:
            key = None
            if privacy.key_env is not None:
                key = os.environ.get(privacy.key_env)
                if key is None:
                    raise ValueError(
                        f"privacy.key_env={privacy.key_env!r} is not set in the environment"
                    )
            self.privacy_manager = PrivacyManager(
                PrivacyBudget(privacy.epsilon_total), key=key, rng=privacy_rng
            )

    def compute_signals(
        self,
        df: pd.DataFrame,
        asset_config: AssetConfig | None = None,
    ) -> pd.DataFrame:
        """Compute signal features from raw OHLCV data.

        When asset_config is provided, computes the full 32-feature quant set
        via SignalProcessor.process_sequence(). Without it, falls back to the
        original 7-signal set for backward compatibility.
        """
        result = df.copy()

        # Always compute signals, asset_config is now required for full features.
        # The fallback logic is removed to ensure consistency.
        if asset_config is None:
            # Default to forex config if not provided
            asset_config = ASSET_CONFIGS["forex"]

        prices = df.get("price", df.get("close"))
        if prices is not None:
            signals = self.signal_processor.process_sequence(
                volumes=df["volume"].values
                if "volume" in df.columns
                else np.ones(len(df)),
                prices=prices.values,
                fees=df["fee_rate"].values if "fee_rate" in df.columns else None,
                high=df["high"].values if "high" in df.columns else None,
                low=df["low"].values if "low" in df.columns else None,
                asset_config=asset_config,
            )
            for name, arr in signals.items():
                if name not in result.columns:
                    result[name] = arr
        return result

    def _apply_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the configured privacy mechanisms on raw columns.

        Runs BEFORE compute_signals so derived features inherit the guarantee
        by post-processing. Returns ``df`` itself when no privacy is configured.
        """
        if self.privacy_manager is None:
            return df
        result = df.copy()
        missing = [n for n in self.privacy.features if n not in result.columns]
        if missing:
            raise ValueError(
                f"privacy configured for column {missing[0]!r}, not in data"
            )
        for name, feat in self.privacy.features.items():
            col = result[name].to_numpy(dtype=float)
            if feat.mechanism == "laplace":
                result[name] = self.privacy_manager.add_laplace_noise(
                    col, epsilon=feat.epsilon, bounds=feat.bounds
                )
            else:
                result[name] = PrivacyManager.generalize_amounts(
                    col, bins=feat.bins, k=self.privacy.k
                )
        return result

    def fit(
        self, df: pd.DataFrame, asset_config: AssetConfig | None = None
    ) -> TransactionPipeline:
        """Fit the scaler on training data. Applies the privacy stage once."""
        self._fit_on(self._apply_privacy(df), asset_config)
        return self

    def transform(
        self, df: pd.DataFrame, asset_config: AssetConfig | None = None
    ) -> np.ndarray:
        """Transform to a scaled feature array. Applies the privacy stage once."""
        return self._transform_on(self._apply_privacy(df), asset_config)

    def fit_transform(
        self, df: pd.DataFrame, asset_config: AssetConfig | None = None
    ) -> np.ndarray:
        """Fit and transform on one privacy release, so the scaler sees the same draw."""
        prepared = self._apply_privacy(df)
        self._fit_on(prepared, asset_config)
        return self._transform_on(prepared, asset_config)

    def _fit_on(
        self, df: pd.DataFrame, asset_config: AssetConfig | None = None
    ) -> None:
        """Fit the scaler. ``df`` must already have had privacy applied."""
        df = self.compute_signals(df, asset_config)

        available_features = [f for f in self.features if f in df.columns]
        if not available_features:
            raise ValueError(
                f"None of the specified features found in data: {self.features}"
            )

        X = df[available_features].values
        self.scaler.fit(X)
        self.fitted = True
        self.available_features = available_features

    def _transform_on(
        self, df: pd.DataFrame, asset_config: AssetConfig | None = None
    ) -> np.ndarray:
        """Scale features. ``df`` must already have had privacy applied."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        df = self.compute_signals(df, asset_config)
        X = df[self.available_features].values
        return self.scaler.transform(X)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Create sliding window sequences.

        Args:
            X: Feature array [n_samples, n_features]
            y: Optional labels [n_samples] or [n_samples, label_dim]
            stride: Stride for sliding window

        Returns:
            Tuple of (X_seq, y_seq) or (X_seq, None)
            - X_seq: [n_sequences, seq_len, n_features]
            - y_seq: [n_sequences, ...] if y provided
        """
        n_samples, n_features = X.shape

        # Calculate number of sequences
        n_sequences = (n_samples - self.seq_len) // stride + 1

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for seq_len={self.seq_len}"
            )

        # Create sequences
        X_seq = np.zeros((n_sequences, self.seq_len, n_features))

        for i in range(n_sequences):
            start_idx = i * stride
            end_idx = start_idx + self.seq_len
            X_seq[i] = X[start_idx:end_idx]

        # Handle labels
        y_seq = None
        if y is not None:
            if y.ndim == 1:
                # Classification labels (take label at sequence end)
                y_seq = np.array(
                    [y[i * stride + self.seq_len - 1] for i in range(n_sequences)]
                )
            else:
                # Regression targets (take window after sequence)
                pred_len = y.shape[1] if y.ndim > 1 else 1
                y_seq = np.array(
                    [
                        y[
                            i * stride + self.seq_len : i * stride
                            + self.seq_len
                            + pred_len
                        ]
                        for i in range(n_sequences)
                    ]
                )

        return X_seq, y_seq

    def process(
        self,
        df: pd.DataFrame,
        labels: np.ndarray | None = None,
        fit: bool = True,
        stride: int = 1,
        asset_config: AssetConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Full pipeline: compute signals, scale, create sequences.

        Args:
            df: Raw dataframe
            labels: Optional labels
            fit: Whether to fit scaler (True for train, False for test)
            stride: Stride for sequence creation
            asset_config: Asset configuration for signal computation

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Transform to features
        if fit:
            X = self.fit_transform(df, asset_config)
        else:
            X = self.transform(df, asset_config)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, labels, stride=stride)

        return X_seq, y_seq
