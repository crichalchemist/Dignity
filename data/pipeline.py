"""Data pipeline for preprocessing transaction sequences."""

import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler

from core.config import PrivacyConfig
from core.privacy import BudgetExhausted, Generalization, PrivacyBudget, PrivacyManager
from core.signals import SignalProcessor
from data.labels import forecast_labels, risk_labels, window_meta, window_starts

# Event-record columns that carry labels: never a feature.
EVENT_LABEL_COLUMNS = (
    "is_anomalous",
    "anomaly_type",
    "anomaly_id",
    "block_label",
    "build_up",
)

# Derived signal -> the raw column it is computed from, in column order.
SIGNAL_SOURCES: dict[str, str] = {
    "volume_volatility": "volume",
    "volatility": "price",
    "momentum": "price",
    "directional_change": "price",
}

_SIGNAL_FNS: dict[str, Callable] = {
    "volume_volatility": SignalProcessor.volatility,
    "volatility": SignalProcessor.volatility,
    "momentum": SignalProcessor.price_momentum,
    "directional_change": SignalProcessor.directional_change,
}

_ALREADY_RELEASED = (
    "process_blocks already released this pipeline's data, one release per split; "
    "build a new TransactionPipeline for a new, disjoint release"
)

_INSTANCE_ALREADY_RELEASED = (
    "fit/transform already released this pipeline's data on its instance ledger; "
    "build a new TransactionPipeline for a new, disjoint release"
)


@dataclass
class WindowSet:
    """Windows of one split with every label the three heads need."""

    X: np.ndarray  # float32 [N, seq_len, F]
    y_risk: np.ndarray  # int8 [N]
    y_forecast: np.ndarray  # float32 [N, H]
    forecast_mask: np.ndarray  # bool [N]
    block_index: np.ndarray  # int64 [N], block_id of each window
    onset_offset: np.ndarray  # int64 [N], row of first anomalous row, -1 if none
    end_row: np.ndarray  # int64 [N], row-in-block of each window's last row
    anomaly_type: np.ndarray  # object [N], dominant type in window or "none"


@dataclass
class SplitData:
    train: WindowSet
    val: WindowSet
    test: WindowSet
    feature_names: list[str]
    scaler_center: np.ndarray
    scaler_scale: np.ndarray
    epsilon_spent: dict[str, float]  # per split; {} if no privacy
    privacy_mode: str = "off"  # off | injected | deterministic | system


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
        privacy_seed: int | None = None,
    ):
        """
        Args:
            seq_len: Sequence length for windowing
            features: List of features to use
            scaler_type: Scaler type ('robust', 'standard', 'minmax')
            privacy: Optional privacy stage config. None => no stage runs.
            privacy_rng: Test seam forwarded to PrivacyManager(rng=...). Leave None.
            privacy_seed: Seed for the per-split streams of ``privacy.deterministic_rng``
                (seeded experiment mode, not DP-grade randomness). Required then.
        """
        self.seq_len = seq_len
        self.features = features or [
            "volume",
            "price",
            "fee_rate",
            "tx_count",
            "volatility",
            "price_change",
            "momentum",
        ]
        bad = [f for f in self.features if f in EVENT_LABEL_COLUMNS]
        if bad:
            raise ValueError(
                f"{bad[0]!r} is an event-record label column and cannot be a feature"
            )

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

        if privacy is not None and privacy.deterministic_rng and privacy_seed is None:
            raise ValueError("privacy.deterministic_rng requires privacy_seed")
        self.privacy = privacy
        self.privacy_rng = privacy_rng
        self.privacy_seed = privacy_seed
        self._split_released = False
        self._instance_released = False
        self.privacy_manager = None
        if privacy is not None:
            self.privacy_manager = self._new_manager("instance")

    @property
    def privacy_mode(self) -> str:
        """off | injected (test seam) | deterministic (seeded) | system (default)."""
        if self.privacy is None:
            return "off"
        if self.privacy_rng is not None:
            return "injected"
        if self.privacy.deterministic_rng:
            return "deterministic"
        return "system"

    def _new_manager(self, split: str) -> PrivacyManager:
        """A fresh ledger of ``epsilon_total`` with this stream's rng.

        ``split`` names the stream in deterministic mode ("train", "val", "test",
        or "instance" for the process/fit/transform path), so no two releases share
        noise. An injected rng is shared by all; system mode uses SystemRandom.
        """
        key = None
        if self.privacy.key_env is not None:
            key = os.environ.get(self.privacy.key_env)
            if key is None:
                raise ValueError(
                    f"privacy.key_env={self.privacy.key_env!r} is not set in the "
                    "environment"
                )
        rng = self.privacy_rng
        if self.privacy_mode == "deterministic":
            rng = random.Random(f"dignity-privacy:{self.privacy_seed}:{split}")
        return PrivacyManager(
            PrivacyBudget(self.privacy.epsilon_total), key=key, rng=rng
        )

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional signal features.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with computed signals
        """
        result = df.copy()

        for name, source in SIGNAL_SOURCES.items():
            if source in df.columns:
                result[name] = _SIGNAL_FNS[name](df[source].values)

        return result

    def _apply_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the configured privacy mechanisms on raw columns.

        Runs BEFORE compute_signals so derived features inherit the guarantee
        by post-processing. Returns ``df`` itself when no privacy is configured.
        """
        if self.privacy_manager is None:
            return df
        if self._split_released:
            raise BudgetExhausted(_ALREADY_RELEASED)
        if self._instance_released:
            raise BudgetExhausted(_INSTANCE_ALREADY_RELEASED)
        released = self._release_split(df, self.privacy_manager, None)[0]
        self._instance_released = True
        return released

    def _release_split(
        self,
        df: pd.DataFrame,
        manager: PrivacyManager,
        generalizations: dict[str, Generalization] | None,
    ) -> tuple[pd.DataFrame, dict[str, Generalization]]:
        """One privacy release of ``df`` on ``manager``'s ledger.

        With ``generalizations=None`` the classes are fitted on ``df``; otherwise
        the given ones are applied. Returns the released frame and the classes.
        """
        result = df.copy()
        missing = [n for n in self.privacy.features if n not in result.columns]
        if missing:
            raise ValueError(
                f"privacy configured for column {missing[0]!r}, not in data"
            )
        fitted = {}
        for name, feat in self.privacy.features.items():
            col = result[name].to_numpy(dtype=float)
            if feat.mechanism == "laplace":
                if feat.transform == "log1p":
                    col = np.log1p(np.maximum(col, 0.0))
                result[name] = manager.add_laplace_noise(
                    col, epsilon=feat.epsilon, bounds=feat.bounds
                )
                continue
            if generalizations is None:
                fitted[name] = PrivacyManager.fit_generalization(
                    col, bins=feat.bins, k=self.privacy.k
                )
            else:
                fitted[name] = generalizations[name]
            result[name] = fitted[name].apply(col)
        return result, fitted

    def fit(self, df: pd.DataFrame) -> "TransactionPipeline":
        """Fit the scaler on training data. Applies the privacy stage once."""
        self._fit_on(self._apply_privacy(df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to a scaled feature array. Applies the privacy stage once.

        An unfitted pipeline raises before releasing, so the release is kept.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        return self._transform_on(self._apply_privacy(df))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform on one privacy release, so the scaler sees the same draw."""
        prepared = self._apply_privacy(df)
        self._fit_on(prepared)
        return self._transform_on(prepared)

    def process_blocks(
        self,
        df: pd.DataFrame,
        split: tuple[float, float, float],
        horizon: int,
        overlap: float,
        rng,
    ) -> SplitData:
        """Split an event-record frame by block and window each split.

        ``df`` is the generator's event record: one row per step with a
        ``block_id`` column plus ``is_anomalous``, ``anomaly_type`` and
        ``anomaly_id``. Blocks are shuffled with ``rng`` (its only use) and split
        at the block level into train/val/test by the ``split`` fractions.
        Each split is its own privacy release with its own budget (parallel
        composition over disjoint blocks); generalize classes and the scaler are
        fitted on the train release only. Signals and windows are computed inside
        each block so nothing crosses a joint. Labels come from ``data.labels``.
        """
        ids = np.unique(df["block_id"])
        if len(ids) < 3:
            raise ValueError(
                f"need at least 3 blocks for a train/val/test split, got {len(ids)}"
            )
        lengths = df.groupby("block_id", sort=False).size()
        if (lengths < self.seq_len + horizon).any():
            raise ValueError(
                f"every block needs seq_len + horizon = {self.seq_len + horizon} "
                f"rows; shortest block has {int(lengths.min())}"
            )

        n = len(ids)
        order = np.asarray(rng.permutation(n))
        n_val = max(1, round(n * split[1]))
        n_test = max(1, round(n * split[2]))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"split {split} leaves no train blocks out of {n}")
        parts = {
            "train": ids[order[:n_train]],
            "val": ids[order[n_train : n_train + n_val]],
            "test": ids[order[n_train + n_val :]],
        }

        frames = {name: df[df["block_id"].isin(b)] for name, b in parts.items()}
        frames, epsilon_spent = self._release(frames)

        self._fit_scaler(
            pd.concat(
                [
                    self.compute_signals(block)
                    for _, block in frames["train"].groupby("block_id", sort=False)
                ]
            )
        )

        sets = {
            name: self._windows(frame, horizon, overlap)
            for name, frame in frames.items()
        }
        return SplitData(
            train=sets["train"],
            val=sets["val"],
            test=sets["test"],
            feature_names=list(self.available_features),
            scaler_center=self.scaler.center_.copy(),
            scaler_scale=self.scaler.scale_.copy(),
            epsilon_spent=epsilon_spent,
            privacy_mode=self.privacy_mode,
        )

    def _release(
        self, frames: dict[str, pd.DataFrame]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
        """Release each split on its own ledger; report epsilon per split.

        Train first, which fits the generalize classes; val and test apply them.
        No privacy => frames unchanged, ``{}``. A pipeline releases once, by either path.
        """
        if self.privacy is None:
            return frames, {}
        if self._split_released:
            raise BudgetExhausted(_ALREADY_RELEASED)
        if self._instance_released:
            raise BudgetExhausted(_INSTANCE_ALREADY_RELEASED)
        released = {}
        spent = {}
        classes = None
        for split in ("train", "val", "test"):
            manager = self._new_manager(split)
            released[split], fitted = self._release_split(
                frames[split], manager, classes
            )
            if classes is None:
                classes = fitted
            spent[split] = float(manager.budget.spent)
        self._split_released = True
        return released, spent

    def _windows(self, frame: pd.DataFrame, horizon: int, overlap: float) -> WindowSet:
        """Window every block of ``frame`` and attach labels from the event record."""
        parts = []
        for block_id, block in frame.groupby("block_id", sort=False):
            x_block = self._transform_on(block)
            starts = window_starts(len(block), self.seq_len, horizon)
            x = sliding_window_view(x_block, self.seq_len, axis=0)[starts]
            flags = block["is_anomalous"].to_numpy(dtype=bool, copy=True)
            types = block["anomaly_type"].to_numpy(dtype=object, copy=True)
            aid = block["anomaly_id"].to_numpy(copy=True)
            y_fc, mask = forecast_labels(aid, flags, starts, self.seq_len, horizon)
            onset, dominant = window_meta(flags, types, starts, self.seq_len)
            parts.append(
                {
                    "X": x.transpose(0, 2, 1).astype(np.float32),
                    "y_risk": risk_labels(flags, starts, self.seq_len, overlap),
                    "y_forecast": y_fc,
                    "forecast_mask": mask,
                    "block_index": np.full(len(starts), block_id, dtype=np.int64),
                    "onset_offset": onset,
                    "end_row": (starts + self.seq_len - 1).astype(np.int64),
                    "anomaly_type": dominant,
                }
            )
        return WindowSet(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})

    def _fit_on(self, df: pd.DataFrame) -> None:
        """Fit the scaler. ``df`` must already have had privacy applied."""
        self._fit_scaler(self.compute_signals(df))

    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the scaler on a frame that already carries its signal columns."""
        available_features = [f for f in self.features if f in df.columns]
        if not available_features:
            raise ValueError(
                f"None of the specified features found in data: {self.features}"
            )

        X = df[available_features].values
        self.scaler.fit(X)
        self.fitted = True
        self.available_features = available_features

    def _transform_on(self, df: pd.DataFrame) -> np.ndarray:
        """Scale features. ``df`` must already have had privacy applied."""
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        df = self.compute_signals(df)
        X = df[self.available_features].values
        return self.scaler.transform(X)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Create sliding window sequences.

        Args:
            X: Feature array [n_samples, n_features]
            y: Optional labels [n_samples]
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
            if y.ndim != 1:
                raise ValueError(
                    "2-D targets are not supported; build forecast targets with "
                    "data.labels.forecast_labels"
                )
            # Classification labels (take label at sequence end)
            y_seq = np.array(
                [y[i * stride + self.seq_len - 1] for i in range(n_sequences)]
            )

        return X_seq, y_seq

    def process(
        self,
        df: pd.DataFrame,
        labels: np.ndarray | None = None,
        fit: bool = True,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Full pipeline: compute signals, scale, create sequences.

        Args:
            df: Raw dataframe
            labels: Optional labels
            fit: Whether to fit scaler (True for train, False for test)
            stride: Stride for sequence creation

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Transform to features
        if fit:
            X = self.fit_transform(df)
        else:
            X = self.transform(df)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X, labels, stride=stride)

        return X_seq, y_seq


def describe_privacy(data: SplitData, privacy: PrivacyConfig | None) -> str:
    """One line for the CLI: the privacy mode and the epsilon each split spent."""
    if privacy is None:
        return "Privacy: off"
    spent = " ".join(f"{k}={v:.3f}" for k, v in data.epsilon_spent.items())
    line = (
        f"Privacy: mode={data.privacy_mode}; ε spent per split {spent}, each of "
        f"{privacy.epsilon_total:.3f} (parallel composition over disjoint blocks)"
    )
    if data.privacy_mode == "system":
        line += (
            "; not reproducible from seed, set privacy.deterministic_rng for "
            "experiments"
        )
    elif data.privacy_mode == "deterministic":
        line += "; seeded experiment mode, not DP-grade randomness"
    return line
