"""Base data agent providing common dataset splitting, normalisation and
window‑generation utilities.

Both the single‑task and multitask pipelines share the same sliding‑window
logic: extract ``t_in`` historic feature rows, compute a primary target (either
classification or regression), and optionally compute auxiliary targets such
as ``max_return`` or ``topk_returns``. The concrete agent subclasses only need to
define how the *primary* target is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.config import DataConfig


# ------------------------------------------------------------------
# Helper functions (migrated from legacy agent_data.py)
# ------------------------------------------------------------------
def _select_range(df: pd.DataFrame, time_col: str, date_range: tuple[str, str] | None) -> pd.DataFrame:
    """Select rows within a datetime range.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a datetime column.
    time_col : str
        Name of the datetime column.
    date_range : Optional[Tuple[str, str]]
        (start, end) date range as strings. If None, returns full dataframe.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe within the specified range.
    """
    if date_range is None:
        return df
    start, end = date_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return df.loc[(df[time_col] >= start) & (df[time_col] <= end)]


def _compute_future_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Compute log return between current and future close price.

    Parameters
    ----------
    close : pd.Series
        Series of close prices.
    horizon : int
        Number of periods to look ahead.

    Returns
    -------
    pd.Series
        Log returns: ln(close[t+horizon] / close[t])
    """
    future = close.shift(-horizon)
    return np.log(future / close)


def _label_from_return(log_ret: float, flat_threshold: float) -> int:
    """Convert log return to classification label (down/flat/up).

    Parameters
    ----------
    log_ret : float
        Log return value.
    flat_threshold : float
        Threshold for flat classification.

    Returns
    -------
    int
        0 (down), 1 (flat), or 2 (up)
    """
    if log_ret > flat_threshold:
        return 2  # up
    if log_ret < -flat_threshold:
        return 0  # down
    return 1  # flat


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class SequenceDataset(Dataset):
    """Dataset holding pre‑computed sequences and a dict of target arrays."""

    def __init__(self, sequences: np.ndarray, targets: dict[str, np.ndarray], target_type: str):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = {}

        # Convert all targets to tensors with appropriate dtypes
        for key, value in targets.items():
            # Classification targets end with "_class" or are named "sell_now" or "primary"
            # Regression targets end with "_reg" or are price/return related
            is_classification = (
                    key.endswith("_class") or
                    key == "sell_now" or
                    (key == "primary" and target_type == "classification")
            )
            dtype = torch.long if is_classification else torch.float32
            self.targets[key] = torch.as_tensor(value, dtype=dtype)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], {k: v[idx] for k, v in self.targets.items()}


class BaseDataAgent:
    """Common functionality for both single‑task and multitask data agents.

    Sub‑classes must implement ``_create_primary_target`` which receives the
    future log‑return for a given index and returns the appropriate primary
    label/value based on ``DataConfig.target_type`` and ``flat_threshold``.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        # Check target_type if present (SingleTaskDataConfig has it, MultiTaskDataConfig doesn't)
        if hasattr(cfg, 'target_type') and cfg.target_type not in {"classification", "regression"}:
            raise ValueError("target_type must be 'classification' or 'regression'")
        self.norm_stats: NormalizationStats | None = None

    # ------------------------------------------------------------------
    # Splitting & normalisation helpers (identical to the previous DataAgent).
    # ------------------------------------------------------------------
    def split_dataframe(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        time_col = self.cfg.datetime_column
        df = df.sort_values(time_col).reset_index(drop=True)
        return {
            "train": _select_range(df, time_col, self.cfg.train_range),
            "val": _select_range(df, time_col, self.cfg.val_range),
            "test": _select_range(df, time_col, self.cfg.test_range),
        }

    def fit_normalization(self, train_df: pd.DataFrame, feature_cols: list[str]) -> NormalizationStats:
        data = train_df[feature_cols].to_numpy(dtype=np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        self.norm_stats = NormalizationStats(mean=mean, std=std)
        return self.norm_stats

    # ------------------------------------------------------------------
    # Primary‑target creation – abstract for subclasses.
    # ------------------------------------------------------------------
    def _create_primary_target(self, log_ret: float) -> float | int:
        """Return a classification label or regression value.

        Sub‑classes should honour ``self.cfg.target_type`` and ``flat_threshold``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Window building – used by both agents.
    # ------------------------------------------------------------------
    def _build_windows(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            norm_stats: NormalizationStats,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        t_in, t_out = self.cfg.t_in, self.cfg.t_out
        lookahead = self.cfg.lookahead_window or t_out
        top_k = max(1, self.cfg.top_k_predictions)
        # Normalise features.
        features = norm_stats.apply(df[feature_cols].to_numpy(dtype=np.float32))
        # Future log‑return series.
        future_log_ret = (
                np.log(df["close"].to_numpy()[t_out:])
                - np.log(df["close"].to_numpy()[:-t_out])
        )
        # Pad to align indices with original df length.
        future_log_ret = np.concatenate([np.full(t_out, np.nan), future_log_ret])

        log_close = np.log(df["close"].to_numpy())

        sequences: list[np.ndarray] = []
        primary_targets: list[float | int] = []
        max_return_targets: list[float] = []
        topk_returns_targets: list[np.ndarray] = []
        topk_prices_targets: list[np.ndarray] = []
        sell_now_targets: list[int] = []

        last_idx = len(df) - max(t_out, lookahead)
        for idx in range(t_in - 1, last_idx):
            seq = features[idx - t_in + 1: idx + 1]
            target_ret = future_log_ret[idx]
            if not np.isfinite(target_ret):
                continue
            primary_targets.append(self._create_primary_target(target_ret))

            # Auxiliary targets – unchanged from original implementation.
            future_returns = log_close[idx + 1: idx + lookahead + 1] - log_close[idx]
            max_future_return = float(np.max(future_returns))
            sorted_returns = np.sort(future_returns)[::-1]
            topk = sorted_returns[:top_k]
            if len(topk) < top_k:
                topk = np.pad(topk, (0, top_k - len(topk)), constant_values=sorted_returns[-1])
            topk_prices = np.exp(topk) * df["close"].iloc[idx]

            sequences.append(seq)
            max_return_targets.append(max_future_return)
            topk_returns_targets.append(topk)
            topk_prices_targets.append(topk_prices)
            if self.cfg.predict_sell_now:
                sell_now_targets.append(int(max_future_return <= 0.0))

        targets: dict[str, np.ndarray] = {
            "primary": np.array(primary_targets),
            "max_return": np.array(max_return_targets),
            "topk_returns": np.stack(topk_returns_targets),
            "topk_prices": np.stack(topk_prices_targets),
        }
        if self.cfg.predict_sell_now:
            targets["sell_now"] = np.array(sell_now_targets)

        return np.stack(sequences), targets

    def build_datasets(self, df: pd.DataFrame) -> dict[str, SequenceDataset]:
        # Determine feature columns (exclude datetime and source file).
        feature_cols = self.cfg.feature_columns or [c for c in df.columns if
                                                    c not in {self.cfg.datetime_column, "source_file"}]
        # Split and fit normalisation on the training split.
        splits = self.split_dataframe(df)
        train_df = splits["train"]
        self.fit_normalization(train_df, feature_cols)
        assert self.norm_stats is not None
        datasets: dict[str, SequenceDataset] = {}
        # Use target_type if available, otherwise default to "classification" (for multitask)
        target_type = getattr(self.cfg, 'target_type', 'classification')
        for name, split_df in splits.items():
            sequences, targets = self._build_windows(split_df, feature_cols, self.norm_stats)
            datasets[name] = SequenceDataset(sequences, targets, target_type)
        return datasets

    @staticmethod
    def build_dataloaders(
            datasets: dict[str, SequenceDataset],
            batch_size: int,
            num_workers: int = 0,
            pin_memory: bool = False,
            prefetch_factor: int | None = None,
            persistent_workers: bool = False,
    ) -> dict[str, torch.utils.data.DataLoader]:
        """Build PyTorch DataLoaders from datasets with performance optimizations.

        Parameters
        ----------
        datasets : Dict[str, SequenceDataset]
            Dictionary mapping split names to SequenceDataset instances.
        batch_size : int
            Batch size for training/validation/test.
        num_workers : int, optional
            Number of worker processes for data loading. Default 0 (single-process).
            **Performance tip**: Set to 2-4 for 3-5x faster data loading.
        pin_memory : bool, optional
            If True, tensors are copied to CUDA pinned memory before returning.
            Useful for GPU training. Default False.
        prefetch_factor : int | None, optional
            Number of batches to prefetch per worker. Default None (uses PyTorch default of 2).
            **Performance tip**: Increase to 4-8 to hide I/O latency.
        persistent_workers : bool, optional
            If True, workers remain alive between epochs. Reduces worker startup overhead.
            Default False.

        Returns
        -------
        Dict[str, torch.utils.data.DataLoader]
            Dictionary mapping split names to DataLoader instances.

        Performance Notes
        -----------------
        For optimal performance with GPU training:
        - Set num_workers=2-4 (3-5x speedup over single-process)
        - Set pin_memory=True when using CUDA
        - Set prefetch_factor=4 to prefetch more batches
        - Set persistent_workers=True to avoid worker restart overhead

        Example:
            loaders = agent.build_dataloaders(
                datasets,
                batch_size=64,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4,
                persistent_workers=True,
            )
        """
        loaders = {}
        for name, ds in datasets.items():
            loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": (name == "train"),
                "num_workers": num_workers,
                "drop_last": False,
                "pin_memory": pin_memory,
            }

            # Only add prefetch_factor and persistent_workers if num_workers > 0
            if num_workers > 0:
                if prefetch_factor is not None:
                    loader_kwargs["prefetch_factor"] = prefetch_factor
                if persistent_workers:
                    loader_kwargs["persistent_workers"] = persistent_workers

            loaders[name] = torch.utils.data.DataLoader(ds, **loader_kwargs)

        return loaders


# ------------------------------------------------------------------
# Data quality helpers: timezone normalization and deduplication
# ------------------------------------------------------------------

def ensure_utc_timezone(df: pd.DataFrame, datetime_col: str = "datetime",
                        assumed_tz: str = "America/Chicago") -> pd.DataFrame:
    """Ensure datetime column is timezone-aware UTC.

    HistData CSVs typically use Central Time without explicit timezone info.
    This function localizes naive datetimes to the assumed timezone, then
    converts to UTC for consistency across all data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime column.
    datetime_col : str, optional
        Name of the datetime column. Default: 'datetime'.
    assumed_tz : str, optional
        Timezone to assume for naive datetimes. Default: 'America/Chicago'.

    Returns
    -------
    pd.DataFrame
        DataFrame with UTC-aware datetime column.
    """
    df = df.copy()
    dt = df[datetime_col]

    if dt.dt.tz is None:
        # Localize naive datetime to assumed timezone
        dt = dt.dt.tz_localize(assumed_tz, ambiguous='NaT', nonexistent='NaT')

    # Convert to UTC
    df[datetime_col] = dt.dt.tz_convert('UTC')
    return df


def deduplicate_on_datetime(df: pd.DataFrame, datetime_col: str = "datetime",
                            keep: str = "first", logger_func=None) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    datetime_col : str, optional
        Name of the datetime column. Default: 'datetime'.
    keep : {'first', 'last'}, optional
        Which duplicate to keep. Default: 'first'.
    logger_func : callable, optional
        Logger function to report duplicates. Default: None.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[datetime_col], keep=keep).reset_index(drop=True)
    after = len(df)

    if before > after and logger_func:
        logger_func(f"Removed {before - after} duplicate timestamps")

    return df
