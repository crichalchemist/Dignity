"""Simple on-disk cache for regime feature vectors."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from .config import REGIME_FEATURE_DIM


class RegimeCache:
    def __init__(self, cache_dir: Path | str = Path("data/gdelt_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, bucket_start: datetime) -> Path:
        ts = bucket_start.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
        return self.cache_dir / f"{ts}.npz"

    def get(self, bucket_start: datetime) -> np.ndarray | None:
        path = self._path_for(bucket_start)
        if not path.exists():
            return None
        data = np.load(path)
        arr = data["feat"]
        if arr.shape[-1] != REGIME_FEATURE_DIM:
            return None
        return arr

    def put(self, bucket_start: datetime, feat: np.ndarray) -> None:
        if feat.shape[-1] != REGIME_FEATURE_DIM:
            raise ValueError(f"Expected last dim {REGIME_FEATURE_DIM}, got {feat.shape[-1]}")
        path = self._path_for(bucket_start)
        np.savez_compressed(path, feat=feat.astype(np.float32))

    def __contains__(self, bucket_start: datetime) -> bool:
        return self._path_for(bucket_start).exists()
