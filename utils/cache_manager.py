"""Feature cache management system for efficient feature computation reuse."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureCacheManager:
    """Manages content-hash based caching for feature computation.

    This cache system ensures that expensive feature computations are only
    performed when the underlying data or configuration changes, dramatically
    improving iteration speed during development and experimentation.
    """

    def __init__(
            self,
            cache_dir: str | Path,
            max_cache_age_days: int = 30,
            enabled: bool = True
    ):
        """Initialize the feature cache manager.

        Parameters
        ----------
        cache_dir : Union[str, Path]
            Directory to store cache files.
        max_cache_age_days : int
            Maximum age of cache files in days. Older files will be cleaned up.
        enabled : bool
            Whether caching is enabled. When False, cache operations are skipped.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age_days = max_cache_age_days
        self.enabled = enabled

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_bytes = 0

        # Cache metadata for cleanup
        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load cache metadata from disk."""
        if not self._metadata_file.exists():
            return {}

        try:
            import json
            with open(self._metadata_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            import json
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _compute_content_hash(
            self,
            raw_data: pd.DataFrame,
            feature_config: Any,
            intrinsic_time: bool = False,
            pair_name: str = ""
    ) -> str:
        """Compute a robust content hash for cache key generation.

        The hash includes:
        - Raw data values and index (not just shape)
        - Feature configuration parameters
        - Intrinsic time conversion flag
        - Pair name for multi-pair caching

        Parameters
        ----------
        raw_data : pd.DataFrame
            The raw dataframe to hash.
        feature_config : Any
            Feature configuration object.
        intrinsic_time : bool
            Whether intrinsic time conversion was used.
        pair_name : str
            Name of the currency pair for multi-pair caching.

        Returns
        -------
        str
            12-character hexadecimal hash string.
        """
        # Hash the raw data content (including index)
        raw_hash_bytes = pd.util.hash_pandas_object(raw_data, index=True).values.tobytes()

        # Hash the feature configuration
        config_hash = str(feature_config).encode()

        # Combine all components
        hash_input = raw_hash_bytes + config_hash + str(intrinsic_time).encode() + pair_name.encode()

        # Generate hash
        cache_hash = hashlib.sha256(hash_input).hexdigest()[:12]
        return cache_hash

    def _get_cache_path(self, cache_key: str, pair_name: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{pair_name}_features_{cache_key}.feather"

    def _is_cache_valid(self, cache_path: Path, cache_key: str) -> bool:
        """Check if a cache file is still valid.

        Parameters
        ----------
        cache_path : Path
            Path to the cache file.
        cache_key : str
            Cache key to validate.

        Returns
        -------
        bool
            True if cache is valid, False otherwise.
        """
        if not cache_path.exists():
            return False

        # Check file age
        file_age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if file_age_days > self.max_cache_age_days:
            logger.debug(f"Cache file {cache_path} is too old ({file_age_days} days)")
            return False

        # Check metadata consistency
        if cache_key in self._metadata:
            expected_stat = cache_path.stat()
            cached_stat = self._metadata[cache_key]

            # Compare size and modification time
            if (expected_stat.st_size != cached_stat.get('size') or
                    expected_stat.st_mtime != cached_stat.get('mtime')):
                logger.debug(f"Cache file {cache_path} metadata mismatch")
                return False

        return True

    def _update_metadata(self, cache_key: str, cache_path: Path) -> None:
        """Update metadata for a cached file."""
        stat = cache_path.stat()
        self._metadata[cache_key] = {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
        self._save_metadata()

    def get_cached_features(
            self,
            raw_data: pd.DataFrame,
            feature_config: Any,
            intrinsic_time: bool = False,
            pair_name: str = ""
    ) -> pd.DataFrame | None:
        """Retrieve cached features if available.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw input data.
        feature_config : Any
            Feature configuration used for caching.
        intrinsic_time : bool
            Whether intrinsic time conversion was used.
        pair_name : str
            Name of the currency pair.

        Returns
        -------
        Optional[pd.DataFrame]
            Cached feature DataFrame if available, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self._compute_content_hash(raw_data, feature_config, intrinsic_time, pair_name)
        cache_path = self._get_cache_path(cache_key, pair_name)

        if self._is_cache_valid(cache_path, cache_key):
            try:
                feature_df = pd.read_feather(cache_path)
                self.cache_hits += 1
                self.cache_size_bytes += cache_path.stat().st_size
                logger.info(f"[cache] loaded pre-computed features from {cache_path.name}")
                return feature_df
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_path}: {e}")
                # Remove corrupted cache file
                if cache_path.exists():
                    cache_path.unlink()

        self.cache_misses += 1
        return None

    def save_features_to_cache(
            self,
            feature_df: pd.DataFrame,
            cache_key: str,
            pair_name: str
    ) -> None:
        """Save computed features to cache.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature DataFrame to cache.
        cache_key : str
            Cache key generated by _compute_content_hash.
        pair_name : str
            Name of the currency pair.
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(cache_key, pair_name)

        try:
            feature_df.to_feather(cache_path)
            self._update_metadata(cache_key, cache_path)
            logger.info(f"[cache] saved computed features to {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def cleanup_expired_cache(self) -> tuple[int, int]:
        """Clean up expired cache files and return cleanup statistics.

        Returns
        -------
        Tuple[int, int]
            (files_removed, bytes_freed)
        """
        files_removed = 0
        bytes_freed = 0

        if not self.enabled:
            return files_removed, bytes_freed

        current_time = datetime.now()

        for cache_file in self.cache_dir.glob("*.feather"):
            if cache_file.name == "cache_metadata.json":
                continue

            try:
                file_age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.days > self.max_cache_age_days:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    files_removed += 1
                    bytes_freed += file_size

                    # Remove from metadata
                    key = cache_file.stem.replace("_features_", "_")
                    if key in self._metadata:
                        del self._metadata[key]

            except Exception as e:
                logger.warning(f"Failed to cleanup cache file {cache_file}: {e}")

        # Save updated metadata
        self._save_metadata()

        if files_removed > 0:
            logger.info(f"[cache] cleaned up {files_removed} expired files, freed {bytes_freed} bytes")

        return files_removed, bytes_freed

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing cache statistics.
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1) * 100

        # Get disk usage
        cache_files = list(self.cache_dir.glob("*.feather"))
        cache_files = [f for f in cache_files if f.name != "cache_metadata.json"]
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'enabled': self.enabled,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'cache_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_files_count': len(cache_files),
            'max_cache_age_days': self.max_cache_age_days
        }

    def clear_all_cache(self) -> int:
        """Clear all cache files and return number of files removed.

        Returns
        -------
        int
            Number of cache files removed.
        """
        if not self.enabled:
            return 0

        files_removed = 0

        for cache_file in self.cache_dir.glob("*.feather"):
            if cache_file.name == "cache_metadata.json":
                continue

            try:
                cache_file.unlink()
                files_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        # Clear metadata
        self._metadata.clear()
        self._save_metadata()

        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_bytes = 0

        logger.info(f"[cache] cleared {files_removed} cache files")
        return files_removed


def get_default_cache_manager() -> FeatureCacheManager:
    """Get the default feature cache manager instance.

    Returns
    -------
    FeatureCacheManager
        Configured cache manager with sensible defaults.
    """
    cache_dir = Path(__file__).resolve().parents[2] / "output_central" / "cache"

    return FeatureCacheManager(
        cache_dir=cache_dir,
        max_cache_age_days=30,
        enabled=True
    )
