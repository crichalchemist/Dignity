"""GDELT processing utilities for regime-aware modeling."""

from .alignment import align_candle_to_regime
from .cache import RegimeCache
from .config import (
    GCAM_FEAR_KEYS,
    GDELT_GKG_BASE_URL,
    REGIME_FEATURE_DIM,
    REGIME_FEATURE_NAMES,
    GDELTThemeConfig,
)
from .consolidated_downloader import GDELTDownloader
from .feature_builder import GDELTTimeSeriesBuilder, RegimeFeatureBuilder
from .parser import GDELTRecord

__all__ = [
    "align_candle_to_regime",
    "RegimeCache",
    "GDELT_GKG_BASE_URL",
    "GCAM_FEAR_KEYS",
    "GDELTThemeConfig",
    "REGIME_FEATURE_DIM",
    "REGIME_FEATURE_NAMES",
    "GDELTDownloader",
    "RegimeFeatureBuilder",
    "GDELTTimeSeriesBuilder",
    "GDELTRecord",
]
