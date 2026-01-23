"""Enhanced GDELT feature builder for time series analysis and ML training."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from .config import (
    COUNTS_OF_INTEREST,
    DEFAULT_MAX_COUNT_REF,
    EM_COUNTRY_CODES,
    G10_COUNTRY_CODES,
    GCAM_FEAR_KEYS,
    REGIME_FEATURE_DIM,
    GDELTThemeConfig,
)
from .parser import GDELTRecord


class GDELTTimeSeriesBuilder:
    """Enhanced GDELT feature builder for time series analysis and ML training."""

    def __init__(self):
        self.numeric_cols = [
            'NumMentions', 'NumSources', 'NumArticles',
            'AvgTone', 'GoldsteinScale', 'Actor1Geo_Lat',
            'Actor1Geo_Long', 'Actor2Geo_Lat', 'Actor2Geo_Long'
        ]

    def build_timeseries_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert GDELT events to ML-ready time-series format"""
        df = raw_data.copy()

        # Ensure datetime index
        if 'SQLDATE' in df.columns:
            df['SQLDATE'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            df = df.set_index('SQLDATE')
        elif 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')

        # Daily aggregation with statistical measures
        features = df.groupby(df.index.date).agg({
            'AvgTone': ['mean', 'std', 'count', 'min', 'max'] if 'AvgTone' in df.columns else ['count'],
            'GoldsteinScale': ['mean', 'std', 'min', 'max'] if 'GoldsteinScale' in df.columns else ['count'],
            'NumMentions': ['sum', 'mean'] if 'NumMentions' in df.columns else ['count'],
            'NumArticles': ['sum', 'count'] if 'NumArticles' in df.columns else ['count'],
            'EventCode': 'nunique' if 'EventCode' in df.columns else 'count',
            'Actor1CountryCode': 'nunique' if 'Actor1CountryCode' in df.columns else 'count',
        }).fillna(0)

        # Flatten columns
        features.columns = ['_'.join(str(col).strip()) for col in features.columns]

        # Add derived mathematical features
        features = self._add_derived_features(features)

        return features

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mathematically derived features"""
        # Safely handle tone normalization
        if 'AvgTone_mean' in df.columns:
            tone_mean = df['AvgTone_mean']
            tone_std = tone_mean.std()
            if tone_std > 0:
                df['tone_zscore'] = (tone_mean - tone_mean.mean()) / tone_std
            else:
                df['tone_zscore'] = 0.0

        # Safely handle conflict normalization
        if 'GoldsteinScale_mean' in df.columns:
            conflict_mean = df['GoldsteinScale_mean']
            conflict_std = conflict_mean.std()
            if conflict_std > 0:
                df['conflict_zscore'] = (conflict_mean - conflict_mean.mean()) / conflict_std
            else:
                df['conflict_zscore'] = 0.0

        # Log transforms for count data
        if 'NumMentions_sum' in df.columns:
            df['mentions_log'] = np.log1p(df['NumMentions_sum'])
        if 'NumArticles_sum' in df.columns:
            df['articles_log'] = np.log1p(df['NumArticles_sum'])

        # Volatility measures
        if 'AvgTone_std' in df.columns and 'AvgTone_mean' in df.columns:
            df['tone_volatility'] = df['AvgTone_std'] / (df['AvgTone_mean'].abs() + 1e-8)

        # Coverage intensity
        if 'NumArticles_sum' in df.columns and 'NumMentions_sum' in df.columns:
            df['coverage_intensity'] = df['NumArticles_sum'] / (df['NumMentions_sum'] + 1)

        return df


class RegimeFeatureBuilder:
    def __init__(
            self,
            theme_cfg: GDELTThemeConfig | None = None,
            max_count_ref: dict[str, float] | None = None,
    ) -> None:
        self.theme_cfg = theme_cfg or GDELTThemeConfig()
        self.max_count_ref = {**DEFAULT_MAX_COUNT_REF, **(max_count_ref or {})}

    def build_features(
            self, records: Iterable[GDELTRecord], bucket_start: datetime, bucket_end: datetime
    ) -> np.ndarray:
        relevant_records: list[GDELTRecord] = [
            r for r in records if bucket_start <= r.datetime < bucket_end
        ]
        if not relevant_records:
            return np.zeros(REGIME_FEATURE_DIM, dtype=np.float32)

        theme_counts = self._count_themes(relevant_records)
        count_metrics = self._aggregate_counts(relevant_records)
        tone_metrics = self._aggregate_tone(relevant_records)
        geo_metrics = self._aggregate_geo(relevant_records, theme_counts)

        feature_vector = np.array(
            [
                self._log_normalize(theme_counts["econ_policy"], self.max_count_ref["themes"]),
                self._log_normalize(theme_counts["central_bank"], self.max_count_ref["themes"]),
                self._log_normalize(theme_counts["conflict"], self.max_count_ref["themes"]),
                self._log_normalize(theme_counts["protest"], self.max_count_ref["themes"]),
                self._log_normalize(theme_counts["policy_regulation"], self.max_count_ref["themes"]),
                self._log_normalize(theme_counts["financial_crisis"], self.max_count_ref["themes"]),
                self._log_normalize(count_metrics.get("KILL", 0.0), self.max_count_ref["counts"]),
                self._log_normalize(count_metrics.get("PROTEST", 0.0), self.max_count_ref["counts"]),
                self._log_normalize(count_metrics.get("KIDNAP", 0.0), self.max_count_ref["counts"]),
                self._log_normalize(count_metrics.get("SEIZE", 0.0), self.max_count_ref["counts"]),
                tone_metrics["avg_tone"],
                tone_metrics["polarity"],
                tone_metrics["activity_density"],
                tone_metrics["fear_anxiety"],
                geo_metrics["g10_share"],
                geo_metrics["em_share"],
                geo_metrics["concentration"],
                geo_metrics["region_conflict_ratio"],
            ],
            dtype=np.float32,
        )
        return feature_vector

    def _count_themes(self, records: Iterable[GDELTRecord]) -> dict[str, float]:
        cfg = self.theme_cfg
        buckets = defaultdict(float)
        for rec in records:
            themes = set(rec.themes)
            buckets["econ_policy"] += len(themes & set(cfg.econ_policy_themes))
            buckets["central_bank"] += len(themes & set(cfg.central_bank_themes))
            buckets["conflict"] += len(themes & set(cfg.conflict_themes))
            buckets["protest"] += len(themes & set(cfg.protest_themes))
            buckets["policy_regulation"] += len(themes & set(cfg.policy_regulation_themes))
            buckets["financial_crisis"] += len(themes & set(cfg.financial_crisis_themes))
        return buckets

    @staticmethod
    def _aggregate_counts(records: Iterable[GDELTRecord]) -> Counter:
        totals: Counter[str] = Counter()
        for rec in records:
            for count in rec.counts:
                if count.type in COUNTS_OF_INTEREST:
                    totals[count.type] += count.value
        return totals

    @staticmethod
    def _aggregate_tone(records: Iterable[GDELTRecord]) -> dict[str, float]:
        if not records:
            return {"avg_tone": 0.0, "polarity": 0.0, "activity_density": 0.0, "fear_anxiety": 0.0}

        tone_values = np.array([[r.tone.tone, r.tone.polarity, r.tone.activity_density] for r in records])
        avg_tone, polarity, activity_density = tone_values.mean(axis=0)

        fear_scores = []
        for rec in records:
            metrics = [rec.gcam.get(k, 0.0) for k in GCAM_FEAR_KEYS]
            if metrics:
                fear_scores.append(np.mean(metrics))
        fear_anxiety = float(np.mean(fear_scores)) if fear_scores else 0.0

        return {
            "avg_tone": float(np.tanh(avg_tone / 10.0)),
            "polarity": float(0.5 * (np.tanh(polarity / 5.0) + 1.0)),
            "activity_density": float(0.5 * (np.tanh(activity_density / 5.0) + 1.0)),
            "fear_anxiety": float(0.5 * (np.tanh(fear_anxiety / 2.0) + 1.0)),
        }

    def _aggregate_geo(self, records: Iterable[GDELTRecord], theme_counts: dict[str, float]) -> dict[str, float]:
        country_counts: Counter[str] = Counter()
        for rec in records:
            for loc in rec.locations:
                if loc.country_code:
                    country_counts[loc.country_code] += 1

        total_events = sum(country_counts.values())
        if total_events == 0:
            return {"g10_share": 0.0, "em_share": 0.0, "concentration": 0.0, "region_conflict_ratio": 0.0}

        g10_events = sum(country_counts[c] for c in G10_COUNTRY_CODES)
        em_events = sum(country_counts[c] for c in EM_COUNTRY_CODES)
        concentration = sum((count / total_events) ** 2 for count in country_counts.values())

        em_conflict = theme_counts.get("conflict", 0.0)
        region_conflict_ratio = 0.0
        if em_events > 0:
            region_conflict_ratio = min(em_conflict / em_events, 1.0)

        return {
            "g10_share": g10_events / total_events,
            "em_share": em_events / total_events,
            "concentration": concentration,
            "region_conflict_ratio": region_conflict_ratio,
        }

    @staticmethod
    def _log_normalize(value: float, max_ref: float) -> float:
        if value <= 0 or max_ref <= 0:
            return 0.0
        return math.log1p(value) / math.log1p(max_ref)
