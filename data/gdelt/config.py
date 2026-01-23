"""Configuration and constants for the GDELT ingestion pipeline."""
from collections.abc import Sequence
from dataclasses import dataclass

GDELT_GKG_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
GDELT_GKG_EXTENSION = ".gkg.csv.zip"
GDELT_TIME_DELTA_MINUTES = 15


def get_gdelt_bucket_minutes() -> int:
    """Return the default bucket width used for aligning GDELT regimes."""

    return GDELT_TIME_DELTA_MINUTES


# Theme groupings for regime feature construction.
@dataclass(frozen=True)
class GDELTThemeConfig:
    econ_policy_themes: Sequence[str] = (
        "ECON_INFLATION",
        "ECON_STOCKMARKET",
        "ECON_TRADE",
        "ECON_UNEMPLOYMENT",
        "DEBT",
        "AUSTERITY",
    )
    central_bank_themes: Sequence[str] = (
        "CENTRAL_BANK",
        "INTEREST_RATE",
        "MONETARY_POLICY",
    )
    conflict_themes: Sequence[str] = (
        "ARMEDCONFLICT",
        "WAR",
        "TERROR",
        "MILITARY",
        "COUP",
    )
    protest_themes: Sequence[str] = (
        "PROTEST",
        "RIOT",
        "CIVIL_UNREST",
        "STRIKE",
    )
    policy_regulation_themes: Sequence[str] = (
        "REGULATION",
        "SANCTIONS",
        "TRADE_RESTRICTIONS",
        "TARIFFS",
    )
    financial_crisis_themes: Sequence[str] = (
        "DEBT_CRISIS",
        "BANK_FAILURE",
        "BAILOUT",
        "DEFAULT",
    )


COUNTS_OF_INTEREST: Sequence[str] = (
    "KILL",
    "PROTEST",
    "KIDNAP",
    "SEIZE",
)

GCAM_FEAR_KEYS: Sequence[str] = (
    "c14.1",
    "c14.2",
)

G10_COUNTRY_CODES: Sequence[str] = (
    "US",
    "GB",
    "FR",
    "DE",
    "JP",
    "CA",
    "AU",
    "NZ",
    "CH",
    "IT",
)

EM_COUNTRY_CODES: Sequence[str] = (
    "BR",
    "ZA",
    "MX",
    "TR",
    "IN",
    "ID",
    "MY",
    "CL",
    "CO",
    "KR",
)

REGIME_FEATURE_NAMES: list[str] = [
    "econ_policy_intensity",
    "central_bank_intensity",
    "conflict_intensity",
    "protest_unrest_intensity",
    "policy_regulation_intensity",
    "financial_crisis_intensity",
    "kill_events_normalized",
    "protestor_counts_normalized",
    "kidnap_events_normalized",
    "seize_events_normalized",
    "avg_tone_scaled",
    "polarity_scaled",
    "activity_density_scaled",
    "fear_anxiety_scaled",
    "g10_event_share",
    "em_event_share",
    "geo_concentration_index",
    "region_conflict_ratio",
]

REGIME_FEATURE_DIM = len(REGIME_FEATURE_NAMES)

# Per-feature reference counts used for log scaling; tweak based on historical maxima.
DEFAULT_MAX_COUNT_REF: dict[str, float] = {
    "themes": 50.0,
    "counts": 25.0,
}
