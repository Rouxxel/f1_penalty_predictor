"""Exclude label and leakage columns from the feature matrix."""

from __future__ import annotations

from fia_ml.preprocessing.feature_engineering import FORBIDDEN_REDUNDANT_DERIVED

# Columns never used as model inputs (identifiers, labels, outcome leakage).
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "penalty",
        "penalty_severity",
        "driver_at_fault",
        "superlicense_points_added",
        "mentioned_article",
        "incident_classification",
        "incident_id",
        "row_id",
        "document_id",
        "decision_id",
        "investigation",
        "source_file",
        "source_season",
        "drivers",
        "nationalities",
        "respective_teams",
        "driver_standings",
        "driver_points",
        "construct_standings",
        "construct_points",
        "current_top_4_drivers",
    }
)

# Optional / sparse columns dropped in V1 when missingness exceeds threshold.
OPTIONAL_DROP_COLUMNS = frozenset(
    {
        "positions_of_involved parties",
    }
)

MISSINGNESS_DROP_THRESHOLD = 0.30

V1_CATEGORICAL_FEATURES = frozenset(
    {
        "circuit",
        "country",
        "sector",
        "flag",
        "safety_car",
        "track_conditions",
        "weather_conditions",
        "session",
        "incident_type",
        "severity",
        "driver",
        "driver_nationality",
        "driver_team",
        "opponent",
        "opponent_nationality",
        "opponent_team",
    }
)

V1_BOOLEAN_FEATURES = frozenset(
    {
        "same_team",
    }
)

V2_CATEGORICAL_FEATURES = frozenset(
    {
        "race_stage",
    }
)

V2_BOOLEAN_FEATURES = frozenset(
    {
        "is_first_round",
        "is_last_round",
        "title_contender",
        "construct_title_contender",
    }
)

V2_NUMERIC_FEATURES = frozenset(
    {
        "round_progress",
        "points_gap_to_leader",
        "points_gap_to_opponent",
        "points_available_remaining",
        "career_incidents",
        "career_penalties",
        "career_major_penalties",
        "career_incidents_per_100_races",
        "career_penalties_per_100_races",
        "incidents_last_3_races",
        "incidents_last_5_races",
        "penalties_last_3_races",
        "penalties_last_5_races",
        "races_since_last_penalty",
        "races_since_last_incident",
        "precedent_count",
        "precedent_no_penalty_rate",
        "precedent_minor_penalty_rate",
        "precedent_major_penalty_rate",
    }
)

V1_NUMERIC_FEATURES = frozenset(
    {
        "round",
        "season",
        "rounds",
        "num_teams",
        "lap",
        "lap_remaining",
        "full_laps",
        "completion_percentage",
        "first_season",
        "driver_standing",
        "driver_points",
        "driver_construct_standing",
        "driver_construct_points",
        "years_in_sport",
        "superlicense_points_before_incident",
        "opponent_standing",
        "opponent_points",
        "opponent_construct_standing",
        "opponent_construct_points",
        "opponent_years_in_sport",
        "opponent_superlicense_points_before",
        "standing_difference",
        "points_difference",
        "num_drivers",
    }
)


def columns_to_drop_for_missingness(df, threshold: float = MISSINGNESS_DROP_THRESHOLD) -> set[str]:
    drops: set[str] = set()
    for col in OPTIONAL_DROP_COLUMNS:
        if col not in df.columns:
            continue
        missing_rate = df[col].isna().mean() if df[col].dtype != object else (
            df[col].astype(str).str.strip() == ""
        ).mean()
        if missing_rate > threshold:
            drops.add(col)
    return drops


def select_feature_columns(df, *, extra_forbidden: set[str] | None = None) -> list[str]:
    """Return ordered feature column names for X."""
    forbidden = set(FORBIDDEN_FEATURE_COLUMNS) | FORBIDDEN_REDUNDANT_DERIVED
    if extra_forbidden:
        forbidden |= extra_forbidden

    missingness_drops = columns_to_drop_for_missingness(df)
    forbidden |= missingness_drops

    candidates = (
        list(V1_CATEGORICAL_FEATURES)
        + list(V2_CATEGORICAL_FEATURES)
        + list(V1_BOOLEAN_FEATURES)
        + list(V2_BOOLEAN_FEATURES)
        + list(V1_NUMERIC_FEATURES)
        + list(V2_NUMERIC_FEATURES)
    )
    return [col for col in candidates if col in df.columns and col not in forbidden]


def assert_no_leakage(feature_columns: list[str]) -> None:
    leaked = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    if leaked:
        raise ValueError(f"Leakage columns present in feature matrix: {sorted(leaked)}")
