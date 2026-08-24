"""Basic V1 derived features (no precedent/history)."""

from __future__ import annotations

import pandas as pd

# Columns that must not be re-derived from a single source categorical/numeric field.
# See documentation/FIA_stewarding_dataset_feature_specification.md §2.1.
FORBIDDEN_REDUNDANT_DERIVED = frozenset(
    {
        "is_race_session",
        "is_qualifying",
        "top_4_driver",
        "top_4_opponent",
        "laps_remaining",  # schema uses lap_remaining
    }
)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fill_lap_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Fill schema lap fields only when missing — never duplicate under another name."""
    out = df.copy()
    full_laps = _to_float(out.get("full_laps", pd.Series(dtype=float)))
    lap = _to_float(out.get("lap", pd.Series(dtype=float)))

    if "lap_remaining" not in out.columns:
        out["lap_remaining"] = pd.NA
    lap_remaining = _to_float(out["lap_remaining"])
    missing_remaining = lap_remaining.isna() & lap.notna() & full_laps.notna()
    out.loc[missing_remaining, "lap_remaining"] = full_laps[missing_remaining] - lap[missing_remaining]

    if "completion_percentage" not in out.columns:
        out["completion_percentage"] = pd.NA
    completion = _to_float(out["completion_percentage"])
    valid_full = full_laps.replace(0, pd.NA)
    missing_completion = completion.isna() & lap.notna() & valid_full.notna()
    out.loc[missing_completion, "completion_percentage"] = (
        lap[missing_completion] / valid_full[missing_completion]
    ) * 100

    return out


def add_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add relational derived columns for V1 training."""
    out = df.copy()
    out = _fill_lap_progression(out)

    if "num_drivers" in out.columns:
        out["num_drivers"] = _to_float(out["num_drivers"])
    else:
        out["num_drivers"] = out.groupby("incident_id")["driver"].transform("count")

    out["same_team"] = (
        out["driver_team"].notna()
        & out["opponent_team"].notna()
        & (out["driver_team"] == out["opponent_team"])
    )

    driver_standing = _to_float(out.get("driver_standing", pd.Series(dtype=float)))
    opponent_standing = _to_float(out.get("opponent_standing", pd.Series(dtype=float)))
    driver_points = _to_float(out.get("driver_points", pd.Series(dtype=float)))
    opponent_points = _to_float(out.get("opponent_points", pd.Series(dtype=float)))

    out["standing_difference"] = driver_standing - opponent_standing
    out["points_difference"] = driver_points - opponent_points

    leaked = FORBIDDEN_REDUNDANT_DERIVED & set(out.columns)
    if leaked:
        raise ValueError(
            f"Redundant derived columns present: {sorted(leaked)}. "
            "Use schema source columns (session, driver_standing, lap_remaining) instead."
        )

    return out
