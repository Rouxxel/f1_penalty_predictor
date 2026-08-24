"""Basic V1 derived features (no precedent/history)."""

from __future__ import annotations

import pandas as pd


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for V1 training."""
    out = df.copy()

    full_laps = _to_float(out.get("full_laps", pd.Series(dtype=float)))
    lap = _to_float(out.get("lap", pd.Series(dtype=float)))

    out["laps_remaining"] = full_laps - lap
    out["completion_percentage"] = (lap / full_laps.replace(0, pd.NA)) * 100

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

    session = out.get("session", pd.Series(dtype=str)).astype(str).str.lower()
    out["is_race_session"] = session == "race"
    out["is_qualifying"] = session.isin(["qualifying", "sprint_qualifying"])

    return out
