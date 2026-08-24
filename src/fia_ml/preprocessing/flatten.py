"""Incident-centric rows → one row per driver under investigation."""

from __future__ import annotations

import pandas as pd

from fia_ml.data.schema import MULTI_VALUE_COLUMNS

# Per-driver fields produced from multi-value incident columns.
_DRIVER_FIELD_MAP: list[tuple[str, str]] = [
    ("nationalities", "driver_nationality"),
    ("respective_teams", "driver_team"),
    ("driver_standings", "driver_standing"),
    ("driver_points", "driver_points"),
    ("construct_standings", "driver_construct_standing"),
    ("construct_points", "driver_construct_points"),
    ("years_in_sport", "years_in_sport"),
    ("superlicense_points_before_incident", "superlicense_points_before_incident"),
]

_OPPONENT_FIELD_MAP: list[tuple[str, str]] = [
    ("nationalities", "opponent_nationality"),
    ("respective_teams", "opponent_team"),
    ("driver_standings", "opponent_standing"),
    ("driver_points", "opponent_points"),
    ("construct_standings", "opponent_construct_standing"),
    ("construct_points", "opponent_construct_points"),
    ("years_in_sport", "opponent_years_in_sport"),
    ("superlicense_points_before_incident", "opponent_superlicense_points_before"),
]

_SHARED_INCIDENT_COLUMNS = [
    "incident_id",
    "season",
    "round",
    "circuit",
    "country",
    "first_season",
    "rounds",
    "num_teams",
    "lap",
    "lap_remaining",
    "full_laps",
    "completion_percentage",
    "sector",
    "flag",
    "safety_car",
    "track_conditions",
    "weather_conditions",
    "session",
    "incident_type",
    "severity",
    "positions_of_involved parties",
    "num_drivers",
    "current_top_4_drivers",
    "investigation",
    "incident_classification",
    "driver_at_fault",
    "penalty",
    "superlicense_points_added",
    "mentioned_article",
    "penalty_severity",
    "source_file",
    "source_season",
]


def split_multi_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",")]


def _validate_alignment(row: pd.Series, drivers: list[str]) -> None:
    n = len(drivers)
    for col in MULTI_VALUE_COLUMNS:
        if col == "current_top_4_drivers" or col == "positions_of_involved parties":
            continue
        if col not in row.index:
            continue
        values = split_multi_value(row.get(col, ""))
        if values and len(values) != n:
            raise ValueError(
                f"incident {row['incident_id']}: column {col!r} has {len(values)} values "
                f"but drivers has {n}"
            )


def _pick_value(values: list[str], index: int) -> str | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0] or None
    if index < len(values):
        val = values[index]
        return val if val else None
    return None


def flatten_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each incident to one row per investigated driver."""
    rows: list[dict[str, object]] = []
    misaligned: list[str] = []

    for _, incident in df.iterrows():
        drivers = split_multi_value(incident.get("drivers", ""))
        if not drivers:
            continue

        try:
            _validate_alignment(incident, drivers)
        except ValueError:
            misaligned.append(str(incident["incident_id"]))
            continue

        multi_cache = {
            col: split_multi_value(incident.get(col, ""))
            for col in MULTI_VALUE_COLUMNS
            if col in incident.index
        }
        top4 = set(split_multi_value(incident.get("current_top_4_drivers", "")))

        for idx, driver in enumerate(drivers):
            row: dict[str, object] = {
                col: incident[col]
                for col in _SHARED_INCIDENT_COLUMNS
                if col in incident.index
            }
            row["row_id"] = f"{incident['incident_id']}_{driver}"
            row["driver"] = driver

            for src_col, dest_col in _DRIVER_FIELD_MAP:
                row[dest_col] = _pick_value(multi_cache.get(src_col, []), idx)

            opponents = [d for i, d in enumerate(drivers) if i != idx]
            if len(opponents) == 1:
                opp_idx = 1 - idx if len(drivers) == 2 else 0
                row["opponent"] = opponents[0]
                for src_col, dest_col in _OPPONENT_FIELD_MAP:
                    row[dest_col] = _pick_value(multi_cache.get(src_col, []), opp_idx)
            else:
                row["opponent"] = None
                for _, dest_col in _OPPONENT_FIELD_MAP:
                    row[dest_col] = None

            row["top_4_driver"] = driver in top4
            row["top_4_opponent"] = (
                row["opponent"] in top4 if row.get("opponent") else False
            )

            rows.append(row)

    if misaligned:
        # Keep pipeline running; misaligned incidents are skipped (same as review queue intent).
        pass

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
