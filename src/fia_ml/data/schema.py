"""Dataset column schema loaded from f1_dataset_example.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from fia_ml.paths import SCHEMA_CSV

MULTI_VALUE_COLUMNS = frozenset(
    {
        "current_top_4_drivers",
        "positions_of_involved parties",
        "drivers",
        "nationalities",
        "respective_teams",
        "driver_standings",
        "driver_points",
        "construct_standings",
        "construct_points",
        "years_in_sport",
        "superlicense_points_before_incident",
    }
)

LABEL_COLUMNS = frozenset(
    {
        "driver_at_fault",
        "penalty",
        "superlicense_points_added",
        "mentioned_article",
    }
)


def load_schema_columns(schema_path: Path | None = None) -> list[str]:
    path = schema_path or SCHEMA_CSV
    with path.open("rb") as handle:
        first_line = handle.readline().decode("utf-8", errors="replace")
    header = next(csv.reader([first_line]))
    return [col for col in header if col.strip()]


def empty_row() -> dict[str, str]:
    return {col: "" for col in load_schema_columns()}


SCHEMA_COLUMNS = load_schema_columns()
