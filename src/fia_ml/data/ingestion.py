"""Load and combine processed season CSV files for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fia_ml.data.schema import SCHEMA_COLUMNS
from fia_ml.paths import PROJECT_ROOT


def _is_legend_row(row: pd.Series) -> bool:
    incident_id = str(row.get("incident_id", "")).strip()
    drivers = str(row.get("drivers", "")).strip()
    return incident_id.startswith("*") or drivers == "**"


def resolve_input_paths(
    *,
    explicit_paths: list[Path] | None = None,
    seasons: list[int] | None = None,
    csv_glob: str = "dataset/csv/processed_*.csv",
) -> list[Path]:
    if explicit_paths:
        return [Path(p) for p in explicit_paths]

    csv_dir = PROJECT_ROOT / "dataset" / "csv"
    if seasons:
        paths = [csv_dir / f"processed_{year}.csv" for year in seasons]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing processed CSV files: {missing}")
        return paths

    paths = sorted(PROJECT_ROOT.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched {csv_glob}")
    return paths


def load_processed_seasons(
    paths: list[Path],
    *,
    schema_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read, validate, and combine processed_{season}.csv files."""
    expected = schema_columns or SCHEMA_COLUMNS
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = df[expected].copy()
        df["source_file"] = str(path.relative_to(PROJECT_ROOT))
        season = path.stem.replace("processed_", "")
        df["source_season"] = season
        df = df[~df.apply(_is_legend_row, axis=1)]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce").astype("Int64")
    combined["round"] = pd.to_numeric(combined["round"], errors="coerce").astype("Int64")
    combined = combined.sort_values(
        ["season", "round", "incident_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined
