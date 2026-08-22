"""Shared helpers for dataset enrichment stages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.utils import secure_file_io as sio


def is_blank(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def load_meta(cfg: PipelineConfig) -> dict[str, dict[str, Any]]:
    meta_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.meta.json"
    if not meta_path.exists():
        return {}
    rows = sio.read_json(meta_path)
    return {row["incident_id"]: row for row in rows}


def map_event_to_round(
    event: str,
    calendar: list[str],
    event_to_circuit: dict[str, str],
) -> tuple[int | None, str | None]:
    """Return (round_number, circuit_slug) for an FIA event name."""
    if not event:
        return None, None

    circuit_slug = event_to_circuit.get(event)
    if not circuit_slug:
        normalized = event.lower().replace("grand prix", "").strip()
        for event_name, slug in event_to_circuit.items():
            if normalized and normalized in event_name.lower():
                circuit_slug = slug
                break

    if not circuit_slug:
        return None, None

    for index, calendar_slug in enumerate(calendar):
        if calendar_slug == circuit_slug:
            return index + 1, circuit_slug

    return None, circuit_slug


def slugify_nationality(value: str) -> str:
    return value.lower().strip().replace(" ", "_")


def resolve_team_id(team_id: str, teams: dict[str, Any]) -> str:
    if team_id in teams:
        return team_id
    for canonical_id, meta in teams.items():
        if not isinstance(meta, dict):
            continue
        legacy_ids = meta.get("legacy_ids") or []
        if team_id in {str(item) for item in legacy_ids if item}:
            return canonical_id
    return team_id
