"""Load manually curated reference data (read-only)."""

from __future__ import annotations

from typing import Any

from fia_ml.data.config import PipelineConfig
from fia_ml.paths import REFERENCE_DIR
from fia_ml.utils import secure_file_io as sio

# Pipeline code must never write under data/reference/.
sio.register_readonly_dirs(REFERENCE_DIR)


def load_circuits(cfg: PipelineConfig) -> dict[str, Any]:
    """Return circuits.json. This file is manually maintained; never auto-generated."""
    return sio.read_json(cfg.path("reference") / "circuits.json")


def load_drivers(cfg: PipelineConfig) -> dict[str, Any]:
    """Return drivers.json (driver profiles keyed by slug id)."""
    return sio.read_json(cfg.path("reference") / "drivers.json")


def load_teams(cfg: PipelineConfig) -> dict[str, Any]:
    """Return teams.json (constructor profiles keyed by slug id)."""
    return sio.read_json(cfg.path("reference") / "teams.json")


def load_seasons(cfg: PipelineConfig) -> dict[str, Any]:
    """Return seasons.json (per-season calendar and championship tables)."""
    return sio.read_json(cfg.path("reference") / "seasons.json")


def load_incident_type_keywords(cfg: PipelineConfig) -> dict[str, list[str]]:
    """Return incident_type_keywords.json (manually maintained)."""
    return sio.read_json(cfg.path("reference") / "incident_type_keywords.json")


def load_reference(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "circuits": load_circuits(cfg),
        "drivers": load_drivers(cfg),
        "teams": load_teams(cfg),
        "seasons": load_seasons(cfg),
        "incident_types": load_incident_type_keywords(cfg),
    }


def build_event_name_to_circuit_map(circuits: dict[str, Any]) -> dict[str, str]:
    """Map FIA event titles (e.g. 'Italian Grand Prix') to circuit slugs."""
    mapping: dict[str, str] = {}
    for slug, meta in circuits.items():
        if not isinstance(meta, dict):
            continue
        event_name = meta.get("event_name")
        if event_name:
            mapping[str(event_name)] = slug
    legacy = circuits.get("event_to_circuit")
    if isinstance(legacy, dict):
        mapping.update({str(k): str(v) for k, v in legacy.items()})
    return mapping
