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


def load_incident_type_keywords(cfg: PipelineConfig) -> dict[str, list[str]]:
    """Return incident_type_keywords.json (manually maintained)."""
    return sio.read_json(cfg.path("reference") / "incident_type_keywords.json")


def load_reference(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "circuits": load_circuits(cfg),
        "incident_types": load_incident_type_keywords(cfg),
    }
