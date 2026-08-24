"""Load feature engineering configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_FEATURES_CONFIG, PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio


@dataclass
class FeaturesConfig:
    precedent: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    championship: dict[str, Any] = field(default_factory=dict)
    race: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> FeaturesConfig:
        path = config_path or DEFAULT_FEATURES_CONFIG
        raw = sio.read_yaml(path)
        return cls(
            precedent=dict(raw.get("precedent", {})),
            history=dict(raw.get("history", {})),
            championship=dict(raw.get("championship", {})),
            race=dict(raw.get("race", {})),
            selection=dict(raw.get("selection", {})),
        )

    def resolve_path(self, rel: str) -> Path:
        return PROJECT_ROOT / rel
