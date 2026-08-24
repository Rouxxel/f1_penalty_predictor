"""Load model training configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_TRAINING_CONFIG, PROJECT_ROOT, TARGET_MAPPING_CONFIG
from fia_ml.utils import secure_file_io as sio


@dataclass
class TrainingConfig:
    paths: dict[str, str]
    inputs: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    class_imbalance: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / self.paths[key]

    @property
    def target_mapping_path(self) -> Path:
        rel = self.paths.get("target_mapping", "configs/target_mapping.yaml")
        return PROJECT_ROOT / rel

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> TrainingConfig:
        path = config_path or DEFAULT_TRAINING_CONFIG
        raw = sio.read_yaml(path)
        return cls(
            paths=dict(raw["paths"]),
            inputs=dict(raw.get("inputs", {})),
            splits=dict(raw.get("splits", {})),
            model=dict(raw.get("model", {})),
            class_imbalance=dict(raw.get("class_imbalance", {})),
            evaluation=dict(raw.get("evaluation", {})),
            random_state=int(raw.get("random_state", 42)),
        )


def default_target_mapping_path() -> Path:
    return TARGET_MAPPING_CONFIG
