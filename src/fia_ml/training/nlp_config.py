"""Load NLP (spec V2) training configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_NLP_CONFIG, PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio

ALLOWED_TEXT_PROFILES = frozenset({"fact_only", "fact_offence", "leaky_full"})


@dataclass(frozen=True)
class NlpTrainingConfig:
    paths: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    fusion: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        if key not in self.paths:
            raise KeyError(f"Unknown path key: {key}")
        return PROJECT_ROOT / self.paths[key]

    def meta_path_for_season(self, season: int) -> Path:
        template = self.paths.get("meta_glob", "dataset/csv/raw_incidents_{season}.meta.json")
        return PROJECT_ROOT / template.format(season=season)

    @property
    def target_mapping_path(self) -> Path:
        rel = self.paths.get("target_mapping", "configs/target_mapping.yaml")
        return PROJECT_ROOT / rel

    @property
    def text_profile(self) -> str:
        return str(self.text.get("profile", "fact_offence"))

    @property
    def max_length(self) -> int:
        return int(self.text.get("max_length", 512))

    @property
    def backbone(self) -> str:
        return str(self.model.get("backbone", "distilbert-base-uncased"))

    @property
    def num_labels(self) -> int:
        return int(self.model.get("num_labels", 3))

    @property
    def training_seed(self) -> int:
        return int(self.training.get("seed", 42))

    def model_dir(self) -> Path:
        return self.path("models")

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> NlpTrainingConfig:
        path = config_path or DEFAULT_NLP_CONFIG
        if not path.exists():
            raise FileNotFoundError(f"NLP config not found: {path}")
        raw = sio.read_yaml(path)
        cfg = cls(
            paths=dict(raw.get("paths", {})),
            inputs=dict(raw.get("inputs", {})),
            splits=dict(raw.get("splits", {})),
            text=dict(raw.get("text", {})),
            model=dict(raw.get("model", {})),
            training=dict(raw.get("training", {})),
            evaluation=dict(raw.get("evaluation", {})),
            fusion=dict(raw.get("fusion", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        profile = self.text_profile
        if profile not in ALLOWED_TEXT_PROFILES:
            raise ValueError(
                f"Invalid text.profile '{profile}'. "
                f"Allowed: {sorted(ALLOWED_TEXT_PROFILES)}"
            )
        if self.max_length < 32:
            raise ValueError("text.max_length must be >= 32")
        if self.num_labels < 2:
            raise ValueError("model.num_labels must be >= 2")
