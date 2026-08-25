"""Load normative engine runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_NORMATIVE_CONFIG, PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio


@dataclass
class NormativeConfig:
    rules_path: str = "configs/normative_rules.yaml"
    escalation: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    unmatched: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> NormativeConfig:
        path = config_path or DEFAULT_NORMATIVE_CONFIG
        raw = sio.read_yaml(path)
        return cls(
            rules_path=str(raw.get("rules_path", "configs/normative_rules.yaml")),
            escalation=dict(raw.get("escalation", {})),
            comparison=dict(raw.get("comparison", {})),
            unmatched=dict(raw.get("unmatched", {})),
            paths=dict(raw.get("paths", {})),
        )

    def resolve_path(self, key: str) -> Path:
        rel = self.paths.get(key, "")
        return PROJECT_ROOT / rel

    def rules_file_path(self) -> Path:
        return PROJECT_ROOT / self.rules_path
