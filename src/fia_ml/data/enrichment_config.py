"""Load enrichment pipeline configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_ENRICHMENT_CONFIG, PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio


@dataclass(frozen=True)
class EnrichmentConfig:
    standings: dict[str, Any] = field(default_factory=dict)
    timestamp: dict[str, Any] = field(default_factory=dict)
    openf1: dict[str, Any] = field(default_factory=dict)
    fastf1: dict[str, Any] = field(default_factory=dict)
    text_fields: dict[str, Any] = field(default_factory=dict)
    superlicense: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> EnrichmentConfig:
        path = config_path or DEFAULT_ENRICHMENT_CONFIG
        if not path.exists():
            return cls()
        raw = sio.read_yaml(path)
        return cls(
            standings=dict(raw.get("standings", {})),
            timestamp=dict(raw.get("timestamp", {})),
            openf1=dict(raw.get("openf1", {})),
            fastf1=dict(raw.get("fastf1", {})),
            text_fields=dict(raw.get("text_fields", {})),
            superlicense=dict(raw.get("superlicense", {})),
        )

    def resolve_path(self, rel: str) -> Path:
        return PROJECT_ROOT / rel

    @property
    def prefer_ergast_round_n_minus_1(self) -> bool:
        return bool(self.standings.get("prefer_ergast_round_n_minus_1", True))

    @property
    def overwrite_reference_standings(self) -> bool:
        return bool(self.standings.get("overwrite_reference_standings", False))

    @property
    def max_match_error_seconds(self) -> float:
        return float(self.timestamp.get("max_match_error_seconds", 45))

    @property
    def qualifying_lap_fallback(self) -> str:
        return str(self.timestamp.get("qualifying_lap_fallback", "none"))

    @property
    def openf1_enabled_from_season(self) -> int:
        return int(self.openf1.get("enabled_from_season", 2023))

    @property
    def openf1_base_url(self) -> str:
        return str(self.openf1.get("base_url", "https://api.openf1.org/v1"))

    @property
    def openf1_cache_dir(self) -> Path:
        rel = str(self.openf1.get("cache_dir", "data/raw/race_data/openf1_cache"))
        return self.resolve_path(rel)

    @property
    def fastf1_enabled(self) -> bool:
        return bool(self.fastf1.get("enabled", True))

    @property
    def fastf1_validate_openf1_from_season(self) -> int:
        return int(self.fastf1.get("validate_openf1_from_season", 2023))

    @property
    def driver_at_fault_min_confidence(self) -> float:
        return float(self.text_fields.get("driver_at_fault_min_confidence", 0.7))

    @property
    def write_severity_suggestions_to_review_queue(self) -> bool:
        return bool(self.text_fields.get("write_severity_suggestions_to_review_queue", True))

    @property
    def superlicense_rolling_by(self) -> str:
        return str(self.superlicense.get("rolling_by", "driver"))
