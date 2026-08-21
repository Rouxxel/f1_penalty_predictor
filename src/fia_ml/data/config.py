"""Load pipeline configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fia_ml.paths import DEFAULT_CONFIG, PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio


@dataclass(frozen=True)
class SeasonConfig:
    year: int
    url: str


@dataclass
class PipelineConfig:
    season_url: str
    season: int
    paths: dict[str, str]
    seasons: list[SeasonConfig] = field(default_factory=list)
    document_include_patterns: list[str] = field(default_factory=list)
    document_exclude_patterns: list[str] = field(default_factory=list)
    scraper: dict[str, Any] = field(default_factory=dict)
    enrichment: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / self.paths[key]

    def for_season(self, year: int) -> PipelineConfig:
        for season in self.seasons:
            if season.year == year:
                return replace(self, season=season.year, season_url=season.url)
        raise ValueError(f"Season {year} is not configured in data.yaml")

    def iter_seasons(self, years: list[int] | None = None) -> list[PipelineConfig]:
        if not self.seasons:
            return [self]
        targets = self.seasons
        if years is not None:
            targets = [season for season in self.seasons if season.year in years]
            missing = sorted(set(years) - {season.year for season in targets})
            if missing:
                raise ValueError(f"Season(s) not configured: {missing}")
        return [replace(self, season=season.year, season_url=season.url) for season in targets]

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> PipelineConfig:
        path = config_path or DEFAULT_CONFIG
        raw = sio.read_yaml(path)

        seasons = [
            SeasonConfig(year=int(item["year"]), url=str(item["url"]))
            for item in raw.get("seasons", [])
        ]
        default_season = int(raw.get("default_season", seasons[0].year if seasons else 0))

        season_url = raw.get("season_url")
        season = raw.get("season")
        if season_url is None or season is None:
            default = next((s for s in seasons if s.year == default_season), seasons[0] if seasons else None)
            if default is None:
                raise ValueError("Config must define seasons or season_url/season")
            season_url = default.url
            season = default.year
        else:
            season = int(season)

        return cls(
            season_url=str(season_url),
            season=season,
            seasons=seasons,
            paths=raw["paths"],
            document_include_patterns=list(raw.get("document_include_patterns", [])),
            document_exclude_patterns=list(raw.get("document_exclude_patterns", [])),
            scraper=dict(raw.get("scraper", {})),
            enrichment=dict(raw.get("enrichment", {})),
            validation=dict(raw.get("validation", {})),
        )
