import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment_config import EnrichmentConfig


@pytest.fixture
def cfg():
    return PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml")


def test_enrichment_config_loads_defaults():
    ec = EnrichmentConfig.from_yaml(ROOT / "configs" / "enrichment.yaml")
    assert ec.prefer_ergast_round_n_minus_1 is True
    assert ec.overwrite_reference_standings is True
    assert ec.max_match_error_seconds == 45
    assert ec.qualifying_lap_fallback == "session_median"
    assert ec.openf1_enabled_from_season == 2023
    assert ec.openf1_base_url == "https://api.openf1.org/v1"
    assert ec.fastf1_enabled is True
    assert ec.driver_at_fault_min_confidence == 0.7
    assert ec.superlicense_rolling_by == "driver"
    assert ec.quality_targets["lap_race_min_fill_rate"] == 0.5


def test_enrichment_config_missing_file_returns_defaults():
    ec = EnrichmentConfig.from_yaml(ROOT / "configs" / "does_not_exist_enrichment.yaml")
    assert ec.prefer_ergast_round_n_minus_1 is True
    assert ec.overwrite_reference_standings is False


def test_pipeline_config_attaches_enrichment_settings(cfg):
    assert isinstance(cfg.enrichment_settings, EnrichmentConfig)
    assert cfg.enrichment_settings.overwrite_reference_standings is True


def test_pipeline_config_for_season_preserves_enrichment_settings(cfg):
    season_cfg = cfg.for_season(2019)
    assert season_cfg.enrichment_settings is cfg.enrichment_settings


def test_pipeline_config_custom_enrichment_path():
    custom = ROOT / "tests" / "fixtures" / "custom_enrichment.yaml"
    cfg = PipelineConfig.from_yaml(
        ROOT / "configs" / "data.yaml",
        enrichment_config_path=custom,
    )
    assert cfg.enrichment_settings.overwrite_reference_standings is False
    assert cfg.enrichment_settings.max_match_error_seconds == 30
