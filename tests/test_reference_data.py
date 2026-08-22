import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.reference_enrich import enrich_with_reference
from fia_ml.data.reference_data import (
    build_event_name_to_circuit_map,
    load_circuits,
    load_drivers,
    load_reference,
    load_seasons,
    load_teams,
)
from fia_ml.utils import secure_file_io as sio
from fia_ml.utils.secure_file_io import ReadOnlyPathError


@pytest.fixture
def cfg():
    return PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml")


def test_load_circuits_has_event_names(cfg):
    circuits = load_circuits(cfg)
    event_map = build_event_name_to_circuit_map(circuits)
    assert event_map["Italian Grand Prix"] == "monza"
    assert event_map["Abu Dhabi Grand Prix"] == "yas_marina"


def test_reference_dir_is_read_only(cfg):
    circuits_path = cfg.path("reference") / "circuits.json"
    with pytest.raises(ReadOnlyPathError, match="read-only"):
        sio.write_json(circuits_path, {"test": True})


def test_load_reference_returns_expected_keys(cfg):
    refs = load_reference(cfg)
    assert {"circuits", "drivers", "teams", "seasons", "incident_types"} <= set(refs)


def test_reference_enrich_fills_2019_row(cfg):
    season_cfg = cfg.for_season(2019)
    df = pd.DataFrame(
        [
            {
                "incident_id": "2019_race_f2c196a891",
                "circuit": "yas_marina",
                "country": "uae",
                "first_season": "2009",
                "round": "",
                "season": "2019",
                "session": "race",
                "drivers": "robert_kubica,car_99",
                "rounds": "",
                "num_teams": "",
                "current_top_4_drivers": "",
                "nationalities": "",
                "driver_standings": "",
                "driver_points": "",
                "respective_teams": "",
                "construct_standings": "",
                "construct_points": "",
                "years_in_sport": "",
                "full_laps": "",
            }
        ]
    )
    enriched = enrich_with_reference(df, season_cfg)
    row = enriched.iloc[0]
    assert row["round"] == "21"
    assert row["rounds"] == "21"
    assert row["num_teams"] == "10"
    assert row["full_laps"] == "58"
    assert "hamilton" in row["current_top_4_drivers"]
    assert row["nationalities"] == "polish"
    assert row["respective_teams"] == "williams"


def test_seasons_and_teams_load(cfg):
    seasons = load_seasons(cfg)
    teams = load_teams(cfg)
    drivers = load_drivers(cfg)
    assert "2019" in seasons
    assert "mercedes" in teams
    assert "lewis_hamilton" in drivers
