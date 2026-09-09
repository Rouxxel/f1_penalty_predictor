import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.ergast import (
    build_per_driver_standing_fields,
    enrich_with_ergast,
    resolve_driver_standing,
)
from fia_ml.data.enrichment.reference_enrich import enrich_with_reference


@pytest.fixture
def cfg():
    return PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)


def _hamilton_standing(position: str, points: str, constructor_id: str = "mercedes") -> dict:
    return {
        "position": position,
        "points": points,
        "Driver": {
            "driverId": "hamilton",
            "familyName": "Hamilton",
            "code": "HAM",
            "nationality": "British",
        },
        "Constructors": [{"constructorId": constructor_id}],
    }


def _verstappen_standing(position: str, points: str) -> dict:
    return {
        "position": position,
        "points": points,
        "Driver": {
            "driverId": "max_verstappen",
            "familyName": "Verstappen",
            "code": "VER",
            "nationality": "Dutch",
        },
        "Constructors": [{"constructorId": "red_bull"}],
    }


def test_resolve_driver_standing_matches_slug_and_ergast_id():
    lookup = {
        "hamilton": _hamilton_standing("2", "150"),
        "max_verstappen": _verstappen_standing("1", "200"),
    }
    assert resolve_driver_standing("lewis_hamilton", lookup)["position"] == "2"
    assert resolve_driver_standing("max_verstappen", lookup)["position"] == "1"


def test_build_per_driver_standing_fields_aligns_multi_driver_rows():
    driver_standings = [
        _hamilton_standing("2", "150"),
        _verstappen_standing("1", "200"),
    ]
    constructor_standings = [
        {"position": "1", "points": "300", "Constructor": {"constructorId": "mercedes"}},
        {"position": "2", "points": "250", "Constructor": {"constructorId": "red_bull"}},
    ]
    teams = {"mercedes": {}, "red_bull": {}}

    fields = build_per_driver_standing_fields(
        ["lewis_hamilton", "car_99"],
        driver_standings,
        constructor_standings,
        teams,
    )

    assert fields["driver_standings"] == "2,"
    assert fields["driver_points"] == "150,"
    assert fields["nationalities"] == "british,"
    assert fields["respective_teams"] == "mercedes,"
    assert fields["construct_standings"] == "1,"
    assert fields["construct_points"] == "300,"


def test_reference_enrich_defers_standings_when_configured(cfg):
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
    assert row["nationalities"] == "polish,"
    assert row["driver_standings"] == ""
    assert row["respective_teams"] == ""
    assert row["current_top_4_drivers"] == ""


@patch("fia_ml.data.enrichment.ergast.load_meta")
@patch("fia_ml.data.enrichment.ergast.load_race_results")
@patch("fia_ml.data.enrichment.ergast.load_constructor_standings")
@patch("fia_ml.data.enrichment.ergast.load_driver_standings")
@patch("fia_ml.data.enrichment.ergast.load_season_calendar")
def test_enrich_with_ergast_uses_round_n_minus_1(
    mock_calendar,
    mock_driver_standings,
    mock_constructor_standings,
    mock_results,
    mock_meta,
    cfg,
):
    mock_calendar.return_value = [
        {"round": 8, "race_name": "British Grand Prix", "circuit_id": "silverstone", "country": "uk"}
    ]
    mock_meta.return_value = {"inc_1": {"event": "British Grand Prix", "car_number": ""}}
    mock_results.return_value = []
    mock_constructor_standings.return_value = [
        {"position": "1", "points": "300", "Constructor": {"constructorId": "mercedes"}}
    ]

    def driver_standings_side_effect(_cfg, round_num: int):
        if round_num == 7:
            return [_hamilton_standing("3", "120")]
        if round_num == 8:
            return [_hamilton_standing("1", "200")]
        return []

    mock_driver_standings.side_effect = driver_standings_side_effect

    df = pd.DataFrame(
        [
            {
                "incident_id": "inc_1",
                "round": "8",
                "season": "2025",
                "session": "race",
                "drivers": "lewis_hamilton",
                "driver_standings": "1",
                "driver_points": "200",
                "nationalities": "",
                "respective_teams": "",
                "construct_standings": "",
                "construct_points": "",
                "current_top_4_drivers": "season_end_leader",
                "rounds": "24",
                "num_teams": "10",
            }
        ]
    )

    enriched = enrich_with_ergast(df, cfg, fill_gaps_only=True)
    row = enriched.iloc[0]
    assert row["driver_standings"] == "3"
    assert row["driver_points"] == "120"
    assert row["current_top_4_drivers"] == "hamilton"
    mock_driver_standings.assert_any_call(cfg, 7)
    assert not any(call.args[1] == 8 for call in mock_driver_standings.call_args_list)
