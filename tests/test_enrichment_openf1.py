import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.openf1 import (
    enrich_with_openf1,
    lap_at_time,
    normalize_flag,
    normalize_track_sector,
    positions_at_time,
    resolve_meeting_key,
    resolve_session_key,
)


def test_resolve_meeting_and_session_key():
    sessions = [
        {
            "meeting_key": 1254,
            "session_key": 9693,
            "session_type": "Race",
            "session_name": "Race",
            "country_name": "Australia",
            "location": "Melbourne",
            "circuit_short_name": "Melbourne",
            "date_start": "2025-03-16T04:00:00+00:00",
        },
        {
            "meeting_key": 1254,
            "session_key": 9691,
            "session_type": "Qualifying",
            "session_name": "Qualifying",
            "country_name": "Australia",
            "location": "Melbourne",
            "circuit_short_name": "Melbourne",
            "date_start": "2025-03-15T04:00:00+00:00",
        },
    ]
    assert resolve_meeting_key("Australian Grand Prix", sessions) == 1254
    race = resolve_session_key("Australian Grand Prix", "race", sessions)
    assert race is not None
    assert race["session_key"] == 9693


def test_lap_at_time_and_flag_normalization():
    incident_dt = datetime(2025, 3, 16, 4, 30, 0, tzinfo=timezone.utc)
    laps = [
        {"lap_number": 5, "date_start": "2025-03-16T04:28:29.996000+00:00"},
        {"lap_number": 6, "date_start": "2025-03-16T04:31:10.000000+00:00"},
    ]
    assert lap_at_time(laps, incident_dt) == 5
    assert normalize_flag("YELLOW", "YELLOW IN TRACK SECTOR 17") == "yellow"
    assert normalize_track_sector(17, "YELLOW IN TRACK SECTOR 17") == "3"


def test_positions_at_time():
    incident_dt = datetime(2025, 3, 16, 3, 7, 30, tzinfo=timezone.utc)
    positions = [
        {
            "driver_number": 1,
            "position": 3,
            "date": "2025-03-16T03:07:26.837000+00:00",
        }
    ]
    assert positions_at_time(positions, [1], incident_dt, tolerance_seconds=45) == ["3"]


@patch("fia_ml.data.enrichment.openf1.fetch_openf1")
@patch("fia_ml.data.enrichment.openf1.load_sessions_for_year")
@patch("fia_ml.data.enrichment.openf1.load_meta")
def test_enrich_with_openf1_fills_columns(mock_meta, mock_sessions, mock_fetch):
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    mock_sessions.return_value = [
        {
            "meeting_key": 1254,
            "session_key": 9693,
            "session_type": "Race",
            "session_name": "Race",
            "country_name": "Australia",
            "location": "Melbourne",
            "circuit_short_name": "Melbourne",
            "date_start": "2025-03-16T04:00:00+00:00",
        }
    ]
    mock_meta.return_value = {
        "inc_1": {
            "incident_id": "inc_1",
            "event": "Australian Grand Prix",
            "car_number": "1",
            "session_offset_seconds": 1800.0,
            "lap_hint": None,
        }
    }

    def fetch_side_effect(endpoint, params, _cfg):
        if endpoint == "laps":
            return [
                {"lap_number": 10, "date_start": "2025-03-16T04:25:00+00:00"},
                {"lap_number": 11, "date_start": "2025-03-16T04:31:00+00:00"},
                {"lap_number": 58, "date_start": "2025-03-16T06:00:00+00:00"},
            ]
        if endpoint == "position":
            return [
                {
                    "driver_number": 1,
                    "position": 2,
                    "date": "2025-03-16T04:29:50+00:00",
                }
            ]
        if endpoint == "race_control":
            return [
                {
                    "date": "2025-03-16T04:29:55+00:00",
                    "flag": "YELLOW",
                    "message": "YELLOW IN TRACK SECTOR 2",
                    "sector": 9,
                }
            ]
        return []

    mock_fetch.side_effect = fetch_side_effect

    df = pd.DataFrame(
        [
            {
                "incident_id": "inc_1",
                "session": "race",
                "round": "1",
                "drivers": "lewis_hamilton",
                "lap": "",
                "flag": "",
                "sector": "",
                "full_laps": "",
                "positions_of_involved parties": "",
            }
        ]
    )

    with patch("fia_ml.data.enrichment.openf1.build_driver_number_map", return_value={"lewis_hamilton": 1}):
        enriched = enrich_with_openf1(df, cfg, fill_gaps_only=True)

    row = enriched.iloc[0]
    assert row["lap"] == "10"
    assert row["full_laps"] == "58"
    assert row["flag"] == "yellow"
    assert row["sector"] == "2"
    assert row["positions_of_involved parties"] == "2"


def test_enrich_with_openf1_skips_pre_2023():
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2019)
    df = pd.DataFrame([{"incident_id": "inc_1", "session": "race", "lap": ""}])
    result = enrich_with_openf1(df, cfg)
    assert result.iloc[0]["lap"] == ""
