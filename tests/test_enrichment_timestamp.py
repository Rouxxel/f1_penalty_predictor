import sys
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.timestamp import (
    compute_session_offset_seconds,
    enrich_meta_timestamp,
    enrich_timestamps,
    parse_incident_date,
    parse_time_text,
)


def test_parse_time_text_document_clock():
    parsed = parse_time_text(
        "18:21 2019 ABU DHABI GRAND PRIX 28 November - 1 December 2019 The Stewards..."
    )
    assert parsed is not None
    assert parsed.method == "document_clock"
    assert parsed.hours == 18
    assert parsed.minutes == 21


def test_parse_time_text_session_elapsed():
    parsed = parse_time_text("1:23:45")
    assert parsed is not None
    assert parsed.method == "session_elapsed"
    assert parsed.hours == 1
    assert parsed.minutes == 23
    assert parsed.seconds == 45


def test_parse_time_text_lap_hint():
    parsed = parse_time_text("Incident on lap 37 during race")
    assert parsed is not None
    assert parsed.method == "lap_hint"
    assert parsed.lap_hint == 37


def test_parse_incident_date_formats():
    assert parse_incident_date("01 December 2019") == date(2019, 12, 1)
    assert parse_incident_date("2019-12-01") == date(2019, 12, 1)


def test_compute_session_offset_seconds_elapsed():
    parsed = parse_time_text("0:45:10")
    offset, confidence = compute_session_offset_seconds(
        parsed,
        session_start=None,
        incident_date=None,
        max_match_error_seconds=45,
    )
    assert offset == 2710.0
    assert confidence > 0.8


def test_compute_session_offset_seconds_document_clock():
    parsed = parse_time_text("18:21 header")
    session_start = datetime(2019, 12, 1, 17, 10, 0)
    offset, confidence = compute_session_offset_seconds(
        parsed,
        session_start=session_start,
        incident_date=date(2019, 12, 1),
        max_match_error_seconds=45,
    )
    assert offset == 4260.0
    assert confidence > 0.0


@patch("fia_ml.data.enrichment.timestamp.load_weekend_schedule")
def test_enrich_meta_timestamp_writes_offset(mock_schedule):
    mock_schedule.return_value = {
        "date": "2019-12-01",
        "time": "17:10:00Z",
    }
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2019)
    row = pd.Series({"incident_id": "inc_1", "session": "race", "round": "21"})
    meta_row = {
        "incident_id": "inc_1",
        "event": "Abu Dhabi Grand Prix",
        "time": "18:21 2019 ABU DHABI GRAND PRIX",
        "date": "01 December 2019",
    }
    updated = enrich_meta_timestamp(meta_row, row, cfg, schedule_cache={})
    assert updated["time_parse_method"] == "document_clock"
    assert updated["session_offset_seconds"] == 4260.0


@patch("fia_ml.data.enrichment.timestamp.save_meta_rows")
def test_enrich_timestamps_returns_audit(mock_save):
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2019)
    df = pd.DataFrame([{"incident_id": "inc_1", "session": "race", "round": "21"}])
    with patch("fia_ml.data.enrichment.timestamp.load_meta") as mock_load:
        mock_load.return_value = {
            "inc_1": {
                "incident_id": "inc_1",
                "event": "Abu Dhabi Grand Prix",
                "time": "1:05:00",
                "date": "01 December 2019",
            }
        }
        with patch("fia_ml.data.enrichment.timestamp.sio.read_json", return_value=[mock_load.return_value["inc_1"]]):
            _, audit = enrich_timestamps(df, cfg)
    assert audit["rows"] == 1
    assert audit["parsed_rate"] == 1.0
    assert audit["offset_rate"] == 1.0
    mock_save.assert_called_once()
