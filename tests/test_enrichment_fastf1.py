import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.fastf1_enrich import (
    enrich_with_fastf1,
    flag_at_session_time,
    lap_number_at_session_time,
    positions_at_session_time,
)


def _sample_laps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DriverNumber": 44,
                "LapNumber": 10,
                "Position": 2,
                "Time": timedelta(seconds=2700),
            },
            {
                "DriverNumber": 44,
                "LapNumber": 11,
                "Position": 1,
                "Time": timedelta(seconds=2880),
            },
            {
                "DriverNumber": 63,
                "LapNumber": 10,
                "Position": 5,
                "Time": timedelta(seconds=2705),
            },
        ]
    )


def _sample_messages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Time": timedelta(seconds=2710),
                "Message": "YELLOW FLAG IN SECTOR 2",
                "Flag": "YELLOW",
                "Sector": 9,
            }
        ]
    )


def test_lap_number_at_session_time():
    assert lap_number_at_session_time(_sample_laps(), 2750.0) == 10


def test_flag_and_positions_helpers():
    messages = _sample_messages()
    assert flag_at_session_time(messages, 2712.0, tolerance_seconds=45) == "yellow"
    positions = positions_at_session_time(_sample_laps(), [44, None], 2712.0, tolerance_seconds=45)
    assert positions == ["2", ""]


@patch("fastf1.get_session")
@patch("fastf1.Cache.configure", create=True)
@patch("fastf1.Cache.enable_cache")
@patch("fia_ml.data.enrichment.fastf1_enrich.load_meta")
def test_enrich_with_fastf1_fills_race_state(mock_meta, _mock_cache, _mock_configure, mock_get_session):
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2019)
    mock_meta.return_value = {
        "inc_1": {
            "incident_id": "inc_1",
            "event": "Abu Dhabi Grand Prix",
            "car_number": "44",
            "session_offset_seconds": 2712.0,
        }
    }

    event_obj = MagicMock()
    event_obj.laps = _sample_laps()
    event_obj.race_control_messages = _sample_messages()
    event_obj.weather_data = pd.DataFrame()

    mock_get_session.return_value = event_obj

    df = pd.DataFrame(
        [
            {
                "incident_id": "inc_1",
                "session": "race",
                "round": "21",
                "drivers": "lewis_hamilton,car_99",
                "lap": "",
                "flag": "",
                "sector": "",
                "positions_of_involved parties": "",
                "full_laps": "",
                "safety_car": "",
            }
        ]
    )

    with patch(
        "fia_ml.data.enrichment.fastf1_enrich.build_driver_number_map_from_ergast",
        return_value={"lewis_hamilton": 44},
    ):
        enriched = enrich_with_fastf1(df, cfg, fill_gaps_only=True)

    row = enriched.iloc[0]
    assert row["lap"] == "10"
    assert row["full_laps"] == "11"
    assert row["flag"] == "yellow"
    assert row["sector"] == "2"
    assert row["positions_of_involved parties"] == "2,"


def test_loaded_attr_returns_none_when_fastf1_data_not_loaded():
    """Session.load() can partially fail; reading the missing part must not crash."""
    from fastf1.exceptions import DataNotLoadedError

    from fia_ml.data.enrichment.fastf1_enrich import _loaded_attr

    class PartialSession:
        weather_data = "loaded"

        @property
        def laps(self):
            raise DataNotLoadedError("laps not loaded")

    session = PartialSession()
    assert _loaded_attr(session, "laps") is None
    assert _loaded_attr(session, "race_control_messages") is None
    assert _loaded_attr(session, "weather_data") == "loaded"


def test_fastf1_offline_mode_is_opt_in():
    """offline_mode is only switched on when configs/enrichment.yaml asks for it."""
    from fia_ml.data.enrichment_config import EnrichmentConfig

    assert EnrichmentConfig(fastf1={}).fastf1_offline_mode is False
    assert EnrichmentConfig(fastf1={"offline_mode": True}).fastf1_offline_mode is True

    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2019)
    df = pd.DataFrame([{"incident_id": "x", "session": "race", "round": ""}])
    with patch("fastf1.Cache.configure", create=True), patch("fastf1.Cache.enable_cache"), patch(
        "fastf1.Cache.offline_mode"
    ) as offline, patch("fia_ml.data.enrichment.fastf1_enrich.load_meta", return_value={}):
        cfg.enrichment_settings.fastf1["offline_mode"] = True
        try:
            enrich_with_fastf1(df, cfg)
        finally:
            cfg.enrichment_settings.fastf1.pop("offline_mode", None)
    offline.assert_called_once_with(True)
