import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.superlicense import compute_superlicense_before_values, enrich_superlicense_points


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "inc_a",
                "round": "3",
                "drivers": "driver_a",
                "superlicense_points_added": "2",
                "superlicense_points_before_incident": "",
            },
            {
                "incident_id": "inc_b",
                "round": "5",
                "drivers": "driver_a,driver_b",
                "superlicense_points_added": "1",
                "superlicense_points_before_incident": "",
            },
            {
                "incident_id": "inc_c",
                "round": "4",
                "drivers": "driver_b,car_99",
                "superlicense_points_added": "",
                "superlicense_points_before_incident": "",
            },
        ]
    )


def test_compute_superlicense_before_values_excludes_future_incidents():
    meta = {
        "inc_a": {"date": "01 March 2025"},
        "inc_b": {"date": "15 March 2025"},
        "inc_c": {"date": "10 March 2025"},
    }
    values = compute_superlicense_before_values(_sample_df(), meta)
    assert values["inc_a"] == "0"
    assert values["inc_b"] == "2,0"
    assert values["inc_c"] == "0,"


def test_compute_superlicense_before_values_no_future_leakage():
    df = pd.DataFrame(
        [
            {
                "incident_id": "early",
                "round": "1",
                "drivers": "driver_a",
                "superlicense_points_added": "",
            },
            {
                "incident_id": "late",
                "round": "2",
                "drivers": "driver_a",
                "superlicense_points_added": "3",
            },
        ]
    )
    meta = {"early": {"date": "01 March 2025"}, "late": {"date": "10 March 2025"}}
    values = compute_superlicense_before_values(df, meta)
    assert values["early"] == "0"
    assert values["late"] == "0"


def test_enrich_superlicense_points_writes_column():
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    meta = {
        "inc_a": {"date": "01 March 2025"},
        "inc_b": {"date": "15 March 2025"},
        "inc_c": {"date": "10 March 2025"},
    }
    with patch("fia_ml.data.enrichment.superlicense.load_meta", return_value=meta):
        enriched = enrich_superlicense_points(_sample_df(), cfg)

    assert enriched.loc[enriched["incident_id"] == "inc_b", "superlicense_points_before_incident"].iloc[0] == "2,0"
