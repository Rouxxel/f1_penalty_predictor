import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.quality_gates import (
    evaluate_quality_gates,
    incidents_with_prior_superlicense,
    merge_enrichment_into_quality,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "inc_a",
                "session": "race",
                "round": "3",
                "drivers": "driver_a",
                "lap": "10",
                "flag": "yellow",
                "sector": "2",
                "positions_of_involved parties": "3",
                "driver_standings": "5",
                "superlicense_points_before_incident": "0",
                "superlicense_points_added": "2",
            },
            {
                "incident_id": "inc_b",
                "session": "race",
                "round": "5",
                "drivers": "driver_a,driver_b",
                "lap": "",
                "flag": "",
                "sector": "",
                "positions_of_involved parties": "",
                "driver_standings": "4,8",
                "superlicense_points_before_incident": "2,0",
                "superlicense_points_added": "",
            },
        ]
    )


def test_incidents_with_prior_superlicense():
    meta = {"inc_a": {"date": "01 March 2025"}, "inc_b": {"date": "15 March 2025"}}
    prior = incidents_with_prior_superlicense(_sample_df(), meta)
    assert prior == {"inc_b"}


def test_evaluate_quality_gates_with_provenance():
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    enrichment_meta = [
        {
            "incident_id": "inc_a",
            "fields": {"driver_standings": {"source": "ergast", "round_used": 2}},
        },
        {
            "incident_id": "inc_b",
            "fields": {"driver_standings": {"source": "ergast", "round_used": 4}},
        },
    ]
    report = evaluate_quality_gates(_sample_df(), cfg, enrichment_meta=enrichment_meta)
    assert report["metrics"]["lap_race_fill_rate"] == 0.5
    assert report["metrics"]["ergast_standings_provenance_rate"] == 1.0
    assert report["metrics"]["rows_with_prior_superlicense"] == 1
    assert any(gate["id"] == "lap_race" for gate in report["gates"])


def test_merge_enrichment_into_quality():
    quality = {"season": 2025, "column_fill_rates": {"lap": 0.5}}
    enrichment = {"gates_passed": 2, "gates_total": 5}
    merged = merge_enrichment_into_quality(quality, enrichment)
    assert merged["enrichment"]["gates_total"] == 5
