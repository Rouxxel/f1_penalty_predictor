import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.text_fields import (
    enrich_text_fields,
    infer_driver_at_fault,
    suggest_severity,
)
from fia_ml.data.validation import build_review_queue


def test_infer_driver_at_fault_patterns():
    fault, confidence = infer_driver_at_fault(
        "Car 44 was wholly to blame for the collision with Car 63.",
        "",
        "",
        ["lewis_hamilton", "car_63"],
    )
    assert fault == "car_44"
    assert confidence >= 0.85

    none_fault, none_conf = infer_driver_at_fault(
        "This was a racing incident.",
        "No driver was wholly or predominantly to blame.",
        "",
        ["driver_a", "driver_b"],
    )
    assert none_fault == "none"
    assert none_conf >= 0.7


def test_suggest_severity_from_penalty():
    severity, confidence = suggest_severity(
        "Dangerous driving at high speed.",
        "",
        "A five-second time penalty and two penalty points.",
        "5s_time_penalty",
    )
    assert severity == 5
    assert confidence >= 0.8


@patch("fia_ml.data.enrichment.text_fields.save_meta_rows")
@patch("fia_ml.data.enrichment.text_fields.load_interim_document")
@patch("fia_ml.data.enrichment.text_fields.load_meta")
def test_enrich_text_fields_writes_driver_at_fault(mock_meta, mock_doc, mock_save):
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    mock_meta.return_value = {
        "inc_1": {
            "incident_id": "inc_1",
            "document_id": "doc_1",
            "parse_confidence": 1.0,
        }
    }
    mock_doc.return_value = {
        "parsed_fields": {
            "fact": "Car 1 was wholly to blame for causing a collision.",
            "reason": "",
            "decision": "Ten second time penalty.",
        }
    }

    df = pd.DataFrame(
        [
            {
                "incident_id": "inc_1",
                "drivers": "max_verstappen,car_63",
                "driver_at_fault": "",
                "severity": "",
                "penalty": "10s_time_penalty",
            }
        ]
    )

    with patch("fia_ml.data.enrichment.text_fields.sio.read_json", return_value=[mock_meta.return_value["inc_1"]]):
        enriched = enrich_text_fields(df, cfg)

    assert enriched.iloc[0]["driver_at_fault"] == "car_1"
    mock_save.assert_called_once()


def test_build_review_queue_includes_text_suggestions():
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    df = pd.DataFrame(
        [
            {
                "incident_id": "inc_1",
                "session": "race",
                "drivers": "driver_a",
                "penalty": "warning",
                "severity": "",
                "driver_at_fault": "",
                "lap": "10",
                "round": "5",
            }
        ]
    )
    meta = {
        "inc_1": {
            "parse_confidence": 1.0,
            "severity_suggestion": 2,
            "driver_at_fault_suggestion": "driver_a",
            "driver_at_fault_confidence": 0.5,
        }
    }
    review = build_review_queue(df, cfg, meta)
    reasons = review.iloc[0]["review_reasons"]
    assert "severity_suggestion:2" in reasons
    assert "driver_at_fault_suggestion:driver_a" in reasons
