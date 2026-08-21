import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.incident_builder import classify_incident_type, make_incident_id, select_primary_documents


def test_make_incident_id_stable():
    a = make_incident_id(2019, "race", "88", "Incident between car 99 and car 88 in turn 12.")
    b = make_incident_id(2019, "race", "88", "Incident between car 99 and car 88 in turn 12.")
    assert a == b


def test_classify_collision():
    keywords = {"collision": ["collision", "incident with car", "incident between"]}
    result = classify_incident_type("Incident between car 99 and car 88", "", keywords)
    assert result == "collision"


def test_correction_doc_preferred():
    docs = [
        {
            "title": "Decision - Car 88",
            "document_type": "decision",
            "parsed_fields": {"car_number": "88", "session": "race", "fact": "Incident", "decision": "No further action"},
        },
        {
            "title": "Correction - Replaces Document 39 - Decision - Car 88",
            "document_type": "decision",
            "parsed_fields": {"car_number": "88", "session": "race", "fact": "Incident", "decision": "5 second time penalty"},
        },
    ]
    selected = select_primary_documents(docs)
    assert len(selected) == 1
    assert "Correction" in selected[0]["title"]
