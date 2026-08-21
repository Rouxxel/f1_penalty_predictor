import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.parsing import classify_document_type, parse_fia_document, parse_penalty_label


def test_parse_sample_decision():
    text = (ROOT / "tests" / "fixtures" / "sample_decision.txt").read_text(encoding="utf-8")
    parsed = parse_fia_document(text, "2019 Abu Dhabi Grand Prix - Decision - Car 88.pdf")
    fields = parsed["parsed_fields"]
    assert fields["car_number"] == "88"
    assert fields["driver_name"] == "Robert Kubica"
    assert fields["session"] == "race"
    assert "turn 12" in fields["fact"].lower()
    assert fields["penalty"] == "no_further_action"
    assert fields["driver_at_fault"] == "none"
    assert parsed["parse_confidence"] >= 0.5


def test_parse_penalty_label_grid():
    assert parse_penalty_label("A 3 place grid penalty will be applied.") == "3_place_grid_drop"


def test_classify_document_type_infringement():
    title = "Doc 52 - Infringement - Car 18 - More than one change of direction.pdf"
    assert classify_document_type(title) == "infringement"
