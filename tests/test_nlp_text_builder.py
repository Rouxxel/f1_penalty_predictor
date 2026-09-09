import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.nlp.text_builder import build_text, leakage_violations, truncate_text


def _sample_doc() -> dict:
    return {
        "document_id": "2019_test_doc",
        "document_type": "decision",
        "event": "British Grand Prix",
        "parsed_fields": {
            "session": "race",
            "fact": "Car 44 left the track at Turn 12 and rejoined unsafely.",
            "offence": "Breach of Appendix L Chapter IV Article 2.",
            "decision": "LEAKY_DECISION_10_SECOND_TIME_PENALTY_APPLIED",
            "reason": "LEAKY_REASON_DRIVER_WAS_WHOLLY_TO_BLAME",
        },
    }


def test_fact_offence_includes_fact_and_offence():
    text = build_text(_sample_doc(), profile="fact_offence")
    assert "[FACT]" in text
    assert "Turn 12" in text
    assert "[OFFENCE]" in text
    assert "Appendix L" in text


def test_fact_offence_excludes_decision_and_reason():
    text = build_text(_sample_doc(), profile="fact_offence")
    assert "[DECISION]" not in text
    assert "[REASON]" not in text
    assert "LEAKY_DECISION_10_SECOND_TIME_PENALTY_APPLIED" not in text
    assert "LEAKY_REASON_DRIVER_WAS_WHOLLY_TO_BLAME" not in text
    assert leakage_violations(text, _sample_doc()["parsed_fields"], profile="fact_offence") == []


def test_fact_only_excludes_offence():
    text = build_text(_sample_doc(), profile="fact_only")
    assert "[FACT]" in text
    assert "[OFFENCE]" not in text
    assert "Appendix L" not in text


def test_leaky_full_includes_decision_and_reason():
    text = build_text(_sample_doc(), profile="leaky_full")
    assert "[DECISION]" in text
    assert "[REASON]" in text
    assert "LEAKY_DECISION_10_SECOND_TIME_PENALTY_APPLIED" in text


def test_metadata_sections_when_enabled():
    text = build_text(_sample_doc(), profile="fact_offence", include_metadata=True)
    assert "[SESSION] race" in text
    assert "[EVENT] British Grand Prix" in text
    assert "[DOCTYPE] decision" in text


def test_metadata_omitted_when_disabled():
    text = build_text(_sample_doc(), profile="fact_offence", include_metadata=False)
    assert "[SESSION]" not in text
    assert "[EVENT]" not in text
    assert "[FACT]" in text


def test_invalid_profile_raises():
    with pytest.raises(ValueError, match="Invalid profile"):
        build_text(_sample_doc(), profile="not_a_profile")


def test_truncation_char_fallback():
    long_text = "a" * 100
    truncated = truncate_text(long_text, max_length=10, tokenizer=None)
    assert len(truncated) < len(long_text)
    assert "..." in truncated


def test_build_text_raises_on_internal_leakage():
    doc = _sample_doc()
    doc["parsed_fields"]["fact"] = doc["parsed_fields"]["decision"]
    with pytest.raises(ValueError, match="leakage"):
        build_text(doc, profile="fact_offence")
