import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.nlp.dataset import (
    build_nlp_dataset,
    persist_nlp_dataset,
    resolve_document_for_incident,
)
from fia_ml.training.nlp_config import NlpTrainingConfig


@pytest.fixture
def fixture_cfg():
    return NlpTrainingConfig.from_yaml(ROOT / "tests" / "fixtures" / "nlp" / "nlp_dataset_test.yaml")


def test_resolve_document_for_incident(fixture_cfg):
    doc = resolve_document_for_incident("nlp_test_2019_001", 2019, fixture_cfg)
    assert doc["document_id"] == "doc_2019_001"
    assert "Turn 4" in doc["parsed_fields"]["fact"]


def test_build_nlp_dataset_temporal_split(fixture_cfg):
    result = build_nlp_dataset(fixture_cfg)
    assert len(result.train) == 1
    assert len(result.validation) == 1
    assert int(result.train.iloc[0]["season"]) == 2019
    assert int(result.validation.iloc[0]["season"]) == 2025
    assert result.train.iloc[0]["penalty_severity"] == 1
    assert result.validation.iloc[0]["penalty_severity"] == 0


def test_build_nlp_dataset_text_and_audit(fixture_cfg):
    result = build_nlp_dataset(fixture_cfg)
    text = result.train.iloc[0]["text"]
    assert "[FACT]" in text
    assert "LEAKY_DECISION" not in text
    assert result.audit["totals"]["rows_with_text"] == 2
    assert result.audit["totals"]["join_rate"] == 1.0
    assert result.audit["seasons"]["2019"]["rows_with_text"] == 1


def test_missing_interim_doc_recorded_in_audit(fixture_cfg):
    cfg = NlpTrainingConfig.from_yaml(ROOT / "tests" / "fixtures" / "nlp" / "nlp_dataset_test.yaml")
    missing_doc = (
        ROOT
        / "tests"
        / "fixtures"
        / "nlp"
        / "extracted_documents"
        / "2019"
        / "doc_2019_001.json"
    )
    backup = missing_doc.read_text(encoding="utf-8")
    missing_doc.unlink()
    try:
        result = build_nlp_dataset(cfg)
        assert result.audit["totals"]["missing_interim_doc"] >= 1
        assert result.train.empty
        assert len(result.validation) == 1
    finally:
        missing_doc.write_text(backup, encoding="utf-8")


def test_persist_nlp_dataset_writes_jsonl(fixture_cfg):
    cfg = replace(
        fixture_cfg,
        paths={
            **fixture_cfg.paths,
            "processed": "tests/fixtures/nlp/_out/processed",
            "models": "tests/fixtures/nlp/_out/models",
        },
    )
    result = build_nlp_dataset(cfg)
    outputs = persist_nlp_dataset(result, cfg)
    assert outputs["audit_json"].exists()
    assert outputs["train_jsonl"].exists()
    assert outputs["validation_jsonl"].exists()
