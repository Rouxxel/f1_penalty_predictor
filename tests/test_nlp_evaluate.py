import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.training.evaluate_nlp import (
    build_error_analysis,
    build_nlp_predictions,
    evaluate_nlp,
    _write_nlp_report,
)
from fia_ml.training.nlp_config import NlpTrainingConfig


@pytest.fixture
def fixture_cfg():
    return NlpTrainingConfig.from_yaml(ROOT / "tests" / "fixtures" / "nlp" / "nlp_dataset_test.yaml")


def test_build_nlp_predictions_shape(fixture_cfg):
    val_df = pd.DataFrame(
        {
            "row_id": ["r1", "r2"],
            "incident_id": ["i1", "i2"],
            "penalty": ["warning", "fine"],
            "session": ["race", "qualifying"],
            "penalty_severity": [1, 1],
        }
    )
    y_pred = np.array([1, 0])
    y_proba = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
    label_names = {0: "no_penalty", 1: "minor", 2: "major"}
    records = build_nlp_predictions(val_df, y_pred, y_proba, label_names=label_names)
    assert len(records) == 2
    assert records[0]["pred_nlp"] == 1
    assert records[0]["proba_minor"] == 0.7
    assert records[1]["proba_no_penalty"] == 0.6


def test_build_error_analysis_counts_misclassifications():
    predictions = [
        {"incident_id": "a", "row_id": "1", "penalty": "warning", "session": "race",
         "penalty_severity": 1, "pred_nlp": 1},
        {"incident_id": "b", "row_id": "2", "penalty": "fine", "session": "race",
         "penalty_severity": 2, "pred_nlp": 1},
    ]
    errors = build_error_analysis(predictions)
    assert len(errors) == 1
    assert errors[0]["incident_id"] == "b"


def test_write_nlp_report_smoke(fixture_cfg):
    report_path = ROOT / "tests" / "fixtures" / "nlp" / "_out" / "nlp_report_smoke.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    nlp_metrics = {
        "backbone": "distilbert-base-uncased",
        "text_profile": "fact_offence",
        "train_rows": 2,
        "validation_rows": 1,
        "macro_f1": 0.5,
        "validation_metrics": {
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "weighted_f1": 0.5,
            "per_class": {
                "0": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
            },
        },
    }
    _write_nlp_report(
        fixture_cfg,
        nlp_metrics=nlp_metrics,
        v1_metrics={"macro_f1": 0.402},
        audit={"totals": {"rows_with_text": 3}},
        error_count=1,
        output_path=report_path,
        figures={"confusion_matrix_val": "reports/figures/nlp_confusion_matrix_val.png"},
    )
    text = report_path.read_text(encoding="utf-8")
    assert "NLP Training Report" in text
    assert "0.402" in text
    assert "fact_offence" in text


@pytest.mark.slow
def test_evaluate_nlp_end_to_end(fixture_cfg):
    smoke_dir = ROOT / "tests" / "fixtures" / "nlp" / "_out" / "train_smoke"
    model_dir = smoke_dir / "model"
    if not model_dir.exists():
        pytest.skip("train smoke artifacts missing — run test_train_nlp_smoke first")

    cfg = NlpTrainingConfig(
        paths={**fixture_cfg.paths, "models": "tests/fixtures/nlp/_out/train_smoke"},
        inputs=fixture_cfg.inputs,
        splits=fixture_cfg.splits,
        text={**fixture_cfg.text, "max_length": 128},
        model=fixture_cfg.model,
        training=fixture_cfg.training,
        evaluation=fixture_cfg.evaluation,
        fusion=fixture_cfg.fusion,
    )
    result = evaluate_nlp(cfg)
    assert Path(result["outputs"]["predictions_val"]).exists()
    assert Path(result["outputs"]["report"]).exists()
    assert result["macro_f1"] >= 0.0
