"""Tests for evaluation and reporting."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.training.config import TrainingConfig
from fia_ml.training.evaluate import (
    audit_leakage,
    build_error_analysis,
    check_domain_sanity,
    plot_confusion_matrix,
    plot_feature_importance,
    run_evaluation,
)
from fia_ml.utils import secure_file_io as sio


def test_audit_leakage_passes_clean_features() -> None:
    result = audit_leakage(["circuit", "round", "driver"])
    assert result["passed"] is True


def test_audit_leakage_fails_on_penalty() -> None:
    result = audit_leakage(["penalty", "circuit"])
    assert result["passed"] is False


def test_build_error_analysis() -> None:
    predictions = [
        {"penalty_severity": 0, "pred_xgboost": 0, "incident_id": "a"},
        {"penalty_severity": 1, "pred_xgboost": 2, "incident_id": "b", "penalty": "5s"},
    ]
    errors = build_error_analysis(predictions)
    assert len(errors) == 1
    assert errors[0]["incident_id"] == "b"


def test_domain_sanity_detects_incident_type() -> None:
    importance = {
        "cat__incident_type": 10.0,
        "cat__circuit": 1.0,
        "num__round": 0.5,
    }
    result = check_domain_sanity(importance, top_n=2)
    assert result["incident_type_in_top10"] is True
    assert result["passed"] is True


def test_run_evaluation_integration() -> None:
    work_dir = ROOT / "tests" / "_tmp_evaluate"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    models = work_dir / "models"
    xgb_dir = models / "xgboost"
    xgb_dir.mkdir(parents=True)

    sio.write_json(
        xgb_dir / "metrics.json",
        {
            "train_rows": 10,
            "validation_rows": 5,
            "best_iteration": 3,
            "macro_f1": 0.5,
            "beats_session_baseline": True,
            "validation_metrics": {
                "accuracy": 0.6,
                "macro_f1": 0.5,
                "weighted_f1": 0.55,
                "per_class": {
                    "0": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
                    "1": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 3},
                },
                "confusion_matrix": [[1, 1], [1, 2]],
                "labels": [0, 1],
                "log_loss": 0.7,
            },
        },
    )
    sio.write_json(
        xgb_dir / "predictions_val.json",
        [
            {"penalty_severity": 0, "pred_xgboost": 1, "incident_id": "i1", "penalty": "5s"},
            {"penalty_severity": 1, "pred_xgboost": 1, "incident_id": "i2", "penalty": "5s"},
        ],
    )
    sio.write_json(
        xgb_dir / "feature_importance.json",
        {"gain": {"cat__incident_type": 2.0, "num__round": 1.0}, "weight": {}},
    )
    sio.write_json(
        models / "preprocessor.meta.json",
        {"feature_columns": ["circuit", "round", "driver"]},
    )

    cfg = TrainingConfig.from_yaml()
    cfg.paths = dict(cfg.paths)
    cfg.paths["models"] = str(models)
    cfg.paths["reports"] = str(work_dir / "reports")

    result = run_evaluation(cfg)
    assert result["leakage_audit_passed"] is True
    assert Path(result["outputs"]["report"]).exists()
    assert Path(result["outputs"]["confusion_matrix_val"]).exists()
    assert Path(result["outputs"]["feature_importance"]).exists()
    assert result["misclassified_validation_rows"] == 1

    shutil.rmtree(work_dir)


def test_plot_helpers_write_files() -> None:
    work_dir = ROOT / "tests" / "_tmp_plot_helpers"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    confusion_path = work_dir / "cm.png"
    plot_confusion_matrix(
        [[2, 1], [0, 3]],
        [0, 1],
        {"0": "no", "1": "minor"},
        confusion_path,
        title="test",
    )
    assert confusion_path.exists()

    importance_path = work_dir / "fi.png"
    plot_feature_importance({"cat__a": 1.0, "num__b": 0.5}, importance_path, top_n=2)
    assert importance_path.exists()

    shutil.rmtree(work_dir)
