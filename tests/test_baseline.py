"""Tests for baseline models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.models.baseline import MajorityClassBaseline, SessionStratifiedBaseline
from fia_ml.training.metrics import compute_metrics


def test_majority_class_predicts_mode() -> None:
    y_train = pd.Series([0, 1, 1, 1, 2])
    model = MajorityClassBaseline().fit(y_train)
    assert model.majority_class == 1
    assert np.all(model.predict(5) == 1)


def test_session_stratified_uses_per_session_mode() -> None:
    y = pd.Series([0, 0, 1, 1, 2])
    session = pd.Series(["race", "race", "qualifying", "qualifying", "race"])
    model = SessionStratifiedBaseline().fit(y, session)
    preds = model.predict(pd.Series(["race", "qualifying", "practice"]))
    assert preds[0] == 0
    assert preds[1] == 1
    assert preds[2] == model.global_class


def test_compute_metrics_macro_f1() -> None:
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 2])
    metrics = compute_metrics(y_true, y_pred)
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_train_baselines_integration() -> None:
    """End-to-end baseline training on synthetic parquets."""
    import shutil

    from fia_ml.training.config import TrainingConfig
    from fia_ml.training.train_baselines import train_baselines

    work_dir = ROOT / "tests" / "_tmp_baseline"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        {
            "cat__session": [0.0, 0.0, 1.0],
            "num__round": [1.0, 2.0, 3.0],
            "session": ["race", "race", "qualifying"],
            "penalty_severity": [1, 1, 0],
            "penalty": ["5s", "5s", "none"],
            "row_id": ["r1", "r2", "r3"],
            "incident_id": ["i1", "i2", "i3"],
            "split": ["train", "train", "train"],
        }
    )
    val_df = pd.DataFrame(
        {
            "cat__session": [0.0, 1.0],
            "num__round": [4.0, 5.0],
            "session": ["race", "qualifying"],
            "penalty_severity": [1, 0],
            "penalty": ["5s", "none"],
            "row_id": ["r4", "r5"],
            "incident_id": ["i4", "i5"],
            "split": ["validation", "validation"],
        }
    )

    processed = work_dir / "processed"
    processed.mkdir()
    train_df.to_parquet(processed / "train.parquet", index=False)
    val_df.to_parquet(processed / "validation.parquet", index=False)

    cfg = TrainingConfig.from_yaml()
    cfg.paths = dict(cfg.paths)
    cfg.paths["processed"] = str(processed)
    cfg.paths["models"] = str(work_dir / "models")

    result = train_baselines(cfg)
    assert result["majority_macro_f1"] >= 0.0
    assert Path(result["outputs"]["metrics"]).exists()

    shutil.rmtree(work_dir)
