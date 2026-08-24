"""Tests for XGBoost training."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.models.xgboost_model import XGBoostTrainer
from fia_ml.training.config import TrainingConfig
from fia_ml.training.train_xgboost import train_xgboost


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.DataFrame(
        {
            "cat__session": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "num__round": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "session": ["race"] * 6,
            "penalty_severity": [0, 1, 1, 1, 2, 0],
            "penalty": ["none", "5s", "5s", "reprimand", "grid", "none"],
            "row_id": [f"r{i}" for i in range(6)],
            "incident_id": [f"i{i}" for i in range(6)],
            "split": ["train"] * 6,
        }
    )
    val_df = pd.DataFrame(
        {
            "cat__session": [0.0, 1.0, 0.0],
            "num__round": [7.0, 8.0, 9.0],
            "session": ["race", "qualifying", "race"],
            "penalty_severity": [1, 0, 2],
            "penalty": ["5s", "none", "grid"],
            "row_id": ["r6", "r7", "r8"],
            "incident_id": ["i6", "i7", "i8"],
            "split": ["validation"] * 3,
        }
    )
    return train_df, val_df


def test_xgboost_trainer_fit_predict() -> None:
    train_df, val_df = _synthetic_frames()
    X_train = train_df[["cat__session", "num__round"]]
    X_val = val_df[["cat__session", "num__round"]]

    trainer = XGBoostTrainer(
        cfg_model={
            "max_depth": 2,
            "learning_rate": 0.3,
            "n_estimators": 20,
            "early_stopping_rounds": 5,
            "num_class": 3,
            "random_state": 42,
        }
    ).fit(X_train, train_df["penalty_severity"], X_val, val_df["penalty_severity"])

    preds = trainer.predict(X_val)
    assert len(preds) == len(val_df)
    assert trainer.best_iteration is not None
    assert "gain" in trainer.feature_importance()


def test_train_xgboost_integration() -> None:
    work_dir = ROOT / "tests" / "_tmp_xgboost"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    processed = work_dir / "processed"
    processed.mkdir(parents=True)
    train_df, val_df = _synthetic_frames()
    train_df.to_parquet(processed / "train.parquet", index=False)
    val_df.to_parquet(processed / "validation.parquet", index=False)

    cfg = TrainingConfig.from_yaml()
    cfg.paths = dict(cfg.paths)
    cfg.paths["processed"] = str(processed)
    cfg.paths["models"] = str(work_dir / "models")
    cfg.model = {
        "max_depth": 2,
        "learning_rate": 0.3,
        "n_estimators": 30,
        "early_stopping_rounds": 5,
        "num_class": 3,
        "random_state": 42,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    result = train_xgboost(cfg)
    assert result["macro_f1"] >= 0.0
    assert Path(result["outputs"]["model"]).exists()
    assert Path(result["outputs"]["feature_importance"]).exists()

    shutil.rmtree(work_dir)
