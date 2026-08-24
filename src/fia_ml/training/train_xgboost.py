"""Train and evaluate V1 XGBoost classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.models.xgboost_model import XGBoostTrainer
from fia_ml.paths import ensure_dir
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.config import TrainingConfig
from fia_ml.training.data_loaders import feature_matrix, labels, load_train_val_frames
from fia_ml.training.metrics import compute_metrics
from fia_ml.utils import secure_file_io as sio


def _label_encoder(mapping_path: Path) -> dict[str, str]:
    mapping = load_target_mapping(mapping_path)
    return {str(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _load_baseline_macro_f1(cfg: TrainingConfig) -> dict[str, float]:
    metrics_path = cfg.path("models") / "baseline" / "metrics.json"
    if not metrics_path.exists():
        return {}
    baseline = sio.read_json(metrics_path)
    return {
        "majority_macro_f1": baseline["majority_class"]["metrics"]["macro_f1"],
        "session_stratified_macro_f1": baseline["session_stratified"]["metrics"]["macro_f1"],
    }


def train_xgboost(cfg: TrainingConfig) -> dict[str, Any]:
    """Fit XGBoost with early stopping; save model and validation metrics."""
    train_df, val_df = load_train_val_frames(cfg)

    X_train = feature_matrix(train_df)
    X_val = feature_matrix(val_df)
    y_train = labels(train_df)
    y_val = labels(val_df)

    trainer = XGBoostTrainer(
        cfg_model=cfg.model,
        class_imbalance_strategy=str(cfg.class_imbalance.get("strategy", "inverse_frequency")),
    ).fit(X_train, y_train, X_val, y_val)

    y_pred = trainer.predict(X_val)
    y_proba = trainer.predict_proba(X_val)
    val_metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba=y_proba)

    xgb_dir = ensure_dir(cfg.model_dir())
    model_path = xgb_dir / "model.json"
    trainer.model.save_model(model_path)

    importance = trainer.feature_importance()
    importance_path = xgb_dir / "feature_importance.json"
    sio.write_json(importance_path, importance)

    label_encoder = _label_encoder(cfg.target_mapping_path)
    label_path = xgb_dir / "label_encoder.json"
    sio.write_json(label_path, label_encoder)

    baseline_scores = _load_baseline_macro_f1(cfg)
    metrics = {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "best_iteration": trainer.best_iteration,
        "validation_metrics": val_metrics,
        "macro_f1": val_metrics["macro_f1"],
        "baseline_comparison": baseline_scores,
        "beats_majority_baseline": val_metrics["macro_f1"] > baseline_scores.get(
            "majority_macro_f1", 0.0
        ),
        "beats_session_baseline": val_metrics["macro_f1"] > baseline_scores.get(
            "session_stratified_macro_f1", 0.0
        ),
    }
    metrics_path = xgb_dir / "metrics.json"
    sio.write_json(metrics_path, metrics)

    predictions = pd.DataFrame(
        {
            "row_id": val_df["row_id"],
            "incident_id": val_df["incident_id"],
            "penalty": val_df["penalty"],
            "session": val_df.get("session"),
            "penalty_severity": y_val,
            "pred_xgboost": y_pred,
        }
    )
    for class_id in sorted(label_encoder.keys(), key=int):
        predictions[f"proba_{label_encoder[class_id]}"] = y_proba[:, int(class_id)]

    predictions_path = xgb_dir / "predictions_val.json"
    sio.write_json(predictions_path, predictions.to_dict(orient="records"))

    return {
        "macro_f1": val_metrics["macro_f1"],
        "best_iteration": trainer.best_iteration,
        "beats_session_baseline": metrics["beats_session_baseline"],
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "feature_importance": str(importance_path),
            "label_encoder": str(label_path),
            "predictions_val": str(predictions_path),
        },
    }
