"""Train and evaluate V1 baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fia_ml.models.baseline import MajorityClassBaseline, SessionStratifiedBaseline
from fia_ml.paths import ensure_dir
from fia_ml.preprocessing.encoding import TARGET_COLUMN
from fia_ml.training.config import TrainingConfig
from fia_ml.training.metrics import compute_metrics
from fia_ml.utils import secure_file_io as sio


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c.startswith(("cat__", "num__"))]]


def _labels(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].astype(int)


def train_baselines(cfg: TrainingConfig) -> dict[str, Any]:
    """Fit majority and session-stratified baselines; evaluate on validation set."""
    processed_dir = cfg.path("processed")
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "validation.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Missing train.parquet or validation.parquet — run --stage prepare first"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    y_train = _labels(train_df)
    y_val = _labels(val_df)

    majority = MajorityClassBaseline().fit(y_train)
    majority_pred = majority.predict(len(val_df))
    majority_metrics = compute_metrics(y_val.to_numpy(), majority_pred)

    if "session" not in train_df.columns:
        raise ValueError("train.parquet missing 'session' column — re-run --stage prepare")

    session_model = SessionStratifiedBaseline().fit(y_train, train_df["session"])
    session_pred = session_model.predict(val_df["session"])
    session_metrics = compute_metrics(y_val.to_numpy(), session_pred)

    baseline_dir = ensure_dir(cfg.path("models") / "baseline")
    model_path = baseline_dir / "model.pkl"
    joblib.dump(
        {
            "majority_class": majority,
            "session_stratified": session_model,
        },
        model_path,
    )

    metrics = {
        "validation_rows": len(val_df),
        "train_rows": len(train_df),
        "majority_class": {
            "predicted_class": majority.majority_class,
            "metrics": majority_metrics,
        },
        "session_stratified": {
            "session_modes": session_model.session_modes,
            "global_fallback": session_model.global_class,
            "metrics": session_metrics,
        },
    }
    metrics_path = baseline_dir / "metrics.json"
    sio.write_json(metrics_path, metrics)

    predictions = pd.DataFrame(
        {
            "row_id": val_df["row_id"],
            "incident_id": val_df["incident_id"],
            "penalty": val_df["penalty"],
            "session": val_df["session"],
            "penalty_severity": y_val,
            "pred_majority": majority_pred,
            "pred_session_stratified": session_pred,
        }
    )
    predictions_path = baseline_dir / "predictions_val.json"
    sio.write_json(predictions_path, predictions.to_dict(orient="records"))

    return {
        "baseline_dir": str(baseline_dir),
        "majority_macro_f1": majority_metrics["macro_f1"],
        "session_stratified_macro_f1": session_metrics["macro_f1"],
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "predictions_val": str(predictions_path),
        },
    }
