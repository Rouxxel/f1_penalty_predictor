"""Classification metrics for model training."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Return standard classification metrics for multiclass targets."""
    label_order = labels if labels is not None else sorted({int(v) for v in y_true})
    report = classification_report(
        y_true,
        y_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in label_order
            if str(label) in report
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_order).tolist(),
        "labels": label_order,
    }
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=label_order))
    return metrics
