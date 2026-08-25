"""Feature selection and pruning for V2 (correlation + importance)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from fia_ml.features.config import FeaturesConfig
from fia_ml.models.xgboost_model import XGBoostTrainer
from fia_ml.preprocessing.encoding import BOOLEAN_FEATURES, NUMERIC_FEATURES
from fia_ml.preprocessing.leakage_filter import (
    V1_CATEGORICAL_FEATURES,
    V2_CATEGORICAL_FEATURES,
)
from fia_ml.training.data_loaders import feature_matrix, labels
from fia_ml.training.metrics import compute_metrics

CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES | V2_CATEGORICAL_FEATURES


def _missing_rate(series: pd.Series) -> float:
    if series.dtype == object or str(series.dtype) == "string":
        return float((series.isna() | (series.astype(str).str.strip() == "")).mean())
    return float(series.isna().mean())


def _drop_high_missing(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    max_missing_rate: float,
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for col in feature_columns:
        if col not in train_df.columns:
            continue
        if _missing_rate(train_df[col]) > max_missing_rate:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def _numeric_columns(feature_columns: list[str]) -> list[str]:
    return [col for col in feature_columns if col in NUMERIC_FEATURES or col in BOOLEAN_FEATURES]


def _mutual_info_scores(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> dict[str, float]:
    y = train_df[target_col].astype(int).to_numpy()
    matrix_cols: list[pd.Series] = []
    discrete_mask: list[bool] = []

    for col in feature_columns:
        if col in NUMERIC_FEATURES or col in BOOLEAN_FEATURES:
            values = pd.to_numeric(train_df[col], errors="coerce")
            fill_value = values.median() if values.notna().any() else 0.0
            matrix_cols.append(values.fillna(fill_value))
            discrete_mask.append(False)
        else:
            codes, _ = pd.factorize(train_df[col].astype("string"), sort=True)
            matrix_cols.append(pd.Series(codes, index=train_df.index))
            discrete_mask.append(True)

    if not matrix_cols:
        return {}

    X = pd.concat(matrix_cols, axis=1)
    X.columns = feature_columns
    scores = mutual_info_classif(
        X.to_numpy(),
        y,
        discrete_features=discrete_mask,
        random_state=42,
    )
    return {col: float(score) for col, score in zip(feature_columns, scores)}


def _correlation_prune(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    threshold: float,
) -> tuple[list[str], list[dict[str, str]]]:
    numeric_cols = _numeric_columns(feature_columns)
    if len(numeric_cols) < 2:
        return feature_columns, []

    numeric_df = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr().abs()
    mi_scores = _mutual_info_scores(train_df, feature_columns, target_col)

    kept = set(feature_columns)
    dropped_pairs: list[dict[str, str]] = []

    for i, col_a in enumerate(numeric_cols):
        if col_a not in kept:
            continue
        for col_b in numeric_cols[i + 1 :]:
            if col_b not in kept:
                continue
            if corr.loc[col_a, col_b] <= threshold:
                continue
            score_a = mi_scores.get(col_a, 0.0)
            score_b = mi_scores.get(col_b, 0.0)
            if score_a >= score_b:
                drop_col, keep_col = col_b, col_a
            else:
                drop_col, keep_col = col_a, col_b
            if drop_col in kept:
                kept.remove(drop_col)
                dropped_pairs.append({"dropped": drop_col, "kept": keep_col})

    return [col for col in feature_columns if col in kept], dropped_pairs


def prune_raw_features(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    cfg: FeaturesConfig,
    *,
    target_col: str = "penalty_severity",
) -> tuple[list[str], dict[str, Any]]:
    """Apply missing-rate and correlation pruning on raw feature columns."""
    selection_cfg = cfg.selection
    max_missing = float(selection_cfg.get("max_missing_rate", 0.40))
    corr_threshold = float(selection_cfg.get("correlation_threshold", 0.95))

    after_missing, dropped_missing = _drop_high_missing(
        train_df, feature_columns, max_missing
    )
    after_corr, dropped_corr = _correlation_prune(
        train_df, after_missing, target_col, corr_threshold
    )

    report = {
        "input_columns": list(feature_columns),
        "kept_columns": after_corr,
        "dropped_missing": dropped_missing,
        "dropped_correlation": dropped_corr,
    }
    return after_corr, report


def prune_encoded_by_importance(
    train_enc: pd.DataFrame,
    val_enc: pd.DataFrame,
    model_cfg: dict[str, Any],
    *,
    class_imbalance_strategy: str = "inverse_frequency",
    drop_percentile: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Train a preliminary model and drop bottom-percentile encoded features by gain."""
    X_train = feature_matrix(train_enc)
    X_val = feature_matrix(val_enc)
    if X_train.empty:
        return train_enc, val_enc, [], {"dropped_importance": []}

    y_train = labels(train_enc)
    y_val = labels(val_enc)

    trainer = XGBoostTrainer(
        cfg_model=dict(model_cfg),
        class_imbalance_strategy=class_imbalance_strategy,
    ).fit(X_train, y_train, X_val, y_val)

    gain = trainer.feature_importance().get("gain", {})
    if not gain:
        return train_enc, val_enc, [], {"dropped_importance": []}

    ranked = sorted(gain.items(), key=lambda item: item[1])
    n_drop = int(len(ranked) * drop_percentile / 100.0)
    if len(ranked) > 1:
        n_drop = max(1, n_drop)
    else:
        n_drop = 0

    dropped = [name for name, _ in ranked[:n_drop]]
    dropped_set = set(dropped)
    kept_encoded = [col for col in X_train.columns if col not in dropped_set]

    def _filter_encoded(df: pd.DataFrame) -> pd.DataFrame:
        meta_cols = [c for c in df.columns if not c.startswith(("cat__", "num__"))]
        return df[kept_encoded + meta_cols]

    report = {
        "dropped_importance": dropped,
        "kept_encoded_columns": kept_encoded,
        "preliminary_macro_f1": float(
            compute_metrics(
                y_val.to_numpy(),
                trainer.predict(X_val),
                y_proba=trainer.predict_proba(X_val),
            )["macro_f1"]
        ),
    }
    return _filter_encoded(train_enc), _filter_encoded(val_enc), dropped, report


def select_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: FeaturesConfig,
    *,
    target_col: str = "penalty_severity",
) -> list[str]:
    """Return pruned raw feature column list (missing + correlation steps only)."""
    kept, _ = prune_raw_features(df, feature_columns, cfg, target_col=target_col)
    return kept
