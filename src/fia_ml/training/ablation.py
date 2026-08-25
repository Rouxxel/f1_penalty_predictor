"""Ablation experiments comparing V1 vs cumulative V2 feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.pipeline import add_v2_features
from fia_ml.features.selection import prune_encoded_by_importance, prune_raw_features
from fia_ml.models.xgboost_model import XGBoostTrainer
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.encoding import fit_encode_splits
from fia_ml.preprocessing.leakage_filter import select_feature_columns_for_groups
from fia_ml.preprocessing.splitting import temporal_split, verify_no_season_overlap
from fia_ml.training.config import TrainingConfig
from fia_ml.training.data_loaders import feature_matrix, labels
from fia_ml.training.metrics import compute_metrics
from fia_ml.utils import secure_file_io as sio

ABLATION_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "A": {
        "label": "V1 baseline",
        "groups": frozenset(),
        "apply_selection": False,
    },
    "B": {
        "label": "+ race/championship",
        "groups": frozenset({"race", "championship"}),
        "apply_selection": False,
    },
    "C": {
        "label": "+ history",
        "groups": frozenset({"race", "championship", "history"}),
        "apply_selection": False,
    },
    "D": {
        "label": "+ precedent",
        "groups": frozenset({"race", "championship", "history", "precedent"}),
        "apply_selection": False,
    },
    "E": {
        "label": "full V2 + selection prune",
        "groups": frozenset({"race", "championship", "history", "precedent"}),
        "apply_selection": True,
    },
}


def _features_config_path(cfg: TrainingConfig) -> Any:
    rel = cfg.paths.get("features_config", "configs/features.yaml")
    return PROJECT_ROOT / rel


def _incidents_path(cfg: TrainingConfig) -> Any:
    processed = cfg.path("processed")
    features_cfg = cfg.features or {}
    return processed / str(features_cfg.get("incidents_file", "incidents.parquet"))


def _train_macro_f1(
    train_enc: pd.DataFrame,
    val_enc: pd.DataFrame,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    X_train = feature_matrix(train_enc)
    X_val = feature_matrix(val_enc)
    y_train = labels(train_enc)
    y_val = labels(val_enc)

    trainer = XGBoostTrainer(
        cfg_model=dict(cfg.model),
        class_imbalance_strategy=str(cfg.class_imbalance.get("strategy", "inverse_frequency")),
    ).fit(X_train, y_train, X_val, y_val)

    y_pred = trainer.predict(X_val)
    y_proba = trainer.predict_proba(X_val)
    metrics = compute_metrics(y_val.to_numpy(), y_pred, y_proba=y_proba)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "best_iteration": trainer.best_iteration,
        "n_features": len(X_train.columns),
        "metrics": metrics,
    }


def run_ablation(cfg: TrainingConfig) -> dict[str, Any]:
    """Run experiments A–E from FEATURE_ENGINEERING_PLAN.md."""
    verify_no_season_overlap(cfg.splits)
    incidents_path = _incidents_path(cfg)
    if not incidents_path.exists():
        raise FileNotFoundError(
            f"Missing {incidents_path} — run --stage prepare first"
        )

    incidents = pd.read_parquet(incidents_path)
    features_cfg = FeaturesConfig.from_yaml(_features_config_path(cfg))
    enriched = add_v2_features(incidents, features_cfg)
    train_df, val_df, test_df = temporal_split(enriched, cfg.splits)

    experiments: dict[str, Any] = {}
    previous_macro_f1: float | None = None

    for exp_id, spec in ABLATION_EXPERIMENTS.items():
        feature_columns = select_feature_columns_for_groups(enriched, spec["groups"])
        selection_report: dict[str, Any] = {}

        if spec["apply_selection"]:
            feature_columns, selection_report = prune_raw_features(
                train_df, feature_columns, features_cfg
            )

        train_enc, val_enc, _, _ = fit_encode_splits(
            train_df, val_df, test_df, feature_columns
        )

        if spec["apply_selection"]:
            drop_percentile = float(
                features_cfg.selection.get("importance_drop_percentile", 20)
            )
            train_enc, val_enc, dropped_importance, importance_report = (
                prune_encoded_by_importance(
                    train_enc,
                    val_enc,
                    cfg.model,
                    class_imbalance_strategy=str(
                        cfg.class_imbalance.get("strategy", "inverse_frequency")
                    ),
                    drop_percentile=drop_percentile,
                )
            )
            selection_report["importance"] = importance_report
            selection_report["dropped_importance"] = dropped_importance

        result = _train_macro_f1(train_enc, val_enc, cfg)
        macro_f1 = result["macro_f1"]
        delta = None if previous_macro_f1 is None else macro_f1 - previous_macro_f1

        experiments[exp_id] = {
            "label": spec["label"],
            "groups": sorted(spec["groups"]),
            "apply_selection": spec["apply_selection"],
            "n_raw_features": len(feature_columns),
            "n_encoded_features": result["n_features"],
            "macro_f1": macro_f1,
            "delta_vs_previous": delta,
            "best_iteration": result["best_iteration"],
            "selection": selection_report or None,
        }
        previous_macro_f1 = macro_f1

    output = {
        "feature_version": cfg.feature_version,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "experiments": experiments,
    }

    reports_dir = ensure_dir(cfg.path("reports"))
    output_path = reports_dir / "ablation_results.json"
    sio.write_json(output_path, output)
    output["output"] = str(output_path.relative_to(PROJECT_ROOT))
    return output
