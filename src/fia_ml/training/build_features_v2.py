"""Build V2 feature matrix from incidents.parquet."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.pipeline import add_v2_features
from fia_ml.features.selection import prune_encoded_by_importance, prune_raw_features
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.encoding import fit_encode_splits, save_encoding_artifacts
from fia_ml.preprocessing.leakage_filter import select_feature_columns
from fia_ml.preprocessing.splitting import temporal_split, verify_no_season_overlap
from fia_ml.training.config import TrainingConfig
from fia_ml.utils import secure_file_io as sio


def _feature_paths(cfg: TrainingConfig) -> dict[str, Path]:
    processed = cfg.path("processed")
    features_cfg = cfg.features or {}
    return {
        "incidents": processed / str(features_cfg.get("incidents_file", "incidents.parquet")),
        "features_v1": processed / str(features_cfg.get("features_v1_file", "features.parquet")),
        "features": processed / str(features_cfg.get("features_file", "features_v2.parquet")),
        "train": processed / str(features_cfg.get("train_file", "train_v2.parquet")),
        "validation": processed / str(features_cfg.get("validation_file", "validation_v2.parquet")),
        "test": processed / str(features_cfg.get("test_file", "test_v2.parquet")),
    }


def _features_config_path(cfg: TrainingConfig) -> Path:
    rel = cfg.paths.get("features_config", "configs/features.yaml")
    return PROJECT_ROOT / rel


def build_features_v2(cfg: TrainingConfig) -> dict[str, Any]:
    """Engineer V2 features, encode splits, and write v2 parquet artifacts."""
    verify_no_season_overlap(cfg.splits)
    paths = _feature_paths(cfg)

    if not paths["incidents"].exists():
        raise FileNotFoundError(
            f"Missing {paths['incidents']} — run V1 --stage prepare first"
        )

    incidents = pd.read_parquet(paths["incidents"])
    features_cfg = FeaturesConfig.from_yaml(_features_config_path(cfg))
    enriched = add_v2_features(incidents, features_cfg)

    train_df, val_df, test_df = temporal_split(enriched, cfg.splits)
    feature_columns = select_feature_columns(enriched)
    feature_columns, selection_report = prune_raw_features(
        train_df, feature_columns, features_cfg
    )
    train_enc, val_enc, test_enc, artifacts = fit_encode_splits(
        train_df, val_df, test_df, feature_columns
    )

    drop_percentile = float(features_cfg.selection.get("importance_drop_percentile", 20))
    train_enc, val_enc, dropped_importance, importance_report = prune_encoded_by_importance(
        train_enc,
        val_enc,
        cfg.model,
        class_imbalance_strategy=str(cfg.class_imbalance.get("strategy", "inverse_frequency")),
        drop_percentile=drop_percentile,
    )
    selection_report["importance"] = importance_report
    selection_report["dropped_importance"] = dropped_importance

    if test_enc is not None and not test_enc.empty:
        kept_encoded = importance_report.get("kept_encoded_columns", [])
        meta_cols = [c for c in test_enc.columns if not c.startswith(("cat__", "num__"))]
        test_enc = test_enc[kept_encoded + meta_cols]

    processed_dir = ensure_dir(cfg.path("processed"))
    models_dir = ensure_dir(cfg.path("models"))

    v1_features = paths["features_v1"]
    v1_source = processed_dir / "features.parquet"
    if v1_source.exists() and not v1_features.exists():
        shutil.copy2(v1_source, v1_features)

    features_all = pd.concat(
        [df for df in (train_enc, val_enc, test_enc) if df is not None and not df.empty],
        ignore_index=True,
    )
    features_all.to_parquet(paths["features"], index=False)
    train_enc.to_parquet(paths["train"], index=False)
    if not val_enc.empty:
        val_enc.to_parquet(paths["validation"], index=False)
    test_path = None
    if test_enc is not None and not test_enc.empty:
        test_enc.to_parquet(paths["test"], index=False)
        test_path = str(paths["test"].relative_to(PROJECT_ROOT))

    model_subdir = cfg.model_subdir()
    encoder_path = models_dir / f"preprocessor_{model_subdir}.joblib"
    save_encoding_artifacts(artifacts, encoder_path)

    reports_dir = ensure_dir(cfg.path("reports"))
    selection_path = reports_dir / "selection_report_v2.json"
    sio.write_json(selection_path, selection_report)

    return {
        "feature_version": cfg.feature_version,
        "incidents_rows": len(enriched),
        "train_rows": len(train_enc),
        "validation_rows": len(val_enc),
        "test_rows": len(test_enc) if test_enc is not None else 0,
        "feature_columns": feature_columns,
        "encoded_feature_columns": list(importance_report.get("kept_encoded_columns", [])),
        "selection": selection_report,
        "v2_groups": ["race", "driver", "history", "precedent"],
        "outputs": {
            "features_v1": str(v1_features.relative_to(PROJECT_ROOT)) if v1_features.exists() else None,
            "features_v2": str(paths["features"].relative_to(PROJECT_ROOT)),
            "train_v2": str(paths["train"].relative_to(PROJECT_ROOT)),
            "validation_v2": str(paths["validation"].relative_to(PROJECT_ROOT)),
            "test_v2": test_path,
            "preprocessor": str(encoder_path.relative_to(PROJECT_ROOT)),
            "selection_report": str(selection_path.relative_to(PROJECT_ROOT)),
        },
    }
