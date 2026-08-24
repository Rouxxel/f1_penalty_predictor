"""Prepare flattened parquet datasets for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.ingestion import load_processed_seasons, resolve_input_paths
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.encoding import (
    TARGET_COLUMN,
    fit_encode_splits,
    save_encoding_artifacts,
)
from fia_ml.preprocessing.feature_engineering import add_v1_features
from fia_ml.preprocessing.flatten import flatten_incidents
from fia_ml.preprocessing.leakage_filter import select_feature_columns
from fia_ml.preprocessing.splitting import temporal_split, verify_no_season_overlap
from fia_ml.preprocessing.target_mapping import add_penalty_severity, load_target_mapping
from fia_ml.training.config import TrainingConfig
from fia_ml.utils import secure_file_io as sio


def prepare_datasets(
    cfg: TrainingConfig,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Run full prepare stage: ingest → flatten → derive → split → encode → parquet."""
    verify_no_season_overlap(cfg.splits)

    inputs = cfg.inputs or {}
    paths = resolve_input_paths(
        explicit_paths=input_paths,
        seasons=inputs.get("seasons"),
        csv_glob=str(cfg.paths.get("csv_input_glob", "dataset/csv/processed_*.csv")),
    )
    raw = load_processed_seasons(paths)
    mapping = load_target_mapping(cfg.target_mapping_path)
    labeled = add_penalty_severity(raw, mapping)

    # Training pool: rows with a mapped penalty label (exclude Summons-only / unmapped).
    trainable = labeled[labeled[TARGET_COLUMN].notna()].copy()
    flattened = flatten_incidents(trainable)
    if flattened.empty:
        raise ValueError("No rows after flattening — check driver columns and alignment")

    enriched = add_v1_features(flattened)
    train_df, val_df, test_df = temporal_split(enriched, cfg.splits)

    feature_columns = select_feature_columns(enriched)
    train_enc, val_enc, test_enc, artifacts = fit_encode_splits(
        train_df, val_df, test_df, feature_columns
    )

    processed_dir = ensure_dir(cfg.path("processed"))
    models_dir = ensure_dir(cfg.path("models"))

    incidents_path = processed_dir / "incidents.parquet"
    enriched.to_parquet(incidents_path, index=False)

    features_all = pd.concat(
        [df for df in (train_enc, val_enc, test_enc) if df is not None and not df.empty],
        ignore_index=True,
    )
    features_path = processed_dir / "features.parquet"
    features_all.to_parquet(features_path, index=False)

    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "validation.parquet"
    train_enc.to_parquet(train_path, index=False)
    if not val_enc.empty:
        val_enc.to_parquet(val_path, index=False)

    test_path = None
    if test_enc is not None and not test_enc.empty:
        test_path = processed_dir / "test.parquet"
        test_enc.to_parquet(test_path, index=False)

    encoder_path = models_dir / "preprocessor.joblib"
    save_encoding_artifacts(artifacts, encoder_path)

    return {
        "input_files": [str(p.relative_to(PROJECT_ROOT)) for p in paths],
        "incidents_rows": len(enriched),
        "train_rows": len(train_enc),
        "validation_rows": len(val_enc),
        "test_rows": len(test_enc) if test_enc is not None else 0,
        "feature_columns": feature_columns,
        "outputs": {
            "incidents": str(incidents_path.relative_to(PROJECT_ROOT)),
            "features": str(features_path.relative_to(PROJECT_ROOT)),
            "train": str(train_path.relative_to(PROJECT_ROOT)),
            "validation": str(val_path.relative_to(PROJECT_ROOT)),
            "test": str(test_path.relative_to(PROJECT_ROOT)) if test_path else None,
            "preprocessor": str(encoder_path.relative_to(PROJECT_ROOT)),
        },
    }
