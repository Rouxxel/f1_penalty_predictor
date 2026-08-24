"""Load prepared parquet splits for training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fia_ml.preprocessing.encoding import TARGET_COLUMN
from fia_ml.training.config import TrainingConfig


def load_train_val_frames(cfg: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = cfg.path("processed")
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "validation.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Missing train.parquet or validation.parquet — run --stage prepare first"
        )

    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c.startswith(("cat__", "num__"))]]


def labels(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].astype(int)


def compute_sample_weights(y: np.ndarray, strategy: str) -> np.ndarray | None:
    if strategy != "inverse_frequency":
        return None
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {cls: len(y) / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.array([weight_map[label] for label in y], dtype=float)
