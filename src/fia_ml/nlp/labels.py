"""Penalty label helpers for NLP training (aligned with V1 tabular pipeline)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fia_ml.preprocessing.encoding import TARGET_COLUMN
from fia_ml.preprocessing.target_mapping import add_penalty_severity, load_target_mapping


def load_labels(mapping_path: Path) -> dict:
    return load_target_mapping(mapping_path)


def add_labels(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    mapping = load_labels(mapping_path)
    return add_penalty_severity(df, mapping)


def filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a mapped penalty_severity (excludes Summons-only / unmapped)."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing {TARGET_COLUMN} column — call add_labels first")
    return df[df[TARGET_COLUMN].notna()].copy()
