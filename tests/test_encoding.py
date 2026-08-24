"""Tests for encoding (fit on train only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.preprocessing.encoding import fit_encode_splits
from fia_ml.preprocessing.leakage_filter import select_feature_columns


def _split_frames() -> tuple[pd.DataFrame, pd.DataFrame, None]:
    train = pd.DataFrame(
        {
            "circuit": ["monza", "spa"],
            "round": [10, 11],
            "season": [2019, 2019],
            "driver": ["a", "b"],
            "same_team": [False, True],
            "penalty_severity": [1, 0],
            "penalty": ["5s", "none"],
            "row_id": ["r1", "r2"],
            "incident_id": ["i1", "i2"],
        }
    )
    val = pd.DataFrame(
        {
            "circuit": ["monaco"],
            "round": [5],
            "season": [2025],
            "driver": ["c"],
            "same_team": [False],
            "penalty_severity": [2],
            "penalty": ["grid"],
            "row_id": ["r3"],
            "incident_id": ["i3"],
        }
    )
    return train, val, None


def test_encoder_fit_on_train_only() -> None:
    train, val, test = _split_frames()
    combined = pd.concat([train, val], ignore_index=True)
    feature_cols = select_feature_columns(combined)
    train_enc, val_enc, _, artifacts = fit_encode_splits(train, val, test, feature_cols)

    assert len(train_enc) == 2
    assert len(val_enc) == 1
    assert "penalty_severity" in train_enc.columns
    assert len(artifacts.feature_columns) == len(feature_cols)
