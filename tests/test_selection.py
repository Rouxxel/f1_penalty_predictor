"""Tests for V2 feature selection (Phase E)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.selection import prune_raw_features
from fia_ml.preprocessing.leakage_filter import select_feature_columns_for_groups


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig.from_yaml()


def test_drop_high_missing_columns(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "penalty_severity": [0, 1, 0, 1],
            "season": [2019, 2019, 2019, 2019],
            "round": [1, 2, 3, 4],
            "mostly_missing": [1.0, np.nan, np.nan, np.nan],
            "driver_standing": [1, 2, 3, 4],
        }
    )
    kept, report = prune_raw_features(
        df,
        ["mostly_missing", "driver_standing"],
        features_cfg,
    )
    assert "driver_standing" in kept
    assert "mostly_missing" in report["dropped_missing"]


def test_correlation_prune_drops_lower_mi_feature(features_cfg: FeaturesConfig) -> None:
    cfg = FeaturesConfig(
        selection={
            "max_missing_rate": 0.40,
            "correlation_threshold": 0.95,
            "importance_drop_percentile": 20,
        }
    )
    x = np.arange(20, dtype=float)
    df = pd.DataFrame(
        {
            "penalty_severity": [i % 3 for i in range(20)],
            "season": [2019] * 20,
            "round": x,
            "driver_standing": x + np.random.default_rng(0).normal(0, 0.01, size=20),
            "num_drivers": np.random.default_rng(1).normal(0, 1, size=20),
        }
    )
    kept, report = prune_raw_features(
        df,
        ["round", "driver_standing", "num_drivers"],
        cfg,
    )
    assert len(kept) == 2
    assert len(report["dropped_correlation"]) >= 1


def test_ablation_group_selection_excludes_v2_when_empty(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "session": ["Race"] * 2,
            "incident_type": ["collision"] * 2,
            "driver_standing": [1, 2],
            "round_progress": [0.1, 0.2],
            "precedent_count": [1, 2],
        }
    )
    v1_only = select_feature_columns_for_groups(df, frozenset())
    assert "round_progress" not in v1_only
    assert "precedent_count" not in v1_only
    assert "driver_standing" in v1_only

    with_precedent = select_feature_columns_for_groups(
        df, frozenset({"race", "championship", "history", "precedent"})
    )
    assert "precedent_count" in with_precedent
    assert "round_progress" in with_precedent
