"""Tests for leakage column filtering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.preprocessing.leakage_filter import (
    FORBIDDEN_FEATURE_COLUMNS,
    assert_no_leakage,
    select_feature_columns,
)


def test_forbidden_columns_never_selected() -> None:
    df = pd.DataFrame(
        {
            "circuit": ["monza"],
            "penalty": ["5s_time_penalty"],
            "penalty_severity": [1],
            "driver_at_fault": ["driver_a"],
            "incident_id": ["x"],
            "driver": ["driver_a"],
            "round": [10],
            "season": [2019],
        }
    )
    features = select_feature_columns(df)
    assert_no_leakage(features)
    assert "penalty" not in features
    assert "penalty_severity" not in features
    assert "incident_id" not in features


def test_positions_dropped_when_sparse() -> None:
    df = pd.DataFrame(
        {
            "circuit": ["a", "b"],
            "driver": ["d1", "d2"],
            "round": [1, 2],
            "season": [2019, 2019],
            "positions_of_involved parties": ["", ""],
        }
    )
    features = select_feature_columns(df)
    assert "positions_of_involved parties" not in features


def test_assert_no_leakage_raises() -> None:
    try:
        assert_no_leakage(["penalty", "circuit"])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_environmental_columns_are_schema_only() -> None:
    """flag, safety_car, track/weather conditions have no derived boolean twins."""
    df = pd.DataFrame(
        {
            "circuit": ["monza"],
            "driver": ["a"],
            "round": [10],
            "season": [2019],
            "flag": ["yellow_flag"],
            "safety_car": ["safety_car"],
            "track_conditions": ["dry"],
            "weather_conditions": ["sunny"],
            "session": ["race"],
        }
    )
    features = select_feature_columns(df)
    assert "flag" in features
    assert "safety_car" in features
    assert "track_conditions" in features
    assert "weather_conditions" in features
    assert "session" in features
    assert not any(name.startswith("is_") for name in features)
