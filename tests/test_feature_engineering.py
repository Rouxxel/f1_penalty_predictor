"""Tests for V1 derived feature rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.preprocessing.feature_engineering import FORBIDDEN_REDUNDANT_DERIVED, add_v1_features
from fia_ml.preprocessing.flatten import flatten_incidents
from fia_ml.preprocessing.leakage_filter import select_feature_columns


def test_add_v1_features_does_not_create_session_flags() -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["i1", "i1"],
            "driver": ["a", "b"],
            "session": ["race", "qualifying"],
            "driver_team": ["t1", "t2"],
            "opponent_team": ["t2", "t1"],
            "driver_standing": [1, 5],
            "opponent_standing": [5, 1],
            "driver_points": [100, 20],
            "opponent_points": [20, 100],
            "full_laps": [50, 50],
            "lap": [10, 20],
            "lap_remaining": [40, 30],
            "completion_percentage": [20.0, 40.0],
            "num_drivers": [2, 2],
        }
    )
    out = add_v1_features(df)
    assert "is_race_session" not in out.columns
    assert "is_qualifying" not in out.columns
    assert "top_4_driver" not in out.columns
    assert "laps_remaining" not in out.columns
    assert out.loc[0, "same_team"].item() is False
    assert out.loc[0, "standing_difference"].item() == -4.0


def test_flatten_does_not_create_top4_flags() -> None:
    df = pd.DataFrame(
        [
            {
                "incident_id": "x",
                "session": "race",
                "drivers": "driver_a",
                "nationalities": "nat_a",
                "respective_teams": "team_a",
                "driver_standings": "2",
                "driver_points": "50",
                "construct_standings": "1",
                "construct_points": "100",
                "years_in_sport": "5",
                "current_top_4_drivers": "driver_a,driver_x,driver_y,driver_z",
                "penalty_severity": 1,
                "penalty": "5s",
            }
        ]
    )
    flat = flatten_incidents(df)
    assert "top_4_driver" not in flat.columns
    assert "top_4_opponent" not in flat.columns


def test_redundant_derived_columns_never_selected() -> None:
    df = pd.DataFrame(
        {
            "circuit": ["monza"],
            "session": ["race"],
            "driver": ["a"],
            "round": [10],
            "season": [2019],
            "is_qualifying": [False],
            "top_4_driver": [True],
            "laps_remaining": [30],
        }
    )
    features = select_feature_columns(df)
    assert not (FORBIDDEN_REDUNDANT_DERIVED & set(features))
