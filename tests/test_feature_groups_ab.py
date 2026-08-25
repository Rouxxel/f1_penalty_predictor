"""Tests for V2 race and championship feature groups"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.driver import compute_driver_features
from fia_ml.features.race import _race_stage_from_completion, compute_race_features


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig.from_yaml()


def test_race_stage_boundaries() -> None:
    completion = pd.Series([10.0, 33.0, 50.0, 66.0, 80.0, 90.0, pd.NA])
    stages = _race_stage_from_completion(
        completion,
        early_max=33,
        middle_max=66,
        late_max=85,
    )
    assert stages.tolist() == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "final_laps",
        pd.NA,
    ]


def test_round_progress_and_round_flags(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "round": [1, 10, 22],
            "rounds": [22, 22, 22],
            "completion_percentage": [50.0, 50.0, 50.0],
        }
    )
    out = compute_race_features(df, features_cfg)
    assert out.loc[0, "round_progress"] == pytest.approx(1 / 22)
    assert bool(out.loc[0, "is_first_round"]) is True
    assert bool(out.loc[0, "is_last_round"]) is False
    assert bool(out.loc[2, "is_last_round"]) is True


def test_title_contender_logic(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "round": [5, 5, 5, 5],
            "driver": ["leader", "challenger", "midfield", "backmarker"],
            "driver_standing": [1, 2, 8, 15],
            "driver_points": [100, 75, 30, 5],
            "driver_construct_standing": [1, 1, 4, 8],
            "opponent": [None, None, None, None],
            "rounds": [24, 24, 24, 24],
        }
    )
    out = compute_driver_features(df, features_cfg)
    assert out.loc[0, "points_gap_to_leader"] == 0
    assert out.loc[1, "points_gap_to_leader"] == 25
    assert bool(out.loc[1, "title_contender"]) is True
    assert bool(out.loc[2, "title_contender"]) is False
    assert bool(out.loc[0, "construct_title_contender"]) is True
    assert bool(out.loc[3, "construct_title_contender"]) is False
    assert out.loc[0, "points_available_remaining"] == pytest.approx(19 * 25)


def test_points_gap_to_opponent_only_when_opponent_exists(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "season": [2019, 2019],
            "round": [1, 1],
            "driver": ["a", "b"],
            "driver_standing": [1, 2],
            "driver_points": [50, 40],
            "driver_construct_standing": [1, 2],
            "opponent": ["b", None],
            "opponent_points": [40, None],
            "rounds": [21, 21],
        }
    )
    out = compute_driver_features(df, features_cfg)
    assert out.loc[0, "points_gap_to_opponent"] == 10
    assert pd.isna(out.loc[1, "points_gap_to_opponent"])
