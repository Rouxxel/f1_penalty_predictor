"""Temporal correctness tests for driver history features"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.history import compute_history_features


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig.from_yaml()


def _driver_rows(
    rounds: list[int],
    severities: list[int],
    *,
    season: int = 2019,
    driver: str = "driver_a",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": [f"inc_{season}_{r}" for r in rounds],
            "driver": [driver] * len(rounds),
            "season": [season] * len(rounds),
            "round": rounds,
            "penalty_severity": severities,
        }
    )


def test_penalties_last_3_races_at_round_5(features_cfg: FeaturesConfig) -> None:
    df = _driver_rows([2, 3, 4, 5], [0, 1, 0, 1])
    out = compute_history_features(df, features_cfg)
    row = out[out["round"] == 5].iloc[0]
    assert row["penalties_last_3_races"] == 1
    assert row["incidents_last_3_races"] == 3
    assert row["career_incidents"] == 3


def test_same_round_rows_do_not_leak_into_career_counts(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["inc_a", "inc_b", "inc_c"],
            "driver": ["driver_a", "driver_a", "driver_a"],
            "season": [2019, 2019, 2019],
            "round": [5, 5, 6],
            "penalty_severity": [1, 0, 1],
        }
    )
    out = compute_history_features(df, features_cfg)
    second_same_round = out[out["incident_id"] == "inc_b"].iloc[0]
    next_round = out[out["incident_id"] == "inc_c"].iloc[0]
    assert second_same_round["career_incidents"] == 0
    assert next_round["career_incidents"] == 2


def test_career_counts_include_prior_seasons(features_cfg: FeaturesConfig) -> None:
    df = pd.concat(
        [
            _driver_rows([10], [1], season=2018),
            _driver_rows([1], [0], season=2019),
        ],
        ignore_index=True,
    )
    out = compute_history_features(df, features_cfg)
    row = out[out["season"] == 2019].iloc[0]
    assert row["career_incidents"] == 1
    assert row["career_penalties"] == 1


def test_races_since_last_penalty_same_season(features_cfg: FeaturesConfig) -> None:
    df = _driver_rows([2, 5], [1, 0])
    out = compute_history_features(df, features_cfg)
    row = out[out["round"] == 5].iloc[0]
    assert row["races_since_last_penalty"] == 2


def test_shuffle_round_order_changes_history(features_cfg: FeaturesConfig) -> None:
    ordered = _driver_rows([2, 3, 4], [0, 1, 0])
    shuffled = ordered.sample(frac=1, random_state=7)
    ordered_out = compute_history_features(ordered, features_cfg)
    shuffled_out = compute_history_features(shuffled, features_cfg).sort_values("round")
    pd.testing.assert_series_equal(
        ordered_out["penalties_last_3_races"].reset_index(drop=True),
        shuffled_out["penalties_last_3_races"].reset_index(drop=True),
    )
