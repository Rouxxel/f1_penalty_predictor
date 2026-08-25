"""Temporal correctness tests for precedent features"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.precedent import compute_precedent_features


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig.from_yaml()


def _precedent_rows(
    rounds: list[int],
    severities: list[int],
    *,
    season: int = 2019,
    incident_type: str = "collision",
    session: str = "Race",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incident_id": [f"inc_{season}_{r}" for r in rounds],
            "season": [season] * len(rounds),
            "round": rounds,
            "penalty_severity": severities,
            "incident_type": [incident_type] * len(rounds),
            "session": [session] * len(rounds),
        }
    )


def test_precedent_rates_at_row_3_reflect_only_rows_1_and_2(features_cfg: FeaturesConfig) -> None:
    df = _precedent_rows([1, 2, 3, 4, 5], [0, 1, 2, 0, 1])
    out = compute_precedent_features(df, features_cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["precedent_count"] == 2
    assert row["precedent_no_penalty_rate"] == pytest.approx(0.5)
    assert row["precedent_minor_penalty_rate"] == pytest.approx(0.5)
    assert row["precedent_major_penalty_rate"] == pytest.approx(0.0)


def test_current_row_label_not_included_in_precedent_rates(features_cfg: FeaturesConfig) -> None:
    df = _precedent_rows([1, 2, 3], [0, 0, 2])
    out = compute_precedent_features(df, features_cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["precedent_count"] == 2
    assert row["precedent_major_penalty_rate"] == pytest.approx(0.0)


def test_same_round_incidents_excluded_from_precedent(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["inc_a", "inc_b", "inc_c"],
            "season": [2019, 2019, 2019],
            "round": [5, 5, 6],
            "penalty_severity": [0, 1, 2],
            "incident_type": ["collision", "collision", "collision"],
            "session": ["Race", "Race", "Race"],
        }
    )
    out = compute_precedent_features(df, features_cfg)
    same_round_second = out[out["incident_id"] == "inc_b"].iloc[0]
    next_round = out[out["incident_id"] == "inc_c"].iloc[0]
    assert same_round_second["precedent_count"] == 0
    assert next_round["precedent_count"] == 2


def test_sparse_group_falls_back_to_global_prior(features_cfg: FeaturesConfig) -> None:
    cfg = FeaturesConfig(
        precedent={
            "active_similarity_key": ["incident_type", "session"],
            "min_precedent_count": 3,
        }
    )
    df = _precedent_rows([1, 2, 3], [0, 1, 2])
    out = compute_precedent_features(df, cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["precedent_count"] == 2
    assert row["precedent_no_penalty_rate"] == pytest.approx(1 / 2)
    assert row["precedent_minor_penalty_rate"] == pytest.approx(1 / 2)
    assert row["precedent_major_penalty_rate"] == pytest.approx(0.0)


def test_empty_precedent_and_global_prior_are_nan(features_cfg: FeaturesConfig) -> None:
    df = _precedent_rows([1], [0])
    out = compute_precedent_features(df, features_cfg)
    row = out.iloc[0]
    assert row["precedent_count"] == 0
    assert np.isnan(row["precedent_no_penalty_rate"])
    assert np.isnan(row["precedent_minor_penalty_rate"])
    assert np.isnan(row["precedent_major_penalty_rate"])


def test_different_similarity_keys_do_not_mix(features_cfg: FeaturesConfig) -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["inc_1", "inc_2", "inc_3"],
            "season": [2019, 2019, 2019],
            "round": [1, 2, 3],
            "penalty_severity": [0, 1, 2],
            "incident_type": ["collision", "collision", "unsafe_release"],
            "session": ["Race", "Race", "Race"],
        }
    )
    out = compute_precedent_features(df, features_cfg)
    row = out[out["incident_id"] == "inc_3"].iloc[0]
    assert row["precedent_count"] == 0
