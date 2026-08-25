"""Temporal correctness tests for normative escalation counters"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.conditions import evaluate_conditions
from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.escalation import add_escalation_columns


@pytest.fixture
def normative_cfg() -> NormativeConfig:
    return NormativeConfig.from_yaml()


def _rows(
    rounds: list[int],
    *,
    incident_types: list[str] | None = None,
    severities: list[int] | None = None,
    penalties: list[str] | None = None,
    season: int = 2019,
    driver: str = "driver_a",
) -> pd.DataFrame:
    n = len(rounds)
    return pd.DataFrame(
        {
            "incident_id": [f"inc_{season}_{r}" for r in rounds],
            "driver": [driver] * n,
            "season": [season] * n,
            "round": rounds,
            "incident_type": incident_types or ["track_limits"] * n,
            "penalty_severity": severities or [1] * n,
            "penalty": penalties or ["warning"] * n,
        }
    )


def test_track_limits_count_excludes_current_row(normative_cfg: NormativeConfig) -> None:
    df = _rows([1, 2, 3], incident_types=["track_limits"] * 3)
    out = add_escalation_columns(df, normative_cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["driver_track_limits_last_5_races"] == 2


def test_same_round_rows_do_not_count_as_prior(normative_cfg: NormativeConfig) -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["inc_a", "inc_b", "inc_c"],
            "driver": ["driver_a", "driver_a", "driver_a"],
            "season": [2019, 2019, 2019],
            "round": [5, 5, 6],
            "incident_type": ["track_limits", "track_limits", "track_limits"],
            "penalty_severity": [1, 1, 1],
            "penalty": ["warning", "warning", "warning"],
        }
    )
    out = add_escalation_columns(df, normative_cfg)
    same_round_second = out[out["incident_id"] == "inc_b"].iloc[0]
    next_round = out[out["incident_id"] == "inc_c"].iloc[0]
    assert same_round_second["driver_track_limits_last_5_races"] == 0
    assert next_round["driver_track_limits_last_5_races"] == 2


def test_repeat_track_limits_triggers_escalation_rule(normative_cfg: NormativeConfig) -> None:
    df = _rows([1, 2, 3], incident_types=["track_limits"] * 3, severities=[1, 1, 1])
    out = add_escalation_columns(df, normative_cfg)
    row = out[out["round"] == 3].iloc[0]
    assert evaluate_conditions(
        {"incident_type": "track_limits", "driver_track_limits_last_5_races": {"gte": 2}},
        row.to_dict(),
    )


def test_window_limits_counts_to_last_five_rounds(normative_cfg: NormativeConfig) -> None:
    df = _rows(
        list(range(1, 9)),
        incident_types=["track_limits"] * 8,
        severities=[1] * 8,
    )
    out = add_escalation_columns(df, normative_cfg)
    row = out[out["round"] == 8].iloc[0]
    assert row["driver_track_limits_last_5_races"] == 5


def test_penalties_counter_uses_fia_severity(normative_cfg: NormativeConfig) -> None:
    df = _rows(
        [1, 2, 3],
        incident_types=["collision", "collision", "collision"],
        severities=[0, 1, 0],
        penalties=["no_further_action", "5 second time penalty", "no_further_action"],
    )
    out = add_escalation_columns(df, normative_cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["driver_penalties_last_5_races"] == 1


def test_normative_history_mode_uses_normative_severity() -> None:
    cfg = NormativeConfig(escalation={"mode": "normative_history"})
    df = _rows([1, 2, 3], severities=[0, 0, 0])
    df["normative_penalty_severity"] = [0, 1, 0]
    df["normative_penalty_detail"] = ["no_action", "warning", "no_action"]
    out = add_escalation_columns(df, cfg)
    row = out[out["round"] == 3].iloc[0]
    assert row["driver_penalties_last_5_races"] == 1
    assert row["driver_warnings_last_10_races"] == 1


def test_shuffle_round_order_is_stable(normative_cfg: NormativeConfig) -> None:
    ordered = _rows([2, 3, 4, 5], incident_types=["track_limits"] * 4)
    shuffled = ordered.sample(frac=1, random_state=3)
    ordered_out = add_escalation_columns(ordered, normative_cfg)
    shuffled_out = add_escalation_columns(shuffled, normative_cfg).sort_values("round")
    pd.testing.assert_series_equal(
        ordered_out["driver_track_limits_last_5_races"].reset_index(drop=True),
        shuffled_out["driver_track_limits_last_5_races"].reset_index(drop=True),
    )
