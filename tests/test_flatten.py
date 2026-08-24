"""Tests for incident flattening."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.preprocessing.flatten import flatten_incidents


def _two_driver_incident() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "2019_race_abc",
                "season": 2019,
                "round": 5,
                "circuit": "monaco",
                "country": "monaco",
                "session": "race",
                "incident_type": "collision",
                "num_drivers": 2,
                "drivers": "driver_a,driver_b",
                "nationalities": "nat_a,nat_b",
                "respective_teams": "team_a,team_b",
                "driver_standings": "3,7",
                "driver_points": "30,20",
                "construct_standings": "2,5",
                "construct_points": "100,50",
                "years_in_sport": "4,8",
                "superlicense_points_before_incident": "",
                "current_top_4_drivers": "driver_a,driver_x,driver_y,driver_z",
                "penalty_severity": 1,
                "penalty": "5s_time_penalty",
            }
        ]
    )


def test_two_driver_incident_produces_two_rows() -> None:
    flat = flatten_incidents(_two_driver_incident())
    assert len(flat) == 2
    assert set(flat["driver"]) == {"driver_a", "driver_b"}


def test_opponents_are_swapped() -> None:
    flat = flatten_incidents(_two_driver_incident())
    row_a = flat[flat["driver"] == "driver_a"].iloc[0]
    row_b = flat[flat["driver"] == "driver_b"].iloc[0]
    assert row_a["opponent"] == "driver_b"
    assert row_b["opponent"] == "driver_a"
    assert row_a["driver_standing"] == "3"
    assert row_b["opponent_standing"] == "3"


def test_single_driver_has_no_opponent() -> None:
    df = _two_driver_incident()
    df.loc[0, "drivers"] = "solo_driver"
    df.loc[0, "nationalities"] = "solo_nat"
    df.loc[0, "respective_teams"] = "solo_team"
    df.loc[0, "driver_standings"] = "1"
    df.loc[0, "driver_points"] = "10"
    df.loc[0, "construct_standings"] = "2"
    df.loc[0, "construct_points"] = "50"
    df.loc[0, "years_in_sport"] = "5"
    df.loc[0, "num_drivers"] = 1
    flat = flatten_incidents(df)
    assert len(flat) == 1
    assert flat.iloc[0]["opponent"] is None


def test_misaligned_multi_value_skipped() -> None:
    df = _two_driver_incident()
    df.loc[0, "driver_standings"] = "3"
    flat = flatten_incidents(df)
    assert flat.empty
