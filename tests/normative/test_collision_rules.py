"""Fixture-based tests for collision normative rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.predict import predict_normative
from fia_ml.normative.rules_loader import load_rules
from fia_ml.paths import DEFAULT_NORMATIVE_RULES_CONFIG


@pytest.fixture
def normative_cfg() -> NormativeConfig:
    return NormativeConfig.from_yaml()


def _incident_row(**kwargs: object) -> pd.DataFrame:
    base = {
        "incident_id": "inc_test_1",
        "row_id": "row_test_1",
        "driver": "driver_a",
        "season": 2019,
        "round": 5,
        "session": "race",
        "incident_type": "collision",
        "penalty_severity": 0,
        "penalty": "no_further_action",
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_two_car_collision_no_fault_both_no_action(normative_cfg: NormativeConfig) -> None:
    rules = load_rules(DEFAULT_NORMATIVE_RULES_CONFIG)
    row = _incident_row(
        fact="The stewards determined this was a racing incident with no driver wholly or predominantly to blame"
    )
    out = predict_normative(row, rules, normative_cfg)
    assert out.iloc[0]["normative_rule_id"] == "collision_racing_no_fault"
    assert out.iloc[0]["normative_penalty_severity"] == 0


def test_left_track_advantage_gets_minor_penalty(normative_cfg: NormativeConfig) -> None:
    rules = load_rules(DEFAULT_NORMATIVE_RULES_CONFIG)
    row = _incident_row(
        fact="Car 33 left the track and gained a lasting advantage",
        incident_type="collision",
    )
    out = predict_normative(row, rules, normative_cfg)
    assert out.iloc[0]["normative_rule_id"] == "collision_avoidable_advantage"
    assert out.iloc[0]["normative_penalty_detail"] == "5s"
    assert out.iloc[0]["normative_penalty_severity"] == 1


def test_unmatched_incident_type_gets_manual_review(normative_cfg: NormativeConfig) -> None:
    rules = load_rules(DEFAULT_NORMATIVE_RULES_CONFIG)
    row = _incident_row(incident_type="weird_new_type", fact="")
    out = predict_normative(row, rules, normative_cfg)
    assert out.iloc[0]["normative_penalty_detail"] == "manual_review"
