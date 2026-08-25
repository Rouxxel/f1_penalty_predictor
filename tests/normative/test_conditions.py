"""Tests for normative condition evaluation (Phase B)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.conditions import evaluate_conditions, normalize_session
from fia_ml.normative.rules_loader import load_rules
from fia_ml.paths import DEFAULT_NORMATIVE_RULES_CONFIG


def test_implicit_eq_on_incident_type() -> None:
    row = {"incident_type": "collision", "session": "race"}
    conditions = {"incident_type": "collision"}
    assert evaluate_conditions(conditions, row) is True
    assert evaluate_conditions(conditions, {**row, "incident_type": "pit_lane"}) is False


def test_session_in_normalizes_qualifying_variants() -> None:
    row = {"session": "Qualifying"}
    conditions = {"session_in": ["race", "qualifying", "sprint"]}
    assert evaluate_conditions(conditions, row) is True


def test_fact_contains_any_is_case_insensitive() -> None:
    row = {"fact": "Car 44 left the track and GAINED AN ADVANTAGE"}
    conditions = {"fact_contains_any": ["gained an advantage"]}
    assert evaluate_conditions(conditions, row) is True


def test_fact_contains_any_false_when_fact_missing() -> None:
    conditions = {"fact_contains_any": ["racing incident"]}
    assert evaluate_conditions(conditions, {"incident_type": "collision"}) is False


def test_numeric_gte_and_lt() -> None:
    row = {"driver_track_limits_last_5_races": 2}
    assert evaluate_conditions({"driver_track_limits_last_5_races": {"gte": 2}}, row) is True
    assert evaluate_conditions({"driver_track_limits_last_5_races": {"lt": 2}}, row) is False


def test_nested_and_group() -> None:
    row = {"incident_type": "collision", "session": "race", "fact": "racing incident"}
    conditions = {
        "and": [
            {"incident_type": "collision"},
            {"fact_contains_any": ["racing incident"]},
        ]
    }
    assert evaluate_conditions(conditions, row) is True


def test_nested_or_group() -> None:
    row = {"incident_type": "yellow_flag"}
    conditions = {
        "or": [
            {"incident_type": "collision"},
            {"incident_type": "yellow_flag"},
        ]
    }
    assert evaluate_conditions(conditions, row) is True


def test_default_rule_matches_any_row() -> None:
    assert evaluate_conditions({"default": True}, {"incident_type": "other"}) is True


def test_field_contains_operator() -> None:
    row = {"penalty": "10 second time penalty"}
    conditions = {"penalty": {"contains": "time penalty"}}
    assert evaluate_conditions(conditions, row) is True


def test_combined_leaf_conditions_use_and_semantics() -> None:
    row = {
        "incident_type": "collision",
        "session": "race",
        "fact": "gained an advantage",
    }
    conditions = {
        "incident_type": "collision",
        "session_in": ["race", "qualifying"],
        "fact_contains_any": ["gained an advantage"],
    }
    assert evaluate_conditions(conditions, row) is True
    assert evaluate_conditions(conditions, {**row, "session": "practice"}) is False


def test_starter_rules_evaluate_without_error() -> None:
    loaded = load_rules(DEFAULT_NORMATIVE_RULES_CONFIG)
    row = {
        "incident_type": "track_limits",
        "session": "race",
        "driver_track_limits_last_5_races": 1,
        "fact": "",
    }
    matched = [
        rule.id
        for rule in loaded.document.rules
        if evaluate_conditions(rule.conditions, row)
    ]
    assert "track_limits_first_offence" in matched


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Race", "race"),
        ("qualifying", "qualifying"),
        ("Sprint Qualifying", "qualifying"),
        ("Sprint", "sprint"),
        ("practice", "practice"),
    ],
)
def test_normalize_session(raw: str, expected: str) -> None:
    assert normalize_session(raw) == expected
