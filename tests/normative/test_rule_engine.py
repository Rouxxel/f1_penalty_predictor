"""Tests for priority-ordered normative rule matching."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.rule_engine import match_rule
from fia_ml.normative.rules_loader import load_rules
from fia_ml.paths import DEFAULT_NORMATIVE_RULES_CONFIG


@pytest.fixture
def rules():
    return load_rules(DEFAULT_NORMATIVE_RULES_CONFIG).document.rules


def test_first_match_wins_for_collision_racing_incident(rules) -> None:
    row = {
        "incident_type": "collision",
        "session": "race",
        "fact": "This was a racing incident with no driver wholly or predominantly to blame",
    }
    rule, outcome = match_rule(rules, row)
    assert rule.id == "collision_racing_no_fault"
    assert outcome.penalty_detail == "no_action"
    assert outcome.penalty_severity == 0


def test_reckless_collision_takes_priority_over_unclassified(rules) -> None:
    row = {
        "incident_type": "collision",
        "session": "race",
        "fact": "reckless manoeuvre at turn 4",
    }
    rule, outcome = match_rule(rules, row)
    assert rule.id == "collision_reckless"
    assert outcome.penalty_severity == 2


def test_collision_without_fact_pattern_goes_to_manual_review(rules) -> None:
    row = {"incident_type": "collision", "session": "race", "fact": ""}
    rule, outcome = match_rule(rules, row)
    assert rule.id == "collision_unclassified"
    assert outcome.penalty_detail == "manual_review"


def test_default_rule_matches_unknown_incident_type(rules) -> None:
    row = {"incident_type": "unknown_type_xyz", "session": "race"}
    rule, outcome = match_rule(rules, row)
    assert rule.id == "default_unmatched"


def test_track_limits_repeat_escalation_rule(rules) -> None:
    row = {
        "incident_type": "track_limits",
        "session": "race",
        "driver_track_limits_last_5_races": 2,
    }
    rule, outcome = match_rule(rules, row)
    assert rule.id == "track_limits_repeat"
    assert outcome.penalty_detail == "5s"
