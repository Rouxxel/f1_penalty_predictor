"""Tests for normative rules loader"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.rules_loader import load_rules, rules_file_hash
from fia_ml.normative.schema import RulesValidationError, parse_rules_document
from fia_ml.paths import DEFAULT_NORMATIVE_RULES_CONFIG


@pytest.fixture
def rules_path() -> Path:
    return DEFAULT_NORMATIVE_RULES_CONFIG


def test_load_default_rules_file(rules_path: Path) -> None:
    loaded = load_rules(rules_path)
    assert loaded.version == "1.0.0"
    assert loaded.rule_count == 10
    assert loaded.document.rules[-1].id == "default_unmatched"
    assert loaded.content_hash == rules_file_hash(rules_path)


def test_rules_sorted_by_priority(rules_path: Path) -> None:
    loaded = load_rules(rules_path)
    priorities = [rule.priority for rule in loaded.document.rules]
    assert priorities == sorted(priorities)


def test_reject_duplicate_rule_ids() -> None:
    raw = {
        "version": "1.0.0",
        "description": "test",
        "assumptions": ["one"],
        "rules": [
            {
                "id": "dup",
                "priority": 1,
                "conditions": {"incident_type": "collision"},
                "outcome": {"penalty_detail": "no_action", "penalty_severity": 0},
            },
            {
                "id": "dup",
                "priority": 2,
                "conditions": {"default": True},
                "outcome": {"penalty_detail": "manual_review", "penalty_severity": 0},
            },
        ],
    }
    with pytest.raises(RulesValidationError, match="Duplicate rule id"):
        parse_rules_document(raw)


def test_reject_invalid_penalty_detail() -> None:
    raw = {
        "version": "1.0.0",
        "description": "test",
        "assumptions": ["one"],
        "rules": [
            {
                "id": "bad",
                "priority": 1,
                "conditions": {"incident_type": "collision"},
                "outcome": {"penalty_detail": "ban_hammer", "penalty_severity": 2},
            },
            {
                "id": "default",
                "priority": 99,
                "conditions": {"default": True},
                "outcome": {"penalty_detail": "manual_review", "penalty_severity": 0},
            },
        ],
    }
    with pytest.raises(RulesValidationError, match="invalid penalty_detail"):
        parse_rules_document(raw)


def test_require_exactly_one_default_rule() -> None:
    raw = {
        "version": "1.0.0",
        "description": "test",
        "assumptions": ["one"],
        "rules": [
            {
                "id": "only",
                "priority": 1,
                "conditions": {"incident_type": "collision"},
                "outcome": {"penalty_detail": "no_action", "penalty_severity": 0},
            }
        ],
    }
    with pytest.raises(RulesValidationError, match="Exactly one rule"):
        parse_rules_document(raw)


def test_validate_rules_cli(rules_path: Path) -> None:
    from fia_ml.normative.run_normative import main

    assert main(["--rules", str(rules_path), "--validate-rules"]) == 0
