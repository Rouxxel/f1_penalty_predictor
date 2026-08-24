"""Tests for penalty → penalty_severity mapping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.paths import TARGET_MAPPING_CONFIG
from fia_ml.preprocessing.target_mapping import load_target_mapping, map_penalty_to_severity


def test_no_penalty_patterns() -> None:
    mapping = load_target_mapping(TARGET_MAPPING_CONFIG)
    assert map_penalty_to_severity("no_further_action", mapping) == 0
    assert map_penalty_to_severity("No Further Action", mapping) == 0


def test_minor_penalty_patterns() -> None:
    mapping = load_target_mapping(TARGET_MAPPING_CONFIG)
    assert map_penalty_to_severity("5s_time_penalty", mapping) == 1
    assert map_penalty_to_severity("reprimand", mapping) == 1


def test_major_penalty_patterns() -> None:
    mapping = load_target_mapping(TARGET_MAPPING_CONFIG)
    assert map_penalty_to_severity("3_place_grid_penalty", mapping) == 2
    assert map_penalty_to_severity("disqualification", mapping) == 2


def test_empty_penalty_returns_none() -> None:
    mapping = load_target_mapping(TARGET_MAPPING_CONFIG)
    assert map_penalty_to_severity("", mapping) is None
