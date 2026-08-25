"""Tests for FIA vs normative deviation comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.normative.compare import add_deviation_columns, compare_outcomes
from fia_ml.normative.config import NormativeConfig


@pytest.fixture
def cfg() -> NormativeConfig:
    return NormativeConfig(
        comparison={
            "fia_label_column": "penalty_severity",
            "normative_label_column": "normative_penalty_severity",
            "include_ml_comparison": True,
        }
    )


def test_add_deviation_columns_marks_agreement(cfg) -> None:
    df = pd.DataFrame(
        {
            "penalty_severity": [0, 1, 2],
            "normative_penalty_severity": [0, 2, 2],
        }
    )
    out = add_deviation_columns(df, cfg)
    assert out["agreement_fia_normative"].tolist() == [True, False, True]
    assert out["deviation_direction"].tolist() == [0, 1, 0]
    assert out["deviation_magnitude"].tolist() == [0, 1, 0]


def test_compare_outcomes_aggregate_metrics(cfg) -> None:
    df = pd.DataFrame(
        {
            "incident_type": ["collision", "collision", "track_limits", "other"],
            "session": ["race", "race", "qualifying", "race"],
            "circuit": ["monaco", "monaco", "spa", "monza"],
            "season": [2024, 2024, 2024, 2025],
            "penalty_severity": [0, 1, 1, 2],
            "normative_penalty_severity": [0, 2, 1, 0],
            "normative_penalty_detail": ["no_action", "5s", "warning", "manual_review"],
            "row_id": ["a", "b", "c", "d"],
        }
    )
    result = compare_outcomes(df, cfg)
    agg = result["aggregate"]
    assert agg["row_count"] == 4
    assert agg["agreement_rate"] == 0.5
    assert agg["fia_harsher_rate"] == 0.25
    assert agg["normative_harsher_rate"] == 0.25
    assert agg["manual_review_rate"] == 0.25
    assert agg["matched_row_count"] == 3
    assert agg["agreement_rate_excl_manual_review"] == pytest.approx(2 / 3)
    assert len(result["breakdowns"]["incident_type"]) == 3
    assert result["top_deviations"][0]["row_id"] == "d"


def test_compare_outcomes_breakdown_sorts_by_disagreement(cfg) -> None:
    df = pd.DataFrame(
        {
            "incident_type": ["collision", "collision", "track_limits"],
            "session": ["race", "race", "race"],
            "circuit": ["monaco", "monaco", "spa"],
            "season": [2024, 2024, 2024],
            "penalty_severity": [0, 1, 1],
            "normative_penalty_severity": [0, 2, 1],
            "normative_penalty_detail": ["no_action", "5s", "warning"],
        }
    )
    result = compare_outcomes(df, cfg)
    incident_rows = result["breakdowns"]["incident_type"]
    assert incident_rows[0]["incident_type"] == "collision"
    assert incident_rows[0]["disagreement_rate"] == 0.5


def test_compare_outcomes_with_ml_predictions(cfg) -> None:
    df = pd.DataFrame(
        {
            "row_id": ["row_a", "row_b"],
            "penalty_severity": [0, 1],
            "normative_penalty_severity": [0, 2],
            "normative_penalty_detail": ["no_action", "5s"],
            "incident_type": ["collision", "collision"],
            "session": ["race", "race"],
            "circuit": ["monaco", "monaco"],
            "season": [2024, 2024],
        }
    )
    ml_path = ROOT / "tests" / "_tmp_compare" / "predictions.json"
    ml_path.parent.mkdir(parents=True, exist_ok=True)
    ml_path.write_text(
        '[{"row_id": "row_a", "pred_xgboost": 0}, {"row_id": "row_b", "pred_xgboost": 0}]',
        encoding="utf-8",
    )
    result = compare_outcomes(df, cfg, ml_predictions_path=ml_path)
    ml = result["ml_comparison"]
    assert ml["overlap_rows"] == 2
    assert ml["agreement_fia_ml"] == 0.5
    assert ml["agreement_normative_ml"] == 0.5
