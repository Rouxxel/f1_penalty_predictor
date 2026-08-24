"""Tests for temporal season splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.preprocessing.splitting import temporal_split, verify_no_season_overlap


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2019, 2019, 2025, 2025],
            "driver": ["a", "b", "c", "d"],
        }
    )


def test_temporal_split_assigns_seasons() -> None:
    splits = {
        "train_seasons": [2019],
        "validation_season": 2025,
        "test_season": None,
    }
    train, val, test = temporal_split(_sample_df(), splits)
    assert len(train) == 2
    assert len(val) == 2
    assert test is None
    assert set(train["season"]) == {2019}
    assert set(val["season"]) == {2025}


def test_unassigned_season_raises() -> None:
    splits = {
        "train_seasons": [2019],
        "validation_season": 2020,
        "test_season": None,
    }
    with pytest.raises(ValueError, match="not assigned"):
        temporal_split(_sample_df(), splits)


def test_overlap_detection() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        verify_no_season_overlap(
            {"train_seasons": [2019], "validation_season": 2019, "test_season": None}
        )
