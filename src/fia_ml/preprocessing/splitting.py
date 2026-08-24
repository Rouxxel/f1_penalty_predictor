"""Temporal train / validation / test splits by season."""

from __future__ import annotations

from typing import Any

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split flattened rows by season. Never random."""
    train_seasons = {int(s) for s in splits.get("train_seasons", [])}
    val_season = splits.get("validation_season")
    test_season = splits.get("test_season")

    if val_season is not None:
        val_season = int(val_season)
    if test_season is not None:
        test_season = int(test_season)

    seasons = set(df["season"].dropna().astype(int).unique())
    assigned: dict[int, str] = {}
    for s in train_seasons:
        assigned[s] = "train"
    if val_season is not None:
        if val_season in assigned:
            raise ValueError(f"Season {val_season} appears in multiple splits")
        assigned[val_season] = "validation"
    if test_season is not None:
        if test_season in assigned:
            raise ValueError(f"Season {test_season} appears in multiple splits")
        assigned[test_season] = "test"

    unassigned = seasons - set(assigned.keys())
    if unassigned:
        raise ValueError(
            f"Seasons in data but not assigned to a split: {sorted(unassigned)}. "
            "Update configs/xgboost.yaml splits."
        )

    train_df = df[df["season"].astype(int).isin(train_seasons)].copy()
    val_df = df[df["season"].astype(int) == val_season].copy() if val_season is not None else pd.DataFrame()
    test_df = (
        df[df["season"].astype(int) == test_season].copy()
        if test_season is not None
        else None
    )

    if train_df.empty:
        raise ValueError("Training split is empty — check train_seasons in config")
    if val_season is not None and val_df.empty:
        raise ValueError(f"Validation split for season {val_season} is empty")

    return train_df, val_df, test_df


def verify_no_season_overlap(splits: dict[str, Any]) -> None:
    train = {int(s) for s in splits.get("train_seasons", [])}
    val = splits.get("validation_season")
    test = splits.get("test_season")
    if val is not None and int(val) in train:
        raise ValueError("validation_season overlaps train_seasons")
    if test is not None and int(test) in train:
        raise ValueError("test_season overlaps train_seasons")
    if val is not None and test is not None and int(val) == int(test):
        raise ValueError("validation_season equals test_season")
