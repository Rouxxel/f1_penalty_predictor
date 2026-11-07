"""Temporal train / validation / test splits by season."""

from __future__ import annotations

from typing import Any

import pandas as pd


def as_season_list(value: Any) -> list[int]:
    """Normalise a config value (None, one season, or a list of seasons) to a list of ints."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(s) for s in value]
    return [int(value)]


def temporal_split(
    df: pd.DataFrame,
    splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split flattened rows by season. Never random.

    ``validation_season`` and ``test_season`` accept a single season or a list.
    """
    train_seasons = set(as_season_list(splits.get("train_seasons")))
    val_seasons = as_season_list(splits.get("validation_season"))
    test_seasons = as_season_list(splits.get("test_season"))

    seasons = set(df["season"].dropna().astype(int).unique())
    assigned: dict[int, str] = {}
    for name, group in (
        ("train", train_seasons),
        ("validation", val_seasons),
        ("test", test_seasons),
    ):
        for s in group:
            if s in assigned:
                raise ValueError(f"Season {s} appears in multiple splits")
            assigned[s] = name

    unassigned = seasons - set(assigned.keys())
    if unassigned:
        raise ValueError(
            f"Seasons in data but not assigned to a split: {sorted(unassigned)}. "
            "Update configs/xgboost.yaml splits."
        )

    season_col = df["season"].astype(int)
    train_df = df[season_col.isin(train_seasons)].copy()
    val_df = df[season_col.isin(val_seasons)].copy() if val_seasons else pd.DataFrame()
    test_df = df[season_col.isin(test_seasons)].copy() if test_seasons else None

    if train_df.empty:
        raise ValueError("Training split is empty — check train_seasons in config")
    if val_seasons and val_df.empty:
        raise ValueError(f"Validation split for season(s) {val_seasons} is empty")

    return train_df, val_df, test_df


def verify_no_season_overlap(splits: dict[str, Any]) -> None:
    train = set(as_season_list(splits.get("train_seasons")))
    val = set(as_season_list(splits.get("validation_season")))
    test = set(as_season_list(splits.get("test_season")))
    if val & train:
        raise ValueError("validation_season overlaps train_seasons")
    if test & train:
        raise ValueError("test_season overlaps train_seasons")
    if val & test:
        raise ValueError("validation_season overlaps test_season")
