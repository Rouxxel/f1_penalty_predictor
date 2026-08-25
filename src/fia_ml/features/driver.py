"""Group B — championship context features (gaps, title contention)."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.common import empty_numeric_series, to_float
from fia_ml.features.config import FeaturesConfig


def _leader_points_by_round(df: pd.DataFrame) -> pd.Series:
    """Estimate championship leader points per (season, round) from flattened rows."""
    standing = to_float(df.get("driver_standing", pd.Series(dtype=float)))
    points = to_float(df.get("driver_points", pd.Series(dtype=float)))
    season = df.get("season")
    round_num = df.get("round")

    work = pd.DataFrame(
        {
            "season": season,
            "round": round_num,
            "_standing": standing,
            "_points": points,
        },
        index=df.index,
    )
    valid = work["_standing"].notna() & work["_points"].notna()
    leader_from_standing = (
        work.loc[valid]
        .sort_values(["season", "round", "_standing"])
        .groupby(["season", "round"], as_index=False)
        .first()
        .rename(columns={"_points": "leader_points"})
    )
    max_points = (
        work.loc[valid]
        .groupby(["season", "round"], as_index=False)["_points"]
        .max()
        .rename(columns={"_points": "leader_points"})
    )
    leaders = leader_from_standing.merge(
        max_points,
        on=["season", "round"],
        how="outer",
        suffixes=("_standing", "_max"),
    )
    leaders["leader_points"] = leaders["leader_points_standing"].fillna(
        leaders["leader_points_max"]
    )
    leaders = leaders[["season", "round", "leader_points"]]

    merged = work.merge(leaders, on=["season", "round"], how="left")
    return merged["leader_points"]


def compute_driver_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add championship gap and title-contender columns."""
    out = df.copy()
    champ_cfg = cfg.championship
    max_standing = int(champ_cfg.get("title_contender_max_standing", 3))
    max_gap = float(champ_cfg.get("title_contender_max_gap", 40))
    points_per_race = float(champ_cfg.get("points_per_race_max", 25))

    driver_points = to_float(out.get("driver_points", pd.Series(dtype=float)))
    opponent_points = to_float(out.get("opponent_points", pd.Series(dtype=float)))
    leader_points = _leader_points_by_round(out)

    out["points_gap_to_leader"] = leader_points - driver_points
    out["points_gap_to_opponent"] = (driver_points - opponent_points).abs()
    if "opponent" in out.columns:
        out.loc[out["opponent"].isna(), "points_gap_to_opponent"] = pd.NA
    else:
        out["points_gap_to_opponent"] = pd.NA

    driver_standing = to_float(out.get("driver_standing", pd.Series(dtype=float)))
    construct_standing = to_float(out.get("driver_construct_standing", pd.Series(dtype=float)))
    gap_to_leader = to_float(out["points_gap_to_leader"])

    out["title_contender"] = (
        driver_standing.notna()
        & gap_to_leader.notna()
        & (driver_standing <= max_standing)
        & (gap_to_leader < max_gap)
    )
    out["construct_title_contender"] = construct_standing.notna() & (
        construct_standing <= max_standing
    )

    round_num = to_float(out.get("round", pd.Series(pd.NA, index=out.index)))
    if "rounds" in out.columns:
        rounds = to_float(out["rounds"])
    else:
        rounds = empty_numeric_series(out.index)
    rounds_remaining = (rounds - round_num).clip(lower=0)
    out["points_available_remaining"] = rounds_remaining * points_per_race

    return out
