"""Group A — race context features (race_stage, round progress)."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.common import empty_numeric_series, to_float
from fia_ml.features.config import FeaturesConfig


def _race_stage_from_completion(
    completion: pd.Series,
    *,
    early_max: float,
    middle_max: float,
    late_max: float,
) -> pd.Series:
    stage = pd.Series(pd.NA, index=completion.index, dtype="object")
    valid = completion.notna()
    stage.loc[valid & (completion <= early_max)] = "early"
    stage.loc[valid & (completion > early_max) & (completion <= middle_max)] = "middle"
    stage.loc[valid & (completion > middle_max) & (completion <= late_max)] = "late"
    stage.loc[valid & (completion > late_max)] = "final_laps"
    return stage


def compute_race_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add race-stage and season-progress columns."""
    out = df.copy()
    race_cfg = cfg.race
    bins = race_cfg.get("race_stage_bins", {})

    completion = to_float(out.get("completion_percentage", pd.Series(dtype=float)))
    out["race_stage"] = _race_stage_from_completion(
        completion,
        early_max=float(bins.get("early_max", 33)),
        middle_max=float(bins.get("middle_max", 66)),
        late_max=float(bins.get("late_max", 85)),
    )

    round_num = to_float(out.get("round", pd.Series(pd.NA, index=out.index)))
    if "rounds" in out.columns:
        rounds = to_float(out["rounds"])
    else:
        rounds = empty_numeric_series(out.index)
    out["round_progress"] = (round_num / rounds.replace(0, pd.NA)).clip(lower=0, upper=1)
    out["is_first_round"] = round_num == 1
    out["is_last_round"] = round_num.notna() & rounds.notna() & (round_num == rounds)

    return out
