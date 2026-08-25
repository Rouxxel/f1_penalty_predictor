"""Groups C & D — career and rolling-window driver history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fia_ml.features.common import is_strictly_prior, to_float
from fia_ml.features.config import FeaturesConfig

HISTORY_OUTPUT_COLUMNS = (
    "career_incidents",
    "career_penalties",
    "career_major_penalties",
    "career_incidents_per_100_races",
    "career_penalties_per_100_races",
    "incidents_last_3_races",
    "incidents_last_5_races",
    "penalties_last_3_races",
    "penalties_last_5_races",
    "races_since_last_penalty",
    "races_since_last_incident",
)


def _rounds_since_same_season(
    season: int,
    round_num: int,
    last_season: int | None,
    last_round: int | None,
) -> float:
    if last_season is None or last_round is None:
        return np.nan
    if int(last_season) != int(season):
        return np.nan
    return float(int(round_num) - int(last_round) - 1)


def _history_for_driver(group: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    sorted_group = group.sort_values(["season", "round", "incident_id"])

    results: dict[str, list[float]] = {col: [] for col in HISTORY_OUTPUT_COLUMNS}
    prior_rows: list[dict[str, int | float]] = []

    for _, row in sorted_group.iterrows():
        season = int(row["season"])
        round_num = int(row["round"])
        severity = float(row["_severity"])

        prior = [
            entry
            for entry in prior_rows
            if is_strictly_prior(season, round_num, entry["season"], entry["round"])
        ]

        career_incidents = len(prior)
        career_penalties = sum(1 for entry in prior if entry["severity"] > 0)
        career_major_penalties = sum(1 for entry in prior if entry["severity"] == 2)

        prior_round_keys = sorted({(entry["season"], entry["round"]) for entry in prior})
        races_started = len(prior_round_keys)

        if races_started > 0:
            career_incidents_rate = career_incidents / races_started * 100
            career_penalties_rate = career_penalties / races_started * 100
        else:
            career_incidents_rate = np.nan
            career_penalties_rate = np.nan

        for window in windows:
            recent_rounds = set(prior_round_keys[-window:])
            recent_rows = [
                entry for entry in prior if (entry["season"], entry["round"]) in recent_rounds
            ]
            results[f"incidents_last_{window}_races"].append(float(len(recent_rows)))
            results[f"penalties_last_{window}_races"].append(
                float(sum(1 for entry in recent_rows if entry["severity"] > 0))
            )

        penalty_rounds = sorted(
            {(entry["season"], entry["round"]) for entry in prior if entry["severity"] > 0}
        )
        incident_rounds = prior_round_keys
        last_penalty = penalty_rounds[-1] if penalty_rounds else None
        last_incident = incident_rounds[-1] if incident_rounds else None

        if last_penalty is not None:
            last_pen_season, last_pen_round = last_penalty
        else:
            last_pen_season, last_pen_round = None, None
        if last_incident is not None:
            last_inc_season, last_inc_round = last_incident
        else:
            last_inc_season, last_inc_round = None, None

        results["career_incidents"].append(float(career_incidents))
        results["career_penalties"].append(float(career_penalties))
        results["career_major_penalties"].append(float(career_major_penalties))
        results["career_incidents_per_100_races"].append(career_incidents_rate)
        results["career_penalties_per_100_races"].append(career_penalties_rate)
        results["races_since_last_penalty"].append(
            _rounds_since_same_season(season, round_num, last_pen_season, last_pen_round)
        )
        results["races_since_last_incident"].append(
            _rounds_since_same_season(season, round_num, last_inc_season, last_inc_round)
        )

        prior_rows.append({"season": season, "round": round_num, "severity": severity})

    history = pd.DataFrame(results, index=sorted_group.index)
    return history


def compute_history_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add temporally correct career and rolling history columns per driver."""
    required = {"driver", "season", "round", "incident_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for history features: {sorted(missing)}")

    out = df.copy()
    out["_severity"] = to_float(out.get("penalty_severity", pd.Series(0, index=out.index))).fillna(0)

    windows = [int(w) for w in cfg.history.get("rolling_windows", [3, 5])]
    history_parts: list[pd.DataFrame] = []
    for _, group in out.groupby("driver", sort=False):
        history_parts.append(_history_for_driver(group, windows))

    history_df = pd.concat(history_parts).sort_index()
    for col in HISTORY_OUTPUT_COLUMNS:
        out[col] = history_df[col]

    out = out.drop(columns=["_severity"])
    return out
