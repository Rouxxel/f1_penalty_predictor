"""Group E — groupby precedent penalty-rate statistics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from fia_ml.features.common import build_similarity_key, is_strictly_prior, to_float
from fia_ml.features.config import FeaturesConfig

PRECEDENT_OUTPUT_COLUMNS = (
    "precedent_count",
    "precedent_no_penalty_rate",
    "precedent_minor_penalty_rate",
    "precedent_major_penalty_rate",
)


def _resolve_similarity_key(cfg: FeaturesConfig) -> list[str]:
    precedent_cfg = cfg.precedent
    if "active_similarity_key" in precedent_cfg:
        return list(precedent_cfg["active_similarity_key"])
    keys = precedent_cfg.get("similarity_keys", [])
    if keys:
        return list(keys[0])
    return ["incident_type", "session"]


def _severity_class(severity: float) -> int:
    sev = int(severity)
    return max(0, min(2, sev))


def _class_rates(counts: list[int], total: int) -> tuple[float, float, float]:
    if total <= 0:
        return (np.nan, np.nan, np.nan)
    return (
        counts[0] / total,
        counts[1] / total,
        counts[2] / total,
    )


def _prior_severities(
    season: int,
    round_num: int,
    rows: list[tuple[int, int, int]],
) -> list[int]:
    return [
        severity
        for prior_season, prior_round, severity in rows
        if is_strictly_prior(season, round_num, prior_season, prior_round)
    ]


def compute_precedent_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add temporally correct precedent rate columns from prior similar incidents."""
    required = {"season", "round", "incident_id", "penalty_severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for precedent features: {sorted(missing)}")

    key_columns = _resolve_similarity_key(cfg)
    missing_keys = set(key_columns) - set(df.columns)
    if missing_keys:
        raise ValueError(
            f"Missing similarity key columns for precedent features: {sorted(missing_keys)}"
        )

    min_count = int(cfg.precedent.get("min_precedent_count", 3))

    out = df.copy()
    out["_severity"] = to_float(out["penalty_severity"]).fillna(0)

    sorted_df = out.sort_values(["season", "round", "incident_id"])
    order = sorted_df.index

    results: dict[str, list[float]] = {col: [] for col in PRECEDENT_OUTPUT_COLUMNS}

    key_buckets: dict[tuple[str, ...], list[tuple[int, int, int]]] = defaultdict(list)
    global_prior_rows: list[tuple[int, int, int]] = []

    for idx in order:
        row = out.loc[idx]
        season = int(row["season"])
        round_num = int(row["round"])
        severity_class = _severity_class(float(row["_severity"]))
        key = build_similarity_key(row, key_columns)

        prior_group = _prior_severities(season, round_num, key_buckets[key])
        precedent_count = len(prior_group)

        global_prior = _prior_severities(season, round_num, global_prior_rows)
        global_counts = [0, 0, 0]
        for severity in global_prior:
            global_counts[severity] += 1
        global_rates = _class_rates(global_counts, len(global_prior))

        if precedent_count < min_count:
            no_rate, minor_rate, major_rate = global_rates
        else:
            group_counts = [0, 0, 0]
            for severity in prior_group:
                group_counts[severity] += 1
            no_rate, minor_rate, major_rate = _class_rates(group_counts, precedent_count)

        results["precedent_count"].append(float(precedent_count))
        results["precedent_no_penalty_rate"].append(no_rate)
        results["precedent_minor_penalty_rate"].append(minor_rate)
        results["precedent_major_penalty_rate"].append(major_rate)

        key_buckets[key].append((season, round_num, severity_class))
        global_prior_rows.append((season, round_num, severity_class))

    for col in PRECEDENT_OUTPUT_COLUMNS:
        out[col] = pd.Series(results[col], index=order)

    out = out.drop(columns=["_severity"])
    return out
