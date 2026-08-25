"""Pre-compute point-in-time history counters for escalation rules."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from fia_ml.features.common import is_strictly_prior, to_float
from fia_ml.normative.config import NormativeConfig

ESCALATION_OUTPUT_COLUMNS = (
    "driver_track_limits_last_5_races",
    "driver_collisions_last_5_races",
    "driver_penalties_last_5_races",
    "driver_warnings_last_10_races",
)

_PRIOR_ENTRY = dict[str, int | float | str | bool]


def _normalize_incident_type(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _is_warning(penalty_text: object, severity: float) -> bool:
    if penalty_text is not None and not (isinstance(penalty_text, float) and pd.isna(penalty_text)):
        text = str(penalty_text).lower()
        if "warning" in text or "reprimand" in text:
            return True
    return severity == 1.0 and (
        penalty_text is None
        or (isinstance(penalty_text, float) and pd.isna(penalty_text))
        or "no_further" not in str(penalty_text).lower()
    )


def _severity_for_row(row: pd.Series, mode: str) -> float:
    if mode == "normative_history" and "normative_penalty_severity" in row.index:
        return float(to_float(pd.Series([row["normative_penalty_severity"]])).fillna(0).iloc[0])
    return float(to_float(pd.Series([row.get("penalty_severity", 0)])).fillna(0).iloc[0])


def _warning_for_row(row: pd.Series, mode: str) -> bool:
    if mode == "normative_history" and "normative_penalty_detail" in row.index:
        detail = row.get("normative_penalty_detail")
        if detail is None or (isinstance(detail, float) and pd.isna(detail)):
            return False
        return str(detail).lower() in {"warning", "reprimand"}
    severity = _severity_for_row(row, mode)
    return _is_warning(row.get("penalty"), severity)


def _count_in_recent_rounds(
    prior: list[_PRIOR_ENTRY],
    *,
    window: int,
    predicate: Callable[[_PRIOR_ENTRY], bool],
) -> int:
    prior_round_keys = sorted({(int(entry["season"]), int(entry["round"])) for entry in prior})
    recent_rounds = set(prior_round_keys[-window:])
    return sum(
        1
        for entry in prior
        if (int(entry["season"]), int(entry["round"])) in recent_rounds and predicate(entry)
    )


def _escalation_for_driver(group: pd.DataFrame, mode: str) -> pd.DataFrame:
    sorted_group = group.sort_values(["season", "round", "incident_id"])
    results: dict[str, list[float]] = {col: [] for col in ESCALATION_OUTPUT_COLUMNS}
    prior_rows: list[_PRIOR_ENTRY] = []

    for _, row in sorted_group.iterrows():
        season = int(row["season"])
        round_num = int(row["round"])
        incident_type = _normalize_incident_type(row.get("incident_type"))
        severity = _severity_for_row(row, mode)
        is_warning = _warning_for_row(row, mode)

        prior = [
            entry
            for entry in prior_rows
            if is_strictly_prior(season, round_num, int(entry["season"]), int(entry["round"]))
        ]

        results["driver_track_limits_last_5_races"].append(
            float(
                _count_in_recent_rounds(
                    prior,
                    window=5,
                    predicate=lambda entry: entry["incident_type"] == "track_limits",
                )
            )
        )
        results["driver_collisions_last_5_races"].append(
            float(
                _count_in_recent_rounds(
                    prior,
                    window=5,
                    predicate=lambda entry: entry["incident_type"] == "collision",
                )
            )
        )
        results["driver_penalties_last_5_races"].append(
            float(
                _count_in_recent_rounds(
                    prior,
                    window=5,
                    predicate=lambda entry: float(entry["severity"]) > 0,
                )
            )
        )
        results["driver_warnings_last_10_races"].append(
            float(
                _count_in_recent_rounds(
                    prior,
                    window=10,
                    predicate=lambda entry: bool(entry["is_warning"]),
                )
            )
        )

        prior_rows.append(
            {
                "season": season,
                "round": round_num,
                "incident_type": incident_type,
                "severity": severity,
                "is_warning": is_warning,
            }
        )

    return pd.DataFrame(results, index=sorted_group.index)


def add_escalation_columns(df: pd.DataFrame, cfg: NormativeConfig) -> pd.DataFrame:
    """Add driver history counters used by normative rule conditions."""
    required = {"driver", "season", "round", "incident_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for escalation counters: {sorted(missing)}")

    mode = str(cfg.escalation.get("mode", "fia_history"))
    if mode not in {"fia_history", "normative_history"}:
        raise ValueError(f"Unsupported escalation mode: {mode}")

    out = df.copy()
    parts: list[pd.DataFrame] = []
    for _, group in out.groupby("driver", sort=False):
        parts.append(_escalation_for_driver(group, mode))

    escalation_df = pd.concat(parts).sort_index()
    for col in ESCALATION_OUTPUT_COLUMNS:
        out[col] = escalation_df[col]

    return out
