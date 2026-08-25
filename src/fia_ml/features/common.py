"""Shared helpers for V2 feature modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def empty_numeric_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype=float)


def is_strictly_prior(
    season: int,
    round_num: int,
    prior_season: int,
    prior_round: int,
) -> bool:
    return (prior_season < season) or (prior_season == season and prior_round < round_num)


def normalize_key_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "__missing__"
    text = str(value).strip().lower()
    return text if text else "__missing__"


def build_similarity_key(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(normalize_key_value(row.get(col)) for col in columns)
