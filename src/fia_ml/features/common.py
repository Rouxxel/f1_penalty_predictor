"""Shared helpers for V2 feature modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def empty_numeric_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype=float)
