"""Groups C & D — career and rolling-window driver history."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig


def compute_history_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add career and rolling history columns. Implemented in Phase C."""
    _ = cfg
    return df.copy()
