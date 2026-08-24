"""Group A — race context features (race_stage, round progress)."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig


def compute_race_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add race-stage and season-progress columns. Implemented in Phase B."""
    _ = cfg
    return df.copy()
