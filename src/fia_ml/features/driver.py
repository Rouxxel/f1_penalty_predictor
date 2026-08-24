"""Group B — championship context features (gaps, title contention)."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig


def compute_driver_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add championship gap and title-contender columns. Implemented in Phase B."""
    _ = cfg
    return df.copy()
