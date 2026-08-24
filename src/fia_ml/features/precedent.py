"""Group E — groupby precedent penalty-rate statistics."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig


def compute_precedent_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Add temporally correct precedent rate columns. Implemented in Phase D."""
    _ = cfg
    return df.copy()
