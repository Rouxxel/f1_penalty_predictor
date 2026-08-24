"""Feature selection and pruning for V2 (correlation + importance)."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig


def select_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: FeaturesConfig,
) -> list[str]:
    """Return pruned feature column list. Implemented in Phase E."""
    _ = cfg
    return list(feature_columns)
