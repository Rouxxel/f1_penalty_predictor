"""Orchestrate V1 + V2 feature engineering."""

from __future__ import annotations

import pandas as pd

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.driver import compute_driver_features
from fia_ml.features.history import compute_history_features
from fia_ml.features.precedent import compute_precedent_features
from fia_ml.features.race import compute_race_features
from fia_ml.preprocessing.feature_engineering import add_v1_features


def add_v2_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Apply V1 relational features then V2 feature groups in order."""
    out = add_v1_features(df)
    out = compute_race_features(out, cfg)
    out = compute_driver_features(out, cfg)
    out = compute_history_features(out, cfg)
    out = compute_precedent_features(out, cfg)
    return out
