"""Batch apply normative rules to incidents. Implemented in Phase D."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.rules_loader import LoadedRules


def predict_normative(
    incidents: pd.DataFrame,
    rules: LoadedRules,
    cfg: NormativeConfig,
) -> pd.DataFrame:
    """Return incidents with normative_* outcome columns."""
    raise NotImplementedError(
        "Batch normative prediction is not implemented yet (Normative Rules Phase D)."
    )


def write_predictions_output(df: pd.DataFrame, output_path: Path) -> None:
    """Persist incidents_with_normative.parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
