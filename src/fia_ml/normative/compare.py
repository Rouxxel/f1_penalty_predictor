"""Compare FIA actual vs normative outcomes. Implemented in Phase E."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.normative.config import NormativeConfig


def compare_outcomes(df: pd.DataFrame, cfg: NormativeConfig) -> dict[str, Any]:
    """Compute agreement metrics and per-row deviation columns."""
    raise NotImplementedError(
        "Deviation comparison is not implemented yet (Normative Rules Phase E)."
    )
