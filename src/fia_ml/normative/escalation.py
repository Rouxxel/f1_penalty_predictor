"""Pre-compute point-in-time history counters for escalation rules. Phase C."""

from __future__ import annotations

import pandas as pd

from fia_ml.normative.config import NormativeConfig


def add_escalation_columns(df: pd.DataFrame, cfg: NormativeConfig) -> pd.DataFrame:
    """Add driver history counters used by normative rule conditions."""
    raise NotImplementedError(
        "Escalation pre-pass is not implemented yet (Normative Rules Phase C)."
    )
