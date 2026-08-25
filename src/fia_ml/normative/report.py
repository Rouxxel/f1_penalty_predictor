"""Generate normative deviation reports and figures. Implemented in Phase E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fia_ml.normative.config import NormativeConfig


def write_deviation_report(
    comparison: dict[str, Any],
    cfg: NormativeConfig,
    report_dir: Path,
) -> dict[str, str]:
    """Write markdown summary, CSV breakdowns, and figures."""
    raise NotImplementedError(
        "Deviation reporting is not implemented yet (Normative Rules Phase E)."
    )
