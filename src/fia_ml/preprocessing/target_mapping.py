"""Map raw penalty strings to penalty_severity classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fia_ml.utils import secure_file_io as sio


def load_target_mapping(path: Path) -> dict[str, Any]:
    return sio.read_yaml(path)


def map_penalty_to_severity(penalty: str, mapping: dict[str, Any]) -> int | None:
    text = str(penalty or "").strip().lower().replace("_", " ")
    if not text:
        return None

    classes = mapping["classes"]
    for class_id in (2, 1, 0):
        class_cfg = classes[class_id]
        for pattern in class_cfg["patterns"]:
            if str(pattern).lower() in text:
                return int(class_id)
    return None


def add_penalty_severity(
    df,
    mapping: dict[str, Any],
    *,
    penalty_col: str = "penalty",
    target_col: str = "penalty_severity",
):
    """Return copy with penalty_severity column (nullable int)."""
    import pandas as pd

    out = df.copy()
    out[target_col] = out[penalty_col].apply(lambda p: map_penalty_to_severity(p, mapping))
    out[target_col] = out[target_col].astype("Int64")
    return out
