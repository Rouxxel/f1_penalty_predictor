"""Ablation experiments comparing V1 vs cumulative V2 feature groups."""

from __future__ import annotations

from typing import Any

from fia_ml.training.config import TrainingConfig


def run_ablation(cfg: TrainingConfig) -> dict[str, Any]:
    """Run experiments A–E from FEATURE_ENGINEERING_PLAN.md. Implemented in Phase E."""
    raise NotImplementedError(
        "Stage 'ablation' is not implemented yet (Feature Engineering Phase E)."
    )
