"""Orchestrate V1 model training stages."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from fia_ml.preprocessing.prepare import prepare_datasets
from fia_ml.training.config import TrainingConfig
from fia_ml.training.train_baselines import train_baselines
from fia_ml.training.train_xgboost import train_xgboost


class Stage(str, Enum):
    ALL = "all"
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"


def run_training(
    cfg: TrainingConfig,
    stage: Stage = Stage.ALL,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Run one or more training pipeline stages."""
    results: dict[str, Any] = {"stage": stage.value, "status": "ok"}

    stages = (
        [Stage.PREPARE, Stage.TRAIN, Stage.EVALUATE]
        if stage == Stage.ALL
        else [stage]
    )

    for current in stages:
        if current == Stage.PREPARE:
            results["prepare"] = _run_prepare(cfg, input_paths=input_paths)
        elif current == Stage.TRAIN:
            results["train"] = _run_train(cfg)
        elif current == Stage.EVALUATE:
            results["evaluate"] = _run_evaluate(cfg)

    return results


def _run_prepare(
    cfg: TrainingConfig,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    return prepare_datasets(cfg, input_paths=input_paths)


def _run_train(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "baseline": train_baselines(cfg),
        "xgboost": train_xgboost(cfg),
    }


def _run_evaluate(cfg: TrainingConfig) -> dict[str, Any]:
    raise NotImplementedError(
        "Stage 'evaluate' is not implemented yet (Phase E: metrics and reports)."
    )
