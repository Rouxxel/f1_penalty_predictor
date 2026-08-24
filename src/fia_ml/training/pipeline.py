"""Orchestrate V1 model training stages."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from fia_ml.preprocessing.prepare import prepare_datasets
from fia_ml.training.config import TrainingConfig
from fia_ml.training.ablation import run_ablation
from fia_ml.training.build_features_v2 import build_features_v2
from fia_ml.training.evaluate import run_evaluation
from fia_ml.training.train_baselines import train_baselines
from fia_ml.training.train_xgboost import train_xgboost


class Stage(str, Enum):
    ALL = "all"
    PREPARE = "prepare"
    FEATURES_V2 = "features_v2"
    TRAIN = "train"
    EVALUATE = "evaluate"
    ABLATION = "ablation"


def run_training(
    cfg: TrainingConfig,
    stage: Stage = Stage.ALL,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Run one or more training pipeline stages."""
    results: dict[str, Any] = {"stage": stage.value, "status": "ok"}

    stages = _resolve_stages(stage, cfg)

    for current in stages:
        if current == Stage.PREPARE:
            results["prepare"] = _run_prepare(cfg, input_paths=input_paths)
        elif current == Stage.FEATURES_V2:
            results["features_v2"] = _run_features_v2(cfg)
        elif current == Stage.TRAIN:
            results["train"] = _run_train(cfg)
        elif current == Stage.EVALUATE:
            results["evaluate"] = _run_evaluate(cfg)
        elif current == Stage.ABLATION:
            results["ablation"] = _run_ablation(cfg)

    return results


def _resolve_stages(stage: Stage, cfg: TrainingConfig) -> list[Stage]:
    if stage != Stage.ALL:
        return [stage]
    if cfg.feature_version == "v2":
        return [Stage.FEATURES_V2, Stage.TRAIN, Stage.EVALUATE]
    return [Stage.PREPARE, Stage.TRAIN, Stage.EVALUATE]


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


def _run_features_v2(cfg: TrainingConfig) -> dict[str, Any]:
    return build_features_v2(cfg)


def _run_ablation(cfg: TrainingConfig) -> dict[str, Any]:
    return run_ablation(cfg)


def _run_evaluate(cfg: TrainingConfig) -> dict[str, Any]:
    return run_evaluation(cfg)
