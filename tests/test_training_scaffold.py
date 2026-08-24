"""Scaffold tests for V1 model training (Phase A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.paths import DEFAULT_TRAINING_CONFIG, TARGET_MAPPING_CONFIG
from fia_ml.training.config import TrainingConfig
from fia_ml.training.pipeline import Stage, run_training
from fia_ml.training.run_training import build_parser
from fia_ml.utils import secure_file_io as sio


@pytest.fixture
def cfg() -> TrainingConfig:
    return TrainingConfig.from_yaml()


def test_training_config_loads(cfg: TrainingConfig) -> None:
    assert DEFAULT_TRAINING_CONFIG.exists()
    assert cfg.path("processed").name == "processed"
    assert cfg.splits["train_seasons"] == [2019]
    assert cfg.splits["validation_season"] == 2025
    assert cfg.model["num_class"] == 3
    assert cfg.random_state == 42


def test_target_mapping_config_exists() -> None:
    assert TARGET_MAPPING_CONFIG.exists()
    mapping = sio.read_yaml(TARGET_MAPPING_CONFIG)
    assert "classes" in mapping
    assert set(mapping["classes"].keys()) == {0, 1, 2}


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.stage == Stage.ALL.value


def test_prepare_stage_not_implemented_yet(cfg: TrainingConfig) -> None:
    with pytest.raises(NotImplementedError, match="Phase B"):
        run_training(cfg, Stage.PREPARE)
