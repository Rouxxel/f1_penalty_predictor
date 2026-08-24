"""Scaffold tests for V2 feature engineering (Phase A)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.features.config import FeaturesConfig
from fia_ml.features.pipeline import add_v2_features
from fia_ml.paths import DEFAULT_FEATURES_CONFIG, DEFAULT_TRAINING_V2_CONFIG
from fia_ml.training.ablation import run_ablation
from fia_ml.training.build_features_v2 import build_features_v2
from fia_ml.training.config import TrainingConfig
from fia_ml.training.pipeline import Stage, run_training
from fia_ml.training.run_training import build_parser


@pytest.fixture
def v2_cfg() -> TrainingConfig:
    return TrainingConfig.from_yaml(DEFAULT_TRAINING_V2_CONFIG)


def test_features_config_loads() -> None:
    assert DEFAULT_FEATURES_CONFIG.exists()
    cfg = FeaturesConfig.from_yaml()
    assert cfg.precedent["min_precedent_count"] == 3
    assert cfg.history["rolling_windows"] == [3, 5]
    assert cfg.championship["title_contender_max_gap"] == 40


def test_v2_training_config_loads(v2_cfg: TrainingConfig) -> None:
    assert DEFAULT_TRAINING_V2_CONFIG.exists()
    assert v2_cfg.feature_version == "v2"
    assert v2_cfg.model_subdir() == "xgboost_v2"
    assert v2_cfg.features["train_file"] == "train_v2.parquet"


def test_cli_accepts_v2_stages() -> None:
    args = build_parser().parse_args(["--stage", "features_v2"])
    assert args.stage == Stage.FEATURES_V2.value
    args = build_parser().parse_args(["--stage", "ablation"])
    assert args.stage == Stage.ABLATION.value


def test_add_v2_features_preserves_rows() -> None:
    df = pd.DataFrame(
        {
            "incident_id": ["i1"],
            "driver": ["a"],
            "session": ["race"],
            "driver_team": ["t1"],
            "opponent_team": [None],
            "driver_standing": [1],
            "opponent_standing": [None],
            "driver_points": [100],
            "opponent_points": [None],
            "full_laps": [50],
            "lap": [10],
            "lap_remaining": [40],
            "completion_percentage": [20.0],
            "num_drivers": [1],
        }
    )
    out = add_v2_features(df, FeaturesConfig.from_yaml())
    assert len(out) == 1
    assert "is_qualifying" not in out.columns


def test_features_v2_stage_builds_parquet(v2_cfg: TrainingConfig) -> None:
    work_dir = ROOT / "tests" / "_tmp_features_v2"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    processed = work_dir / "processed"
    processed.mkdir(parents=True)
    incidents = pd.DataFrame(
        {
            "incident_id": ["2019_a", "2025_b"],
            "season": [2019, 2025],
            "round": [1, 1],
            "driver": ["a", "b"],
            "session": ["race", "qualifying"],
            "driver_team": ["t1", "t2"],
            "opponent_team": [None, None],
            "driver_standing": [1, 3],
            "opponent_standing": [None, None],
            "driver_points": [100, 50],
            "opponent_points": [None, None],
            "full_laps": [50, 50],
            "lap": [10, 20],
            "lap_remaining": [40, 30],
            "completion_percentage": [20.0, 40.0],
            "num_drivers": [1, 1],
            "penalty_severity": [1, 0],
            "penalty": ["5s", "no further action"],
            "circuit": ["monza", "monaco"],
            "country": ["italy", "monaco"],
            "incident_type": ["collision", "track_limits"],
        }
    )
    incidents.to_parquet(processed / "incidents.parquet", index=False)
    incidents.to_parquet(processed / "features.parquet", index=False)

    v2_cfg.paths = dict(v2_cfg.paths)
    v2_cfg.paths["processed"] = str(processed)
    v2_cfg.paths["models"] = str(work_dir / "models")
    v2_cfg.paths["reports"] = str(work_dir / "reports")

    result = build_features_v2(v2_cfg)
    assert result["train_rows"] == 1
    assert result["validation_rows"] == 1
    assert (processed / "features_v2.parquet").exists()
    assert (processed / "train_v2.parquet").exists()

    shutil.rmtree(work_dir)


def test_ablation_stage_not_implemented(v2_cfg: TrainingConfig) -> None:
    with pytest.raises(NotImplementedError):
        run_ablation(v2_cfg)
