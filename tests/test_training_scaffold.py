"""Scaffold tests for V1 model training"""

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


def test_prepare_stage_runs_on_fixture(cfg: TrainingConfig) -> None:
    """Prepare stage runs end-to-end on a minimal synthetic CSV."""
    import pandas as pd
    import shutil

    work_dir = ROOT / "tests" / "_tmp_prepare"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    fixture = pd.DataFrame(
        [
            {
                "incident_id": "2019_race_test01",
                "circuit": "monza",
                "country": "italy",
                "first_season": "1950",
                "round": "14",
                "season": "2019",
                "current_top_4_drivers": "lewis_hamilton,max_verstappen",
                "rounds": "21",
                "num_teams": "10",
                "lap": "",
                "lap_remaining": "",
                "full_laps": "53",
                "completion_percentage": "",
                "sector": "1",
                "flag": "",
                "safety_car": "none",
                "track_conditions": "dry",
                "weather_conditions": "sunny",
                "session": "race",
                "incident_type": "collision",
                "severity": "",
                "positions_of_involved parties": "",
                "num_drivers": "2",
                "drivers": "max_verstappen,lewis_hamilton",
                "nationalities": "dutch,british",
                "respective_teams": "red_bull,mercedes",
                "driver_standings": "1,2",
                "driver_points": "100,90",
                "construct_standings": "1,2",
                "construct_points": "200,180",
                "years_in_sport": "5,12",
                "superlicense_points_before_incident": "",
                "investigation": "True",
                "incident_classification": "collision",
                "driver_at_fault": "",
                "penalty": "5s_time_penalty",
                "superlicense_points_added": "1",
                "mentioned_article": "Article 1",
            },
            {
                "incident_id": "2025_race_test02",
                "circuit": "monaco",
                "country": "monaco",
                "first_season": "1950",
                "round": "6",
                "season": "2025",
                "current_top_4_drivers": "lewis_hamilton,max_verstappen",
                "rounds": "24",
                "num_teams": "10",
                "lap": "",
                "lap_remaining": "",
                "full_laps": "78",
                "completion_percentage": "",
                "sector": "1",
                "flag": "",
                "safety_car": "none",
                "track_conditions": "dry",
                "weather_conditions": "sunny",
                "session": "race",
                "incident_type": "collision",
                "severity": "",
                "positions_of_involved parties": "",
                "num_drivers": "1",
                "drivers": "charles_leclerc",
                "nationalities": "monegasque",
                "respective_teams": "ferrari",
                "driver_standings": "3",
                "driver_points": "80",
                "construct_standings": "2",
                "construct_points": "150",
                "years_in_sport": "7",
                "superlicense_points_before_incident": "",
                "investigation": "True",
                "incident_classification": "collision",
                "driver_at_fault": "",
                "penalty": "reprimand",
                "superlicense_points_added": "",
                "mentioned_article": "Article 2",
            },
        ]
    )
    csv_path = work_dir / "processed_fixture.csv"
    fixture.to_csv(csv_path, index=False)

    cfg.paths = dict(cfg.paths)
    cfg.inputs = None
    cfg.splits = {
        "train_seasons": [2019],
        "validation_season": 2025,
        "test_season": None,
    }
    cfg.paths["processed"] = str(work_dir / "processed")
    cfg.paths["models"] = str(work_dir / "models")

    result = run_training(cfg, Stage.PREPARE, input_paths=[csv_path])
    assert result["prepare"]["train_rows"] == 2
    assert result["prepare"]["validation_rows"] == 1
    assert (work_dir / "processed" / "train.parquet").exists()

    shutil.rmtree(work_dir)
