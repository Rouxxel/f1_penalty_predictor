import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.training.nlp_config import NlpTrainingConfig


@pytest.fixture
def cfg():
    return NlpTrainingConfig.from_yaml(ROOT / "configs" / "bert.yaml")


def test_nlp_config_loads_bert_yaml(cfg):
    assert cfg.text_profile == "fact_offence"
    assert cfg.max_length == 512
    assert cfg.backbone == "distilbert-base-uncased"
    assert cfg.num_labels == 3
    assert cfg.training_seed == 42
    assert cfg.splits["train_seasons"] == [2019]
    assert cfg.splits["validation_season"] == 2025
    assert cfg.inputs["seasons"] == [2019, 2025]


def test_nlp_config_paths_resolve(cfg):
    assert cfg.path("models").name == "nlp"
    assert cfg.path("interim_docs") == ROOT / "data" / "interim" / "extracted_documents"
    assert cfg.meta_path_for_season(2019) == ROOT / "dataset" / "csv" / "raw_incidents_2019.meta.json"


def test_nlp_config_target_mapping_path(cfg):
    assert cfg.target_mapping_path == ROOT / "configs" / "target_mapping.yaml"


def test_nlp_config_rejects_invalid_profile():
    import yaml

    bad_path = ROOT / "tests" / "fixtures" / "bad_nlp_config.yaml"
    bad_path.write_text(
        yaml.dump(
            {
                "paths": {"models": "ml_models/nlp"},
                "text": {"profile": "invalid_profile"},
                "model": {"num_labels": 3},
            }
        ),
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="Invalid text.profile"):
            NlpTrainingConfig.from_yaml(bad_path)
    finally:
        bad_path.unlink(missing_ok=True)


def test_nlp_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        NlpTrainingConfig.from_yaml(ROOT / "configs" / "does_not_exist_bert.yaml")
