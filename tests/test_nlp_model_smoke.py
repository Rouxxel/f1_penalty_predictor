import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")
pytest.importorskip("transformers")

from fia_ml.models.nlp_model import NlpClassifierTrainer
from fia_ml.nlp.dataset import NlpDatasetResult, build_nlp_dataset
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.training.train_nlp import train_nlp


@pytest.fixture
def fixture_cfg():
    return NlpTrainingConfig.from_yaml(ROOT / "tests" / "fixtures" / "nlp" / "nlp_dataset_test.yaml")


def test_nlp_classifier_predict_after_fit(fixture_cfg, tmp_path):
    dataset = build_nlp_dataset(fixture_cfg)
    cfg = fixture_cfg
    trainer = NlpClassifierTrainer(
        backbone="distilbert-base-uncased",
        num_labels=3,
        max_length=128,
    )
    trainer.load_tokenizer()
    trainer.build_model({0: "no_penalty", 1: "minor", 2: "major"})
    train_ds = trainer.make_dataset(
        dataset.train["text"].tolist(),
        dataset.train["penalty_severity"].astype(int).tolist(),
    )
    batch = [train_ds[0]]
    assert "input_ids" in batch[0]
    trainer.save(tmp_path)
    reloaded = NlpClassifierTrainer(backbone=cfg.backbone, num_labels=3, max_length=128).load(tmp_path)
    preds = reloaded.predict(dataset.validation["text"].tolist())
    assert preds.shape[0] == len(dataset.validation)


@pytest.mark.slow
def test_train_nlp_smoke(fixture_cfg):
    cfg = fixture_cfg
    fast_cfg = NlpTrainingConfig(
        paths={
            **cfg.paths,
            "models": "tests/fixtures/nlp/_out/train_smoke",
        },
        inputs=cfg.inputs,
        splits=cfg.splits,
        text={**cfg.text, "max_length": 128},
        model={**cfg.model, "backbone": "distilbert-base-uncased"},
        training={
            **cfg.training,
            "epochs": 1,
            "batch_size": 1,
            "early_stopping_patience": 1,
        },
        evaluation=cfg.evaluation,
        fusion=cfg.fusion,
    )
    dataset = build_nlp_dataset(fast_cfg)
    out_dir = ROOT / "tests" / "fixtures" / "nlp" / "_out" / "train_smoke"
    try:
        result = train_nlp(fast_cfg, dataset=dataset)
        assert Path(result["metrics_path"]).exists()
        assert (out_dir / "model").exists()
        assert (out_dir / "tokenizer").exists()
        assert not (out_dir / "checkpoints").exists()  # removed after a successful run
    finally:
        # A full DistilBERT save is ~1 GB with checkpoints; never leave it behind.
        shutil.rmtree(out_dir, ignore_errors=True)


def test_remove_checkpoints_only_deletes_checkpoints_dir(tmp_path):
    from fia_ml.training.train_nlp import _remove_checkpoints

    output_dir = tmp_path / "nlp"
    ckpt = output_dir / "checkpoints" / "checkpoint-116"
    ckpt.mkdir(parents=True)
    (ckpt / "optimizer.pt").write_bytes(b"x")
    model_file = output_dir / "model" / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"weights")

    # A path that isn't output_dir/checkpoints is refused.
    _remove_checkpoints(output_dir / "model", output_dir)
    assert model_file.exists()

    _remove_checkpoints(output_dir / "checkpoints", output_dir)
    assert not (output_dir / "checkpoints").exists()
    assert model_file.exists()
