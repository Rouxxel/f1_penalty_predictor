"""Fine-tune HuggingFace text classifier for penalty_severity."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fia_ml.nlp.dataset import NlpDatasetResult, build_nlp_dataset, persist_nlp_dataset
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.metrics import compute_metrics
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.utils import secure_file_io as sio


def _label_names(mapping_path: Path) -> dict[int, str]:
    mapping = load_target_mapping(mapping_path)
    return {int(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _class_weight_tensor(y_train: np.ndarray, strategy: str, num_labels: int) -> torch.Tensor | None:
    if strategy != "inverse_frequency":
        return None
    counts = np.bincount(y_train.astype(int), minlength=num_labels)
    weights = len(y_train) / (num_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _snapshot_config(cfg: NlpTrainingConfig, output_dir: Path) -> Path:
    source = PROJECT_ROOT / "configs" / "bert.yaml"
    dest = output_dir / "training_config_snapshot.yaml"
    if source.exists():
        shutil.copy2(source, dest)
    else:
        sio.write_yaml(dest, {
            "paths": cfg.paths,
            "inputs": cfg.inputs,
            "splits": cfg.splits,
            "text": cfg.text,
            "model": cfg.model,
            "training": cfg.training,
            "evaluation": cfg.evaluation,
        })
    return dest


def prepare_nlp_data(cfg: NlpTrainingConfig) -> dict[str, Any]:
    result = build_nlp_dataset(cfg)
    outputs = persist_nlp_dataset(result, cfg)
    return {
        "train_rows": len(result.train),
        "validation_rows": len(result.validation),
        "audit": result.audit,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def train_nlp(
    cfg: NlpTrainingConfig,
    *,
    dataset: NlpDatasetResult | None = None,
) -> dict[str, Any]:
    """Fine-tune transformer on prepared NLP JSONL splits."""
    if dataset is None:
        dataset = build_nlp_dataset(cfg)

    if dataset.train.empty:
        raise ValueError("Training split is empty — run prepare stage or fix interim document joins")
    if dataset.validation.empty:
        raise ValueError("Validation split is empty — check splits and dataset joins")

    from fia_ml.models.nlp_model import NlpClassifierTrainer
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    class _WeightedTrainer(Trainer):
        def __init__(self, class_weights: torch.Tensor | None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    train_cfg = cfg.training
    label_names = _label_names(cfg.target_mapping_path)
    trainer_bundle = NlpClassifierTrainer(
        backbone=cfg.backbone,
        num_labels=cfg.num_labels,
        max_length=cfg.max_length,
    )
    trainer_bundle.load_tokenizer()
    trainer_bundle.build_model(label_names)

    y_train = dataset.train["penalty_severity"].astype(int).to_numpy()
    y_val = dataset.validation["penalty_severity"].astype(int).to_numpy()
    train_ds = trainer_bundle.make_dataset(
        dataset.train["text"].tolist(),
        y_train.tolist(),
    )
    val_ds = trainer_bundle.make_dataset(
        dataset.validation["text"].tolist(),
        y_val.tolist(),
    )

    output_dir = ensure_dir(cfg.model_dir())
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    class_weights = _class_weight_tensor(
        y_train,
        str(train_cfg.get("class_weight", "inverse_frequency")),
        cfg.num_labels,
    )

    freeze_epochs = int(cfg.model.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0 and trainer_bundle.model is not None:
        backbone = getattr(trainer_bundle.model, trainer_bundle.model.base_model_prefix, None)
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=float(train_cfg.get("epochs", 8)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(train_cfg.get("batch_size", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        seed=int(train_cfg.get("seed", 42)),
        fp16=bool(train_cfg.get("fp16", False)) and torch.cuda.is_available(),
        report_to=[],
    )

    patience = int(train_cfg.get("early_stopping_patience", 3))
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    hf_trainer = _WeightedTrainer(
        class_weights,
        model=trainer_bundle.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
    )
    hf_trainer.train()

    trainer_bundle.model = hf_trainer.model
    trainer_bundle.save(output_dir)
    _snapshot_config(cfg, output_dir)

    y_pred = trainer_bundle.predict(dataset.validation["text"].tolist())
    y_proba = trainer_bundle.predict_proba(dataset.validation["text"].tolist())
    val_metrics = compute_metrics(y_val, y_pred, y_proba=y_proba)

    metrics = {
        "backbone": cfg.backbone,
        "text_profile": cfg.text_profile,
        "train_rows": len(dataset.train),
        "validation_rows": len(dataset.validation),
        "validation_metrics": val_metrics,
        "macro_f1": val_metrics["macro_f1"],
        "best_checkpoint": str(checkpoints_dir),
    }
    sio.write_json(output_dir / "metrics.json", metrics)

    return {
        "model_dir": str(output_dir),
        "metrics": metrics,
        "metrics_path": str(output_dir / "metrics.json"),
    }
