"""HuggingFace transformer classifier for penalty_severity from FIA text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class TextLabelDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_tokenizer(backbone: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(backbone)


def build_model(
    backbone: str,
    num_labels: int,
    *,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {"num_labels": num_labels}
    if id2label is not None and label2id is not None:
        kwargs["id2label"] = {int(k): v for k, v in id2label.items()}
        kwargs["label2id"] = {str(k): int(v) for k, v in label2id.items()}
    return AutoModelForSequenceClassification.from_pretrained(backbone, **kwargs)


@dataclass
class NlpClassifierTrainer:
    backbone: str
    num_labels: int = 3
    max_length: int = 512
    tokenizer: PreTrainedTokenizerBase | None = None
    model: PreTrainedModel | None = None

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        self.tokenizer = load_tokenizer(self.backbone)
        return self.tokenizer

    def build_model(self, label_names: dict[int, str] | None = None) -> PreTrainedModel:
        id2label = label_names
        label2id = {name: int(class_id) for class_id, name in (label_names or {}).items()}
        self.model = build_model(
            self.backbone,
            self.num_labels,
            id2label=id2label,
            label2id=label2id or None,
        )
        return self.model

    def make_dataset(self, texts: list[str], labels: list[int]) -> TextLabelDataset:
        if self.tokenizer is None:
            self.load_tokenizer()
        return TextLabelDataset(texts, labels, self.tokenizer, self.max_length)

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before predict")
        self.model.eval()
        preds: list[int] = []
        device = self.model.device
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoding = {key: value.to(device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = self.model(**encoding).logits
            preds.append(int(torch.argmax(logits, dim=-1).item()))
        return np.asarray(preds, dtype=int)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before predict_proba")
        self.model.eval()
        probs: list[list[float]] = []
        device = self.model.device
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoding = {key: value.to(device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = self.model(**encoding).logits
                prob = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            probs.append(prob.tolist())
        return np.asarray(probs, dtype=float)

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before save")
        model_dir = output_dir / "model"
        tokenizer_dir = output_dir / "tokenizer"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)

    def load(self, output_dir: Path) -> NlpClassifierTrainer:
        model_dir = output_dir / "model"
        tokenizer_dir = output_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return self
