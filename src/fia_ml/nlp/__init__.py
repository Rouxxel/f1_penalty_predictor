"""NLP dataset and text utilities for spec V2 penalty classification."""

from fia_ml.nlp.dataset import (
    NlpDatasetResult,
    build_nlp_dataset,
    persist_nlp_dataset,
    resolve_document_for_incident,
)
from fia_ml.nlp.labels import add_labels, filter_trainable_rows
from fia_ml.nlp.text_builder import build_text, leakage_violations, truncate_text
from fia_ml.training.nlp_config import NlpTrainingConfig

__all__ = [
    "NlpTrainingConfig",
    "NlpDatasetResult",
    "add_labels",
    "build_nlp_dataset",
    "build_text",
    "filter_trainable_rows",
    "leakage_violations",
    "persist_nlp_dataset",
    "resolve_document_for_incident",
    "truncate_text",
]
