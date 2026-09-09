"""NLP model evaluation, predictions export, and training reports."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fia_ml.nlp.dataset import NlpDatasetResult, build_nlp_dataset
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.metrics import compute_metrics
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.utils import secure_file_io as sio


def _label_names(mapping_path: Path) -> dict[str, str]:
    mapping = load_target_mapping(mapping_path)
    return {str(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _class_name_map(mapping_path: Path) -> dict[int, str]:
    mapping = load_target_mapping(mapping_path)
    return {int(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def build_nlp_predictions(
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    label_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Build validation prediction records for JSON export."""
    records: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(val_df.iterrows()):
        record: dict[str, Any] = {
            "row_id": row.get("row_id"),
            "incident_id": row.get("incident_id"),
            "penalty": row.get("penalty"),
            "session": row.get("session"),
            "penalty_severity": int(row["penalty_severity"]),
            "pred_nlp": int(y_pred[index]),
        }
        for class_id in sorted(label_names):
            name = label_names[class_id]
            record[f"proba_{name}"] = float(y_proba[index, class_id])
        records.append(record)
    return records


def build_error_analysis(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for row in predictions:
        actual = int(row["penalty_severity"])
        predicted = int(row["pred_nlp"])
        if actual != predicted:
            errors.append(
                {
                    "incident_id": row.get("incident_id"),
                    "row_id": row.get("row_id"),
                    "penalty": row.get("penalty"),
                    "session": row.get("session"),
                    "actual_severity": actual,
                    "predicted_severity": predicted,
                }
            )
    return errors


def _load_comparison_metrics(cfg: NlpTrainingConfig) -> dict[str, Any] | None:
    rel = cfg.evaluation.get("compare_to_v1")
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    return sio.read_json(path)


def _write_nlp_report(
    cfg: NlpTrainingConfig,
    *,
    nlp_metrics: dict[str, Any],
    v1_metrics: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    error_count: int,
    output_path: Path,
    figures: dict[str, str],
) -> None:
    val = nlp_metrics["validation_metrics"]
    nlp_f1 = float(nlp_metrics.get("macro_f1", val["macro_f1"]))
    v1_f1 = float(v1_metrics["macro_f1"]) if v1_metrics else None
    min_f1 = float(cfg.evaluation.get("min_macro_f1", 0.35))

    lines = [
        f"# NLP Training Report — {date.today().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Backbone: `{nlp_metrics.get('backbone', cfg.backbone)}`",
        f"- Text profile: `{nlp_metrics.get('text_profile', cfg.text_profile)}`",
        f"- Validation macro-F1: **{nlp_f1:.3f}**",
    ]
    if v1_f1 is not None:
        delta = nlp_f1 - v1_f1
        lines.append(
            f"- V1 tabular macro-F1: **{v1_f1:.3f}** ({delta:+.3f} vs NLP)"
        )
    lines.extend([
        "",
        "## Configuration",
        "",
        f"- Train seasons: {cfg.splits.get('train_seasons')}",
        f"- Validation season: {cfg.splits.get('validation_season')}",
        f"- Max sequence length: {cfg.max_length}",
        f"- Random seed: {cfg.training_seed}",
        "",
        "## Data counts",
        "",
        f"- Train rows: {nlp_metrics.get('train_rows', 'n/a')}",
        f"- Validation rows: {nlp_metrics.get('validation_rows', 'n/a')}",
        "",
        "## Model comparison (validation macro-F1)",
        "",
        "| Model | Macro-F1 |",
        "|-------|----------|",
    ])
    if v1_f1 is not None:
        lines.append(f"| V1 XGBoost (tabular) | {v1_f1:.3f} |")
    lines.append(f"| NLP ({cfg.backbone}) | {nlp_f1:.3f} |")
    lines.append("")
    lines.append("## NLP validation metrics")
    lines.append("")
    lines.append(f"- Accuracy: {val['accuracy']:.3f}")
    lines.append(f"- Macro-F1: {val['macro_f1']:.3f}")
    lines.append(f"- Weighted F1: {val['weighted_f1']:.3f}")
    if "log_loss" in val:
        lines.append(f"- Log loss: {val['log_loss']:.3f}")
    lines.append("")
    lines.append("### Per-class")
    lines.append("")
    for class_id, stats in val["per_class"].items():
        lines.append(
            f"- Class {class_id}: precision={stats['precision']:.3f}, "
            f"recall={stats['recall']:.3f}, f1={stats['f1']:.3f}, support={stats['support']}"
        )
    lines.append("")

    if audit:
        totals = audit.get("totals", {})
        lines.extend([
            "## Dataset audit",
            "",
            f"- Labeled incident rows: {totals.get('labeled_incident_rows', 'n/a')}",
            f"- Rows with text: {totals.get('rows_with_text', 'n/a')}",
            f"- Missing interim docs: {totals.get('missing_interim_doc', 'n/a')}",
            f"- Text build errors: {totals.get('text_build_errors', 'n/a')}",
            "",
        ])

    lines.extend([
        "## Quality checks",
        "",
        f"- Misclassified validation rows: {error_count}",
        f"- Text profile leakage-safe: {'YES' if cfg.text_profile != 'leaky_full' else 'NO (debug profile)'}",
        "",
        "## Figures",
        "",
    ])
    for name, path in figures.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    lines.append("## Success criteria")
    lines.append("")
    lines.append(f"- Macro-F1 ≥ {min_f1:.2f}: {'YES' if nlp_f1 >= min_f1 else 'NO'}")
    if v1_f1 is not None:
        beats_v1 = nlp_f1 >= v1_f1
        lines.append(
            f"- NLP macro-F1 ≥ V1 tabular: {'YES' if beats_v1 else 'NO (acceptable on tiny train set)'}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_nlp(
    cfg: NlpTrainingConfig,
    *,
    dataset: NlpDatasetResult | None = None,
) -> dict[str, Any]:
    """Evaluate saved NLP model on validation split; write predictions and report."""
    output_dir = cfg.model_dir()
    model_path = output_dir / "model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path} — run --stage train first"
        )

    if dataset is None:
        dataset = build_nlp_dataset(cfg)
    if dataset.validation.empty:
        raise ValueError("Validation split is empty — check dataset joins and splits")

    from fia_ml.models.nlp_model import NlpClassifierTrainer

    trainer = NlpClassifierTrainer(
        backbone=cfg.backbone,
        num_labels=cfg.num_labels,
        max_length=cfg.max_length,
    ).load(output_dir)

    val_df = dataset.validation
    texts = val_df["text"].tolist()
    y_val = val_df["penalty_severity"].astype(int).to_numpy()
    y_pred = trainer.predict(texts)
    y_proba = trainer.predict_proba(texts)

    class_names = _class_name_map(cfg.target_mapping_path)
    val_metrics = compute_metrics(y_val, y_pred, y_proba=y_proba)
    predictions = build_nlp_predictions(val_df, y_pred, y_proba, label_names=class_names)
    errors = build_error_analysis(predictions)

    metrics_path = output_dir / "metrics.json"
    existing = sio.read_json(metrics_path) if metrics_path.exists() else {}
    nlp_metrics = {
        **existing,
        "backbone": cfg.backbone,
        "text_profile": cfg.text_profile,
        "train_rows": len(dataset.train),
        "validation_rows": len(dataset.validation),
        "validation_metrics": val_metrics,
        "macro_f1": val_metrics["macro_f1"],
        "evaluated_at": date.today().isoformat(),
    }
    sio.write_json(metrics_path, nlp_metrics)

    predictions_path = output_dir / "predictions_val.json"
    sio.write_json(predictions_path, predictions)
    errors_path = output_dir / "error_analysis_val.json"
    sio.write_json(errors_path, errors)

    from fia_ml.training.evaluate import plot_confusion_matrix

    label_display = _label_names(cfg.target_mapping_path)
    figures_dir = ensure_dir(cfg.path("reports") / "figures")
    confusion_path = figures_dir / "nlp_confusion_matrix_val.png"
    plot_confusion_matrix(
        val_metrics["confusion_matrix"],
        val_metrics["labels"],
        label_display,
        confusion_path,
        title=f"NLP ({cfg.backbone}) — validation split",
    )

    audit_path = output_dir / "nlp_dataset_audit.json"
    audit = sio.read_json(audit_path) if audit_path.exists() else dataset.audit
    v1_metrics = _load_comparison_metrics(cfg)

    report_dir = ensure_dir(cfg.path("reports") / "model_reports")
    report_path = report_dir / f"nlp_training_report_{date.today().isoformat()}.md"
    figures = {
        "confusion_matrix_val": str(confusion_path.relative_to(PROJECT_ROOT)),
    }
    _write_nlp_report(
        cfg,
        nlp_metrics=nlp_metrics,
        v1_metrics=v1_metrics,
        audit=audit,
        error_count=len(errors),
        output_path=report_path,
        figures=figures,
    )

    result: dict[str, Any] = {
        "macro_f1": val_metrics["macro_f1"],
        "misclassified_validation_rows": len(errors),
        "beats_v1": (
            v1_metrics is not None
            and val_metrics["macro_f1"] >= float(v1_metrics["macro_f1"])
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "predictions_val": str(predictions_path),
            "error_analysis": str(errors_path),
            "confusion_matrix_val": str(confusion_path),
            "report": str(report_path),
        },
    }
    if v1_metrics:
        result["v1_macro_f1"] = v1_metrics["macro_f1"]
    return result
