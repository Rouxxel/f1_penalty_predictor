"""Evaluation plots, error analysis, and training reports."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.leakage_filter import assert_no_leakage
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.config import TrainingConfig
from fia_ml.training.data_loaders import feature_matrix, labels
from fia_ml.training.metrics import compute_metrics
from fia_ml.utils import secure_file_io as sio


def audit_leakage(feature_columns: list[str]) -> dict[str, Any]:
    try:
        assert_no_leakage(feature_columns)
        return {"passed": True, "forbidden_in_features": []}
    except ValueError as exc:
        return {"passed": False, "error": str(exc)}


def check_domain_sanity(
    importance_gain: dict[str, float],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    ranked = sorted(importance_gain.items(), key=lambda item: item[1], reverse=True)
    top_features = [name for name, _ in ranked[:top_n]]
    checks = {
        "incident_type_in_top10": any("incident_type" in name for name in top_features),
        "severity_in_top10": any("severity" in name for name in top_features),
    }
    checks["passed"] = checks["incident_type_in_top10"] or checks["severity_in_top10"]
    checks["top_features"] = top_features
    return checks


def plot_confusion_matrix(
    confusion: list[list[int]],
    labels: list[int],
    label_names: dict[str, str],
    output_path: Path,
    *,
    title: str,
) -> None:
    display_labels = [label_names.get(str(label), str(label)) for label in labels]
    matrix = np.array(confusion)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    importance_gain: dict[str, float],
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    ranked = sorted(importance_gain.items(), key=lambda item: item[1], reverse=True)[:top_n]
    if not ranked:
        return
    names = [name.replace("cat__", "").replace("num__", "") for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} feature importance (gain)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_error_analysis(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for row in predictions:
        actual = int(row["penalty_severity"])
        predicted = int(row["pred_xgboost"])
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


def _label_names(mapping_path: Path) -> dict[str, str]:
    mapping = load_target_mapping(mapping_path)
    return {str(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _write_report(
    cfg: TrainingConfig,
    *,
    xgb_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    leakage_audit: dict[str, Any],
    domain_sanity: dict[str, Any],
    error_count: int,
    output_path: Path,
    figures: dict[str, str],
) -> None:
    val = xgb_metrics["validation_metrics"]
    lines = [
        f"# V1 Training Report — {date.today().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Train seasons: {cfg.splits.get('train_seasons')}",
        f"- Validation season: {cfg.splits.get('validation_season')}",
        f"- Test season: {cfg.splits.get('test_season')}",
        f"- Random seed: {cfg.random_state}",
        "",
        "## Data counts",
        "",
        f"- Train rows: {xgb_metrics.get('train_rows', 'n/a')}",
        f"- Validation rows: {xgb_metrics.get('validation_rows', 'n/a')}",
        "",
        "## Model comparison (validation macro-F1)",
        "",
        "| Model | Macro-F1 |",
        "|-------|----------|",
    ]
    if baseline_metrics:
        lines.append(
            f"| Majority baseline | {baseline_metrics['majority_class']['metrics']['macro_f1']:.3f} |"
        )
        lines.append(
            f"| Session-stratified baseline | "
            f"{baseline_metrics['session_stratified']['metrics']['macro_f1']:.3f} |"
        )
    lines.append(f"| XGBoost | {xgb_metrics['macro_f1']:.3f} |")
    lines.append("")
    lines.append("## XGBoost validation metrics")
    lines.append("")
    lines.append(f"- Accuracy: {val['accuracy']:.3f}")
    lines.append(f"- Macro-F1: {val['macro_f1']:.3f}")
    lines.append(f"- Weighted F1: {val['weighted_f1']:.3f}")
    lines.append(f"- Log loss: {val.get('log_loss', 'n/a')}")
    lines.append(f"- Best iteration: {xgb_metrics.get('best_iteration', 'n/a')}")
    lines.append("")
    lines.append("### Per-class")
    lines.append("")
    for class_id, stats in val["per_class"].items():
        lines.append(
            f"- Class {class_id}: precision={stats['precision']:.3f}, "
            f"recall={stats['recall']:.3f}, f1={stats['f1']:.3f}, support={stats['support']}"
        )
    lines.append("")
    lines.append("## Quality checks")
    lines.append("")
    lines.append(f"- Leakage audit: {'PASS' if leakage_audit['passed'] else 'FAIL'}")
    lines.append(f"- Domain sanity (incident_type/severity in top-10): "
                 f"{'PASS' if domain_sanity['passed'] else 'FAIL'}")
    if domain_sanity.get("top_features"):
        lines.append(f"- Top features: {', '.join(domain_sanity['top_features'])}")
    lines.append(f"- Misclassified validation rows: {error_count}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, path in figures.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    success_macro = val["macro_f1"] > cfg.evaluation.get("min_macro_f1", 0.40)
    lines.append("## Success criteria")
    lines.append("")
    lines.append(f"- Macro-F1 > {cfg.evaluation.get('min_macro_f1', 0.40)}: "
                 f"{'YES' if success_macro else 'NO'}")
    lines.append(
        f"- Beats session baseline: "
        f"{'YES' if xgb_metrics.get('beats_session_baseline') else 'NO'}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_test_split(cfg: TrainingConfig, label_names: dict[str, str]) -> dict[str, Any] | None:
    features_cfg = cfg.features or {}
    test_name = features_cfg.get("test_file", "test.parquet")
    test_path = cfg.path("processed") / test_name
    model_path = cfg.model_dir() / "model.json"
    if not test_path.exists() or not model_path.exists():
        return None

    test_df = pd.read_parquet(test_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    X_test = feature_matrix(test_df)
    y_test = labels(test_df).to_numpy()
    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba=y_proba)

    figures_dir = ensure_dir(cfg.path("reports") / "figures")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["labels"],
        label_names,
        figures_dir / "confusion_matrix_test.png",
        title="XGBoost — test split",
    )
    return {"rows": len(test_df), "metrics": metrics}


def run_evaluation(cfg: TrainingConfig) -> dict[str, Any]:
    """Generate plots, error analysis, and markdown report from saved model artifacts."""
    models_dir = cfg.path("models")
    xgb_dir = cfg.model_dir()
    metrics_path = xgb_dir / "metrics.json"
    predictions_path = xgb_dir / "predictions_val.json"
    importance_path = xgb_dir / "feature_importance.json"
    preprocessor_meta_path = cfg.preprocessor_path().with_suffix(".meta.json")

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path} — run --stage train first"
        )

    xgb_metrics = sio.read_json(metrics_path)
    predictions = sio.read_json(predictions_path) if predictions_path.exists() else []
    importance = sio.read_json(importance_path) if importance_path.exists() else {"gain": {}}

    baseline_path = models_dir / "baseline" / "metrics.json"
    baseline_metrics = sio.read_json(baseline_path) if baseline_path.exists() else None

    feature_columns = []
    if preprocessor_meta_path.exists():
        feature_columns = sio.read_json(preprocessor_meta_path).get("feature_columns", [])
    leakage_audit = audit_leakage(feature_columns)
    domain_sanity = check_domain_sanity(importance.get("gain", {}))

    label_names = _label_names(cfg.target_mapping_path)
    val_metrics = xgb_metrics["validation_metrics"]

    figures_dir = ensure_dir(cfg.path("reports") / "figures")
    confusion_val_path = figures_dir / "confusion_matrix_val.png"
    importance_fig_path = figures_dir / "feature_importance_top20.png"

    plot_confusion_matrix(
        val_metrics["confusion_matrix"],
        val_metrics["labels"],
        label_names,
        confusion_val_path,
        title="XGBoost — validation split",
    )
    plot_feature_importance(importance.get("gain", {}), importance_fig_path)

    errors = build_error_analysis(predictions)
    errors_path = xgb_dir / "error_analysis_val.json"
    sio.write_json(errors_path, errors)

    report_dir = ensure_dir(cfg.path("reports") / "model_reports")
    report_prefix = (
        "v2_feature_engineering_report"
        if cfg.feature_version == "v2"
        else "v1_training_report"
    )
    report_path = report_dir / f"{report_prefix}_{date.today().isoformat()}.md"
    figures = {
        "confusion_matrix_val": str(confusion_val_path.relative_to(PROJECT_ROOT)),
        "feature_importance_top20": str(importance_fig_path.relative_to(PROJECT_ROOT)),
    }

    test_eval = _evaluate_test_split(cfg, label_names)
    if test_eval:
        figures["confusion_matrix_test"] = "reports/figures/confusion_matrix_test.png"

    _write_report(
        cfg,
        xgb_metrics=xgb_metrics,
        baseline_metrics=baseline_metrics,
        leakage_audit=leakage_audit,
        domain_sanity=domain_sanity,
        error_count=len(errors),
        output_path=report_path,
        figures=figures,
    )

    return {
        "macro_f1": xgb_metrics["macro_f1"],
        "leakage_audit_passed": leakage_audit["passed"],
        "domain_sanity_passed": domain_sanity["passed"],
        "misclassified_validation_rows": len(errors),
        "test_evaluation": test_eval,
        "outputs": {
            "confusion_matrix_val": str(confusion_val_path),
            "feature_importance": str(importance_fig_path),
            "error_analysis": str(errors_path),
            "report": str(report_path),
        },
    }
