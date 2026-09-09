"""Late fusion of NLP text probabilities with V1 tabular predictions."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fia_ml.nlp.dataset import NlpDatasetResult, build_nlp_dataset
from fia_ml.paths import DEFAULT_TRAINING_CONFIG, PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.target_mapping import load_target_mapping
from fia_ml.training.config import TrainingConfig
from fia_ml.training.data_loaders import feature_matrix, labels, load_train_val_frames
from fia_ml.training.metrics import compute_metrics
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.utils import secure_file_io as sio

PROBA_COLUMNS = ("proba_no_penalty", "proba_minor", "proba_major")


def _class_name_map(mapping_path: Path) -> dict[int, str]:
    mapping = load_target_mapping(mapping_path)
    return {int(class_id): cfg["name"] for class_id, cfg in mapping["classes"].items()}


def _proba_columns(label_names: dict[int, str]) -> list[str]:
    return [f"proba_{label_names[class_id]}" for class_id in sorted(label_names)]


def _records_to_frame(records: list[dict[str, Any]], *, proba_cols: list[str]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "row_id": str(record["row_id"]),
            "penalty_severity": int(record["penalty_severity"]),
        }
        for col in proba_cols:
            row[col] = float(record[col])
        rows.append(row)
    return pd.DataFrame(rows)


def merge_modalities(
    nlp_df: pd.DataFrame,
    tab_df: pd.DataFrame,
    *,
    proba_cols: list[str],
) -> pd.DataFrame:
    """Inner-join NLP and tabular validation rows on row_id."""
    nlp_cols = ["row_id", "penalty_severity"] + [f"nlp_{col}" for col in proba_cols]
    tab_cols = ["row_id"] + [f"tab_{col}" for col in proba_cols]
    nlp_renamed = nlp_df.rename(
        columns={col: f"nlp_{col}" for col in proba_cols},
    )[nlp_cols]
    tab_renamed = tab_df.rename(
        columns={col: f"tab_{col}" for col in proba_cols},
    )[tab_cols]
    merged = nlp_renamed.merge(tab_renamed, on="row_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping row_id between NLP and tabular validation predictions")
    severity_nlp = merged["penalty_severity"].astype(int)
    return merged.assign(penalty_severity=severity_nlp)


def _stack_proba(merged: pd.DataFrame, proba_cols: list[str], prefix: str) -> np.ndarray:
    return merged[[f"{prefix}_{col}" for col in proba_cols]].to_numpy(dtype=float)


def fuse_concat_logits(nlp_proba: np.ndarray, tab_proba: np.ndarray) -> np.ndarray:
    """Average class probabilities from both modalities."""
    fused = (nlp_proba + tab_proba) / 2.0
    return np.argmax(fused, axis=1).astype(int)


def fuse_stack_xgb(
    nlp_train: np.ndarray,
    tab_train: np.ndarray,
    y_train: np.ndarray,
    nlp_val: np.ndarray,
    tab_val: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a shallow XGBoost stacker on concatenated train probabilities."""
    x_train = np.hstack([nlp_train, tab_train])
    x_val = np.hstack([nlp_val, tab_val])
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, verbose=False)
    return model.predict(x_val).astype(int), model.predict_proba(x_val)


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "validation_metrics": metrics,
    }


def _load_tabular_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tabular predictions: {path}")
    payload = sio.read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def _load_nlp_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run --stage evaluate first to produce NLP predictions"
        )
    payload = sio.read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def _tabular_proba_from_model(
    tab_cfg: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate train/validation tabular probabilities from saved V1 XGBoost."""
    model_path = tab_cfg.model_dir() / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing V1 model at {model_path}")
    train_df, val_df = load_train_val_frames(tab_cfg)
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    label_names = _class_name_map(tab_cfg.target_mapping_path)
    proba_cols = _proba_columns(label_names)

    def _frame(df: pd.DataFrame) -> pd.DataFrame:
        proba = model.predict_proba(feature_matrix(df))
        out = pd.DataFrame({"row_id": df["row_id"].astype(str), "penalty_severity": labels(df)})
        for index, col in enumerate(proba_cols):
            out[col] = proba[:, index]
        return out

    return _frame(train_df), _frame(val_df)


def _nlp_proba_from_model(
    cfg: NlpTrainingConfig,
    dataset: NlpDatasetResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate train/validation NLP probabilities from saved transformer."""
    from fia_ml.models.nlp_model import NlpClassifierTrainer

    output_dir = cfg.model_dir()
    if not (output_dir / "model").exists():
        raise FileNotFoundError(f"Missing NLP model under {output_dir}")

    trainer = NlpClassifierTrainer(
        backbone=cfg.backbone,
        num_labels=cfg.num_labels,
        max_length=cfg.max_length,
    ).load(output_dir)

    label_names = _class_name_map(cfg.target_mapping_path)
    proba_cols = _proba_columns(label_names)

    def _frame(split_df: pd.DataFrame) -> pd.DataFrame:
        proba = trainer.predict_proba(split_df["text"].tolist())
        out = pd.DataFrame(
            {
                "row_id": split_df["row_id"].astype(str),
                "penalty_severity": split_df["penalty_severity"].astype(int),
            }
        )
        for index, col in enumerate(proba_cols):
            out[col] = proba[:, index]
        return out

    return _frame(dataset.train), _frame(dataset.validation)


def _write_fusion_report(
    cfg: NlpTrainingConfig,
    *,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    metrics = summary["metrics"]
    lines = [
        f"# NLP vs Tabular Fusion — {date.today().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Fusion method: `{summary['method']}`",
        f"- Merged validation rows: {summary['merged_validation_rows']}",
        f"- NLP validation rows: {summary['nlp_validation_rows']}",
        f"- Tabular validation rows: {summary['tabular_validation_rows']}",
        f"- Best single modality: **{summary['best_single_modality']}**",
        f"- Fusion beats best single modality: "
        f"**{'YES' if summary['beats_best_single_modality'] else 'NO'}**",
        "",
        "## Validation macro-F1 (merged rows)",
        "",
        "| Model | Macro-F1 | Accuracy |",
        "|-------|----------|----------|",
        f"| NLP text-only | {metrics['nlp_only']['macro_f1']:.3f} | "
        f"{metrics['nlp_only']['accuracy']:.3f} |",
        f"| V1 tabular-only | {metrics['tabular_only']['macro_f1']:.3f} | "
        f"{metrics['tabular_only']['accuracy']:.3f} |",
        f"| Fused ({summary['method']}) | {metrics['fused']['macro_f1']:.3f} | "
        f"{metrics['fused']['accuracy']:.3f} |",
        "",
        "## Notes",
        "",
        "- Metrics are computed on the inner join of NLP and tabular validation `row_id`s.",
        "- A negative fusion result is acceptable on the current two-season corpus.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_fusion_nlp(
    cfg: NlpTrainingConfig,
    *,
    dataset: NlpDatasetResult | None = None,
) -> dict[str, Any]:
    """Compare NLP-only, tabular-only, and fused validation performance."""
    fusion_cfg = cfg.fusion or {}
    method = str(fusion_cfg.get("method", "concat_logits"))
    if method not in {"concat_logits", "stack_xgb"}:
        raise ValueError(f"Unsupported fusion method: {method}")

    label_names = _class_name_map(cfg.target_mapping_path)
    proba_cols = _proba_columns(label_names)

    nlp_pred_path = cfg.model_dir() / "predictions_val.json"
    tab_rel = fusion_cfg.get("tabular_predictions", "ml_models/xgboost/predictions_val.json")
    tab_pred_path = Path(tab_rel)
    if not tab_pred_path.is_absolute():
        tab_pred_path = PROJECT_ROOT / tab_pred_path

    if method == "concat_logits":
        nlp_records = _load_nlp_predictions(nlp_pred_path)
        tab_records = _load_tabular_predictions(tab_pred_path)
        nlp_df = _records_to_frame(nlp_records, proba_cols=proba_cols)
        tab_df = _records_to_frame(tab_records, proba_cols=proba_cols)
        nlp_row_count = len(nlp_df)
        tab_row_count = len(tab_df)
        merged = merge_modalities(nlp_df, tab_df, proba_cols=proba_cols)
        nlp_proba = _stack_proba(merged, proba_cols, "nlp")
        tab_proba = _stack_proba(merged, proba_cols, "tab")
        y_true = merged["penalty_severity"].to_numpy(dtype=int)
        y_pred = fuse_concat_logits(nlp_proba, tab_proba)
        fused_proba = (nlp_proba + tab_proba) / 2.0
    else:
        if dataset is None:
            dataset = build_nlp_dataset(cfg)
        tab_cfg = TrainingConfig.from_yaml(DEFAULT_TRAINING_CONFIG)
        nlp_train_df, nlp_val_df = _nlp_proba_from_model(cfg, dataset)
        tab_train_df, tab_val_df = _tabular_proba_from_model(tab_cfg)
        nlp_row_count = len(nlp_val_df)
        tab_row_count = len(tab_val_df)
        train_merged = merge_modalities(nlp_train_df, tab_train_df, proba_cols=proba_cols)
        merged = merge_modalities(nlp_val_df, tab_val_df, proba_cols=proba_cols)
        nlp_proba = _stack_proba(merged, proba_cols, "nlp")
        tab_proba = _stack_proba(merged, proba_cols, "tab")
        y_true = merged["penalty_severity"].to_numpy(dtype=int)
        y_train = train_merged["penalty_severity"].to_numpy(dtype=int)
        y_pred, fused_proba = fuse_stack_xgb(
            _stack_proba(train_merged, proba_cols, "nlp"),
            _stack_proba(train_merged, proba_cols, "tab"),
            y_train,
            nlp_proba,
            tab_proba,
            seed=cfg.training_seed,
        )

    nlp_pred = np.argmax(nlp_proba, axis=1).astype(int)
    tab_pred = np.argmax(tab_proba, axis=1).astype(int)

    modality_metrics = {
        "nlp_only": _metrics_block(y_true, nlp_pred, nlp_proba),
        "tabular_only": _metrics_block(y_true, tab_pred, tab_proba),
        "fused": _metrics_block(y_true, y_pred, fused_proba),
    }
    best_single = max(
        ("nlp_only", modality_metrics["nlp_only"]["macro_f1"]),
        ("tabular_only", modality_metrics["tabular_only"]["macro_f1"]),
        key=lambda item: item[1],
    )
    beats_best = modality_metrics["fused"]["macro_f1"] >= best_single[1]

    fusion_predictions = []
    for index in range(len(merged)):
        row = merged.iloc[index]
        fusion_predictions.append(
            {
                "row_id": row["row_id"],
                "penalty_severity": int(row["penalty_severity"]),
                "pred_nlp": int(nlp_pred[index]),
                "pred_tabular": int(tab_pred[index]),
                "pred_fused": int(y_pred[index]),
            }
        )

    output_dir = ensure_dir(cfg.model_dir())
    metrics_payload = {
        "method": method,
        "merged_validation_rows": len(merged),
        "nlp_validation_rows": nlp_row_count,
        "tabular_validation_rows": tab_row_count,
        "metrics": {
            key: {
                "macro_f1": block["macro_f1"],
                "accuracy": block["accuracy"],
                "weighted_f1": block["weighted_f1"],
            }
            for key, block in modality_metrics.items()
        },
        "best_single_modality": best_single[0],
        "beats_best_single_modality": beats_best,
        "evaluated_at": date.today().isoformat(),
    }
    metrics_path = output_dir / "fusion_metrics.json"
    predictions_path = output_dir / "fusion_predictions_val.json"
    sio.write_json(metrics_path, metrics_payload)
    sio.write_json(predictions_path, fusion_predictions)

    report_dir = ensure_dir(cfg.path("reports") / "model_reports")
    report_path = report_dir / f"nlp_comparison_{date.today().isoformat()}.md"
    _write_fusion_report(cfg, summary=metrics_payload, output_path=report_path)

    return {
        "method": method,
        "macro_f1": modality_metrics["fused"]["macro_f1"],
        "beats_best_single_modality": beats_best,
        "best_single_modality": best_single[0],
        "outputs": {
            "fusion_metrics": str(metrics_path),
            "fusion_predictions_val": str(predictions_path),
            "report": str(report_path),
        },
    }
