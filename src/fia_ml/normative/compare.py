"""Compare FIA actual vs normative outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from fia_ml.normative.config import NormativeConfig
from fia_ml.utils import secure_file_io as sio

SEVERITY_LABELS: tuple[int, ...] = (0, 1, 2)


def _label_columns(cfg: NormativeConfig) -> tuple[str, str]:
    comparison = cfg.comparison or {}
    fia_col = str(comparison.get("fia_label_column", "penalty_severity"))
    norm_col = str(comparison.get("normative_label_column", "normative_penalty_severity"))
    return fia_col, norm_col


def add_deviation_columns(df: pd.DataFrame, cfg: NormativeConfig) -> pd.DataFrame:
    """Add per-row agreement and deviation columns."""
    fia_col, norm_col = _label_columns(cfg)
    out = df.copy()
    fia = pd.to_numeric(out[fia_col], errors="coerce").fillna(0).astype(int)
    norm = pd.to_numeric(out[norm_col], errors="coerce").fillna(0).astype(int)
    out["agreement_fia_normative"] = fia == norm
    out["deviation_direction"] = norm - fia
    out["deviation_magnitude"] = out["deviation_direction"].abs()
    return out


def _severity_arrays(df: pd.DataFrame, cfg: NormativeConfig) -> tuple[np.ndarray, np.ndarray]:
    fia_col, norm_col = _label_columns(cfg)
    fia = pd.to_numeric(df[fia_col], errors="coerce").fillna(0).astype(int).to_numpy()
    norm = pd.to_numeric(df[norm_col], errors="coerce").fillna(0).astype(int).to_numpy()
    return fia, norm


def _confusion_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(SEVERITY_LABELS)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "row_label": "fia",
        "column_label": "normative",
    }


def _breakdown_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False):
        n = len(group)
        dev = group["deviation_direction"]
        manual_review_rate = 0.0
        if "normative_penalty_detail" in group.columns and n:
            manual_review_rate = float(
                (group["normative_penalty_detail"] == "manual_review").mean()
            )
        rows.append(
            {
                group_col: key,
                "n": n,
                "agreement_rate": float(group["agreement_fia_normative"].mean()) if n else 0.0,
                "disagreement_rate": 1.0 - float(group["agreement_fia_normative"].mean())
                if n
                else 0.0,
                "mean_deviation_direction": float(dev.mean()) if n else 0.0,
                "mean_deviation_magnitude": float(group["deviation_magnitude"].mean())
                if n
                else 0.0,
                "fia_harsher_rate": float((dev < 0).mean()) if n else 0.0,
                "normative_harsher_rate": float((dev > 0).mean()) if n else 0.0,
                "manual_review_rate": manual_review_rate,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("disagreement_rate", ascending=False).reset_index(drop=True)


def _aggregate_metrics(df: pd.DataFrame, cfg: NormativeConfig) -> dict[str, Any]:
    fia, norm = _severity_arrays(df, cfg)
    agreement_rate = float(df["agreement_fia_normative"].mean()) if len(df) else 0.0
    dev = df["deviation_direction"]
    manual_review_rate = 0.0
    if "normative_penalty_detail" in df.columns and len(df):
        manual_review_rate = float((df["normative_penalty_detail"] == "manual_review").mean())

    metrics: dict[str, Any] = {
        "row_count": len(df),
        "agreement_rate": agreement_rate,
        "disagreement_rate": 1.0 - agreement_rate,
        "cohens_kappa": float(cohen_kappa_score(fia, norm, labels=list(SEVERITY_LABELS)))
        if len(df)
        else 0.0,
        "mean_deviation_direction": float(dev.mean()) if len(df) else 0.0,
        "mean_deviation_magnitude": float(df["deviation_magnitude"].mean()) if len(df) else 0.0,
        "fia_harsher_rate": float((dev < 0).mean()) if len(df) else 0.0,
        "normative_harsher_rate": float((dev > 0).mean()) if len(df) else 0.0,
        "manual_review_rate": manual_review_rate,
        "confusion_matrix": _confusion_data(fia, norm),
    }

    matched = df
    if "normative_penalty_detail" in df.columns:
        matched = df[df["normative_penalty_detail"] != "manual_review"]
    if len(matched):
        fia_m, norm_m = _severity_arrays(matched, cfg)
        metrics["agreement_rate_excl_manual_review"] = float(
            matched["agreement_fia_normative"].mean()
        )
        metrics["cohens_kappa_excl_manual_review"] = float(
            cohen_kappa_score(fia_m, norm_m, labels=list(SEVERITY_LABELS))
        )
        metrics["matched_row_count"] = len(matched)
    else:
        metrics["agreement_rate_excl_manual_review"] = None
        metrics["cohens_kappa_excl_manual_review"] = None
        metrics["matched_row_count"] = 0

    return metrics


def _compare_ml(
    df: pd.DataFrame,
    cfg: NormativeConfig,
    ml_predictions_path: Path,
) -> dict[str, Any] | None:
    comparison_cfg = cfg.comparison or {}
    if not comparison_cfg.get("include_ml_comparison", True):
        return None

    raw = sio.read_json(ml_predictions_path)
    if not isinstance(raw, list) or not raw:
        return None

    ml_df = pd.DataFrame(raw)
    pred_col = "pred_xgboost"
    if pred_col not in ml_df.columns:
        for candidate in ("pred", "prediction", "penalty_severity_pred"):
            if candidate in ml_df.columns:
                pred_col = candidate
                break
        else:
            return None

    join_cols = ["row_id"] if "row_id" in ml_df.columns else ["incident_id"]
    merged = df.merge(ml_df[join_cols + [pred_col]], on=join_cols[0], how="inner")
    if merged.empty:
        return {
            "ml_predictions_path": str(ml_predictions_path),
            "overlap_rows": 0,
        }

    enriched = add_deviation_columns(merged, cfg)
    fia_col, norm_col = _label_columns(cfg)
    fia = pd.to_numeric(enriched[fia_col], errors="coerce").fillna(0).astype(int).to_numpy()
    norm = pd.to_numeric(enriched[norm_col], errors="coerce").fillna(0).astype(int).to_numpy()
    ml_pred = pd.to_numeric(enriched[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()

    return {
        "ml_predictions_path": str(ml_predictions_path),
        "prediction_column": pred_col,
        "overlap_rows": len(enriched),
        "agreement_fia_ml": float((fia == ml_pred).mean()),
        "agreement_normative_ml": float((norm == ml_pred).mean()),
        "agreement_fia_normative": float(enriched["agreement_fia_normative"].mean()),
        "cohens_kappa_fia_ml": float(cohen_kappa_score(fia, ml_pred, labels=list(SEVERITY_LABELS))),
        "cohens_kappa_normative_ml": float(
            cohen_kappa_score(norm, ml_pred, labels=list(SEVERITY_LABELS))
        ),
        "confusion_fia_vs_ml": _confusion_data(fia, ml_pred),
        "confusion_normative_vs_ml": _confusion_data(norm, ml_pred),
    }


def compare_outcomes(
    df: pd.DataFrame,
    cfg: NormativeConfig,
    *,
    ml_predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Compute agreement metrics, breakdown tables, and optional ML comparison."""
    fia_col, norm_col = _label_columns(cfg)
    missing = [col for col in (fia_col, norm_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required comparison columns: {missing}")

    enriched = add_deviation_columns(df, cfg)
    breakdown_dims = ("incident_type", "session", "circuit", "season")
    breakdowns = {
        dim: _breakdown_table(enriched, dim).to_dict(orient="records")
        for dim in breakdown_dims
        if dim in enriched.columns
    }

    result: dict[str, Any] = {
        "aggregate": _aggregate_metrics(enriched, cfg),
        "breakdowns": breakdowns,
        "top_deviations": _top_deviations(enriched),
    }

    if ml_predictions_path is not None and ml_predictions_path.exists():
        result["ml_comparison"] = _compare_ml(enriched, cfg, ml_predictions_path)

    return result


def _top_deviations(df: pd.DataFrame, *, limit: int = 15) -> list[dict[str, Any]]:
    disagreements = df[~df["agreement_fia_normative"]].copy()
    if disagreements.empty:
        return []

    sort_cols = [
        col for col in ("deviation_magnitude", "season", "round") if col in disagreements.columns
    ]
    ascending = [False] + [True] * (len(sort_cols) - 1)
    disagreements = disagreements.sort_values(sort_cols, ascending=ascending)
    columns = [
        col
        for col in (
            "row_id",
            "incident_id",
            "season",
            "round",
            "circuit",
            "session",
            "incident_type",
            "driver",
            "penalty",
            "penalty_severity",
            "normative_penalty_detail",
            "normative_penalty_severity",
            "normative_rule_id",
            "deviation_direction",
        )
        if col in disagreements.columns
    ]
    return disagreements[columns].head(limit).to_dict(orient="records")
