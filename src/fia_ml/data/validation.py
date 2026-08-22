"""Validate enriched dataset and export processed CSV + review queue."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import load_meta
from fia_ml.data.schema import MULTI_VALUE_COLUMNS, SCHEMA_COLUMNS
from fia_ml.paths import ensure_dir
from fia_ml.utils import secure_file_io as sio


REQUIRED_LABEL_COLUMNS = ["incident_id", "session", "drivers", "penalty"]


def _split_multi(value: str) -> list[str]:
    if not value or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        drivers = _split_multi(str(row.get("drivers", "")))
        out.at[idx, "num_drivers"] = str(len(drivers)) if drivers else str(row.get("num_drivers", ""))

        lap = row.get("lap", "")
        full_laps = row.get("full_laps", "")
        try:
            if lap and full_laps:
                lap_i = int(float(lap))
                full_i = int(float(full_laps))
                out.at[idx, "lap_remaining"] = str(max(full_i - lap_i, 0))
                out.at[idx, "completion_percentage"] = str(round((lap_i / full_i) * 100, 6))
        except ValueError:
            pass
    return out


def validate_schema(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [col for col in SCHEMA_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing schema columns: {missing}")
    return errors


def validate_rows(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if "incident_id" in df.columns and df["incident_id"].duplicated().any():
        dupes = df[df["incident_id"].duplicated()]["incident_id"].tolist()
        errors.append(f"Duplicate incident_id values: {dupes[:5]}")
    return errors


def validate_multi_value_alignment(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    groups = [
        {"drivers", "nationalities", "driver_standings", "driver_points", "years_in_sport", "superlicense_points_before_incident"},
        {"drivers", "respective_teams"},
        {"drivers", "construct_standings", "construct_points"},
    ]
    for idx, row in df.iterrows():
        for group in groups:
            lengths = {col: len(_split_multi(str(row.get(col, "")))) for col in group if col in df.columns}
            non_zero = [length for length in lengths.values() if length > 0]
            if non_zero and len(set(non_zero)) > 1:
                errors.append(f"Row {idx} incident {row.get('incident_id')} misaligned multi-value lengths: {lengths}")
                break
    return errors[:20]


def build_review_queue(df: pd.DataFrame, cfg: PipelineConfig, meta: dict[str, dict[str, Any]]) -> pd.DataFrame:
    min_conf = float(cfg.validation.get("min_parse_confidence", 0.7))
    flags: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        incident_id = row.get("incident_id", "")
        meta_row = meta.get(incident_id, {})
        reasons: list[str] = []
        if float(meta_row.get("parse_confidence", 1)) < min_conf:
            reasons.append("low_parse_confidence")
        if not str(row.get("severity", "")).strip():
            reasons.append("missing_severity")
        if str(row.get("session", "")).lower() == "race" and not str(row.get("lap", "")).strip():
            reasons.append("missing_lap")
        if not str(row.get("round", "")).strip():
            reasons.append("missing_round")
        if reasons:
            flags.append({**row.to_dict(), "review_reasons": "|".join(reasons)})
    return pd.DataFrame(flags)


def column_fill_rates(df: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            rates[col] = 0.0
            continue
        series = df[col].astype(str).str.strip()
        filled = ((series != "") & (series != "nan")).sum()
        rates[col] = round(float(filled) / max(len(df), 1), 4)
    return rates


def validate_and_export(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[Path, Path, dict[str, Any]]:
    df = compute_derived_fields(df)
    errors = validate_schema(df)
    errors.extend(validate_rows(df))
    errors.extend(validate_multi_value_alignment(df))

    meta = load_meta(cfg)
    review = build_review_queue(df, cfg, meta)

    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    processed = df[SCHEMA_COLUMNS]

    out_dir = ensure_dir(cfg.path("csv_out"))
    processed_path = out_dir / f"processed_{cfg.season}.csv"
    review_path = out_dir / f"review_queue_{cfg.season}.csv"
    processed.to_csv(processed_path, index=False)
    review.to_csv(review_path, index=False)

    quality = {
        "season": cfg.season,
        "row_count": len(processed),
        "review_count": len(review),
        "validation_errors": errors,
        "column_fill_rates": column_fill_rates(processed),
    }
    report_dir = ensure_dir(cfg.path("reports"))
    sio.write_json(report_dir / f"data_quality_{cfg.season}.json", quality)
    return processed_path, review_path, quality
