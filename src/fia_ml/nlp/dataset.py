"""Build NLP training datasets from processed CSVs and interim FIA documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.ingestion import load_processed_seasons
from fia_ml.nlp.audit import bump_total, finalize_audit, new_audit, season_bucket
from fia_ml.nlp.labels import add_labels, filter_trainable_rows
from fia_ml.nlp.text_builder import build_text
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.preprocessing.encoding import TARGET_COLUMN
from fia_ml.preprocessing.flatten import flatten_incidents, split_multi_value
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.utils import secure_file_io as sio


@dataclass(frozen=True)
class NlpDatasetResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame | None
    audit: dict[str, Any]


def load_meta_index(cfg: NlpTrainingConfig, season: int) -> dict[str, dict[str, Any]]:
    path = cfg.meta_path_for_season(season)
    if not path.exists():
        return {}
    payload = sio.read_json(path)
    if not isinstance(payload, list):
        return {}
    return {
        str(row["incident_id"]): row
        for row in payload
        if isinstance(row, dict) and row.get("incident_id")
    }


def load_interim_document(
    cfg: NlpTrainingConfig,
    season: int,
    document_id: str,
) -> dict[str, Any]:
    if not document_id:
        return {}
    path = cfg.path("interim_docs") / str(season) / f"{document_id}.json"
    if not path.exists():
        return {}
    payload = sio.read_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_document_for_incident(
    incident_id: str,
    season: int,
    cfg: NlpTrainingConfig,
    *,
    meta_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve interim extracted-document JSON for an incident_id."""
    meta = meta_index if meta_index is not None else load_meta_index(cfg, season)
    meta_row = meta.get(incident_id, {})
    document_id = str(meta_row.get("document_id", "")).strip()
    if not document_id:
        return {}
    return load_interim_document(cfg, season, document_id)


def _temporal_split_nlp(
    df: pd.DataFrame,
    splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Season split for NLP rows; allows empty train when join coverage is partial."""
    if df.empty:
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty, None

    train_seasons = {int(s) for s in splits.get("train_seasons", [])}
    val_season = splits.get("validation_season")
    test_season = splits.get("test_season")
    if val_season is not None:
        val_season = int(val_season)
    if test_season is not None:
        test_season = int(test_season)

    train_df = df[df["season"].astype(int).isin(train_seasons)].copy()
    val_df = df[df["season"].astype(int) == val_season].copy() if val_season is not None else pd.DataFrame()
    test_df = df[df["season"].astype(int) == test_season].copy() if test_season is not None else None
    return train_df, val_df, test_df


def _count_misaligned_skips(trainable: pd.DataFrame, flattened: pd.DataFrame) -> int:
    trainable_ids = set(trainable["incident_id"].astype(str))
    flat_ids = set(flattened["incident_id"].astype(str))
    skipped = trainable_ids - flat_ids
    no_driver = {
        str(row["incident_id"])
        for _, row in trainable.iterrows()
        if not split_multi_value(row.get("drivers", ""))
    }
    return len(skipped - no_driver)


def _build_rows_for_season(
    season_df: pd.DataFrame,
    cfg: NlpTrainingConfig,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    if season_df.empty:
        return []

    season = int(season_df["season"].iloc[0])
    bucket = season_bucket(audit, season)
    bucket["labeled_incident_rows"] = len(season_df)
    bump_total(audit, "labeled_incident_rows", len(season_df))

    flattened = flatten_incidents(season_df)
    misaligned = _count_misaligned_skips(season_df, flattened)
    bucket["misaligned_skipped"] = misaligned
    bump_total(audit, "misaligned_skipped", misaligned)

    meta_index = load_meta_index(cfg, season)
    include_metadata = bool(cfg.text.get("include_metadata", True))
    profile = cfg.text_profile

    rows: list[dict[str, Any]] = []
    for _, flat_row in flattened.iterrows():
        incident_id = str(flat_row["incident_id"])
        meta_row = meta_index.get(incident_id)
        if not meta_row:
            bucket["missing_meta"] += 1
            bump_total(audit, "missing_meta")
            continue

        document_id = str(meta_row.get("document_id", "")).strip()
        if not document_id:
            bucket["missing_document_id"] += 1
            bump_total(audit, "missing_document_id")
            continue

        doc = load_interim_document(cfg, season, document_id)
        if not doc:
            bucket["missing_interim_doc"] += 1
            bucket["missing_interim_incident_ids"].append(incident_id)
            bump_total(audit, "missing_interim_doc")
            continue

        try:
            text = build_text(
                doc,
                profile=profile,
                include_metadata=include_metadata,
            )
        except ValueError:
            bucket["text_build_errors"] += 1
            bucket["text_build_error_incident_ids"].append(incident_id)
            bump_total(audit, "text_build_errors")
            continue

        if not text.strip():
            bucket["text_build_errors"] += 1
            bucket["text_build_error_incident_ids"].append(incident_id)
            bump_total(audit, "text_build_errors")
            continue

        bucket["rows_with_text"] += 1
        bump_total(audit, "rows_with_text")
        rows.append(
            {
                "row_id": flat_row.get("row_id", f"{incident_id}_{flat_row.get('driver', '')}"),
                "incident_id": incident_id,
                "season": season,
                "round": flat_row.get("round"),
                "session": flat_row.get("session", ""),
                "driver": flat_row.get("driver", ""),
                "penalty": flat_row.get("penalty", ""),
                "penalty_severity": int(flat_row[TARGET_COLUMN]),
                "document_id": document_id,
                "parse_confidence": meta_row.get("parse_confidence", ""),
                "text": text,
            }
        )

    bucket["flattened_rows"] = len(flattened)
    return rows


def build_nlp_dataset(cfg: NlpTrainingConfig) -> NlpDatasetResult:
    """Join processed CSVs, labels, interim docs, and text; split by season."""
    inputs = cfg.inputs or {}
    glob_pattern = str(cfg.paths.get("csv_input_glob", "dataset/csv/processed_*.csv"))
    paths = sorted(PROJECT_ROOT.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {glob_pattern}")
    seasons = inputs.get("seasons")
    if seasons:
        allowed = {f"processed_{int(s)}.csv" for s in seasons}
        paths = [path for path in paths if path.name in allowed]
        missing = [f"processed_{int(s)}.csv" for s in seasons if f"processed_{int(s)}.csv" not in {p.name for p in paths}]
        if missing:
            raise FileNotFoundError(f"Missing processed CSV files for seasons: {missing}")
    raw = load_processed_seasons(paths)
    labeled = add_labels(raw, cfg.target_mapping_path)
    trainable = filter_trainable_rows(labeled)

    audit = new_audit(cfg.text_profile)
    all_rows: list[dict[str, Any]] = []
    for season in sorted(trainable["season"].dropna().astype(int).unique()):
        season_df = trainable[trainable["season"].astype(int) == int(season)].copy()
        all_rows.extend(_build_rows_for_season(season_df, cfg, audit))

    finalize_audit(audit)

    if not all_rows:
        empty = pd.DataFrame(
            columns=[
                "row_id",
                "incident_id",
                "season",
                "round",
                "session",
                "driver",
                "penalty",
                "penalty_severity",
                "document_id",
                "parse_confidence",
                "text",
            ]
        )
        return NlpDatasetResult(
            train=empty,
            validation=empty,
            test=None,
            audit=audit,
        )

    dataset = pd.DataFrame(all_rows)
    train_df, val_df, test_df = _temporal_split_nlp(dataset, cfg.splits)
    return NlpDatasetResult(
        train=train_df,
        validation=val_df,
        test=test_df,
        audit=audit,
    )


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def persist_nlp_dataset(result: NlpDatasetResult, cfg: NlpTrainingConfig) -> dict[str, Path]:
    """Write JSONL splits and audit JSON."""
    processed_dir = ensure_dir(cfg.path("processed"))
    models_dir = ensure_dir(cfg.path("models"))

    outputs = {
        "train_jsonl": processed_dir / "nlp_train.jsonl",
        "validation_jsonl": processed_dir / "nlp_validation.jsonl",
        "audit_json": models_dir / "nlp_dataset_audit.json",
    }

    _write_jsonl(result.train, outputs["train_jsonl"])
    _write_jsonl(result.validation, outputs["validation_jsonl"])
    if result.test is not None and not result.test.empty:
        test_path = processed_dir / "nlp_test.jsonl"
        _write_jsonl(result.test, test_path)
        outputs["test_jsonl"] = test_path

    sio.write_json(outputs["audit_json"], result.audit)
    return outputs
