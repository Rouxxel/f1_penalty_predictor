"""Batch apply normative rules to incidents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.escalation import add_escalation_columns
from fia_ml.normative.rule_engine import match_rule
from fia_ml.normative.rules_loader import LoadedRules
from fia_ml.paths import PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio

INTERIM_DOCUMENTS_ROOT = PROJECT_ROOT / "data" / "interim" / "extracted_documents"


def _iter_document_records(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _build_fact_lookup(seasons: list[int]) -> dict[str, str]:
    """Map incident_id to Fact text from parsed interim JSON when available."""
    lookup: dict[str, str] = {}
    for season in seasons:
        season_dir = INTERIM_DOCUMENTS_ROOT / str(season)
        if not season_dir.exists():
            continue
        for path in season_dir.glob("*.json"):
            for doc in _iter_document_records(sio.read_json(path)):
                incident_id = doc.get("incident_id")
                fields = doc.get("parsed_fields", {})
                if not isinstance(fields, dict):
                    fields = {}
                fact = fields.get("fact") or fields.get("Fact") or ""
                if incident_id and fact:
                    lookup[str(incident_id)] = str(fact)
    return lookup


def enrich_fact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `fact` text for condition matching without using outcome fields."""
    out = df.copy()
    seasons = sorted({int(season) for season in out["season"].dropna().unique()})
    lookup = _build_fact_lookup(seasons)

    if "fact" in out.columns:
        base_fact = out["fact"].fillna("").astype(str)
    else:
        base_fact = pd.Series("", index=out.index)

    mapped_fact = out["incident_id"].astype(str).map(lookup).fillna("")
    out["fact"] = (base_fact + " " + mapped_fact).str.strip()
    return out


def predict_normative(
    incidents: pd.DataFrame,
    rules: LoadedRules,
    cfg: NormativeConfig,
) -> pd.DataFrame:
    """Return incidents with normative_* outcome columns."""
    enriched = enrich_fact_columns(incidents)
    with_escalation = add_escalation_columns(enriched, cfg)

    normative_rows: list[dict[str, Any]] = []
    for _, row in with_escalation.iterrows():
        rule, outcome = match_rule(rules.document.rules, row.to_dict())
        normative_rows.append(
            {
                "normative_penalty_detail": outcome.penalty_detail,
                "normative_penalty_severity": int(outcome.penalty_severity),
                "normative_rule_id": rule.id,
                "normative_reason": rule.reason or "",
                "normative_cited_regulation": outcome.cited_regulation or "",
            }
        )

    normative_df = pd.DataFrame(normative_rows, index=with_escalation.index)
    return pd.concat([with_escalation, normative_df], axis=1)


def write_predictions_output(df: pd.DataFrame, output_path: Path) -> None:
    """Persist incidents_with_normative.parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def write_rules_version(rules: LoadedRules, models_dir: Path) -> Path:
    """Write rules version metadata for reproducibility."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "rules_version.json"
    sio.write_json(path, rules.to_version_metadata())
    return path


def write_predictions_json(df: pd.DataFrame, models_dir: Path) -> Path:
    """Write row_id → normative outcome mapping."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "predictions.json"
    records: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = str(row.get("row_id", row.get("incident_id")))
        records[key] = {
            "normative_penalty_detail": row["normative_penalty_detail"],
            "normative_penalty_severity": int(row["normative_penalty_severity"]),
            "normative_rule_id": row["normative_rule_id"],
            "normative_reason": row.get("normative_reason", ""),
        }
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path
