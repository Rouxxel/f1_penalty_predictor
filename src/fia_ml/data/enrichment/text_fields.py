"""Rule-based extraction for driver_at_fault and severity suggestions."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta
from fia_ml.data.enrichment.provenance import EnrichmentProvenance
from fia_ml.data.enrichment.timestamp import save_meta_rows
from fia_ml.data.parsing import normalize_text, parse_driver_at_fault
from fia_ml.utils import secure_file_io as sio


def load_interim_document(cfg: PipelineConfig, document_id: str) -> dict[str, Any]:
    if not document_id:
        return {}
    path = cfg.path("interim_docs") / str(cfg.season) / f"{document_id}.json"
    if not path.exists():
        return {}
    payload = sio.read_json(path)
    return payload if isinstance(payload, dict) else {}


def _document_text(doc: dict[str, Any]) -> tuple[str, str, str]:
    fields = doc.get("parsed_fields", {})
    return (
        str(fields.get("fact", "")),
        str(fields.get("reason", "")),
        str(fields.get("decision", "")),
    )


def infer_driver_at_fault(
    fact: str,
    reason: str,
    decision: str,
    drivers: list[str],
) -> tuple[str, float]:
    combined = normalize_text(f"{fact} {reason} {decision}").lower()
    if not combined.strip():
        return "", 0.0

    parsed = parse_driver_at_fault(reason, fact)
    if parsed == "none":
        return "none", 0.88
    if parsed.startswith("car_"):
        return parsed, 0.9

    no_fault_patterns = (
        "no driver was wholly or predominantly to blame",
        "no driver predominantly to blame",
        "racing incident",
        "no further action",
    )
    if any(pattern in combined for pattern in no_fault_patterns):
        if "wholly to blame" not in combined and "predominantly to blame" not in combined:
            return "none", 0.72

    fault_patterns = (
        "wholly to blame",
        "predominantly to blame",
        "caused a collision",
        "caused the collision",
        "wholly responsible",
        "predominantly responsible",
    )
    if any(pattern in combined for pattern in fault_patterns):
        car_match = re.search(r"\bcar\s+(\d+)\b", combined)
        if car_match:
            return f"car_{car_match.group(1)}", 0.86
        primary_driver = next((driver for driver in drivers if not driver.startswith("car_")), "")
        if primary_driver:
            return primary_driver, 0.78
        return "unknown", 0.65

    return "", 0.0


def suggest_severity(
    fact: str,
    reason: str,
    decision: str,
    penalty: str,
) -> tuple[int | None, float]:
    text = normalize_text(f"{fact} {reason} {decision}").lower()
    penalty_text = str(penalty or "").lower().strip()

    if "disqualification" in penalty_text or "disqualified" in penalty_text:
        base = 5
        confidence = 0.82
    elif "drive_through" in penalty_text or re.search(r"\d+s_time_penalty", penalty_text):
        base = 4
        confidence = 0.8
    elif "grid" in penalty_text and "place" in penalty_text:
        base = 3
        confidence = 0.78
    elif "reprimand" in penalty_text:
        base = 2
        confidence = 0.76
    elif "warning" in penalty_text:
        base = 2
        confidence = 0.74
    elif penalty_text in {"", "no_further_action"}:
        base = 1
        confidence = 0.7
    else:
        base = 3
        confidence = 0.55

    if any(keyword in text for keyword in ("dangerous", "reckless", "unsafe", "high speed")):
        base = min(5, base + 1)
        confidence = min(0.9, confidence + 0.05)
    if "minor" in text:
        base = max(1, base - 1)

    return base, round(confidence, 3)


def enrich_text_fields(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    provenance: EnrichmentProvenance | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    meta = load_meta(cfg)
    meta_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.meta.json"
    meta_rows: list[dict[str, Any]] = sio.read_json(meta_path) if meta_path.exists() else []

    min_confidence = cfg.enrichment_settings.driver_at_fault_min_confidence
    write_severity_to_review = cfg.enrichment_settings.write_severity_suggestions_to_review_queue

    for index, row in out.iterrows():
        incident_id = str(row.get("incident_id", ""))
        meta_row = dict(meta.get(incident_id, {"incident_id": incident_id}))
        document_id = str(meta_row.get("document_id", ""))
        doc = load_interim_document(cfg, document_id)
        fact, reason, decision = _document_text(doc)
        drivers = [part.strip() for part in str(row.get("drivers", "")).split(",") if part.strip()]

        fault_value, fault_confidence = infer_driver_at_fault(fact, reason, decision, drivers)
        severity_value, severity_confidence = suggest_severity(
            fact,
            reason,
            decision,
            str(row.get("penalty", "")),
        )

        meta_row["driver_at_fault_suggestion"] = fault_value
        meta_row["driver_at_fault_confidence"] = fault_confidence
        if write_severity_to_review and severity_value is not None:
            meta_row["severity_suggestion"] = severity_value
            meta_row["severity_confidence"] = severity_confidence

        if (
            fault_value
            and fault_confidence >= min_confidence
            and is_blank(row.get("driver_at_fault"))
        ):
            out.at[index, "driver_at_fault"] = fault_value
            if provenance:
                provenance.record(
                    incident_id,
                    "driver_at_fault",
                    "text_fields",
                    value=fault_value,
                    confidence=fault_confidence,
                )
        elif provenance and fault_value:
            provenance.record(
                incident_id,
                "driver_at_fault",
                "text_fields",
                value=fault_value,
                confidence=fault_confidence,
                suggestion_only=True,
            )

        if provenance and severity_value is not None:
            provenance.record(
                incident_id,
                "severity",
                "text_fields",
                value=str(severity_value),
                confidence=severity_confidence,
                suggestion_only=not is_blank(row.get("severity")),
            )

        meta[incident_id] = meta_row

    updated_rows = []
    seen = set()
    for row in meta_rows:
        incident_id = row.get("incident_id")
        if incident_id in meta:
            updated_rows.append(meta[incident_id])
            seen.add(incident_id)
        else:
            updated_rows.append(row)
    for incident_id, row in meta.items():
        if incident_id not in seen:
            updated_rows.append(row)
    save_meta_rows(cfg, updated_rows)

    return out
