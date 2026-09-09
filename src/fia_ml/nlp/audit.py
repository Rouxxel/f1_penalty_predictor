"""Audit helpers for NLP dataset construction."""

from __future__ import annotations

from typing import Any


def new_audit(text_profile: str) -> dict[str, Any]:
    return {
        "text_profile": text_profile,
        "seasons": {},
        "totals": {
            "labeled_incident_rows": 0,
            "flattened_rows": 0,
            "rows_with_text": 0,
            "missing_meta": 0,
            "missing_document_id": 0,
            "missing_interim_doc": 0,
            "text_build_errors": 0,
            "misaligned_skipped": 0,
        },
    }


def season_bucket(audit: dict[str, Any], season: int) -> dict[str, Any]:
    key = str(season)
    if key not in audit["seasons"]:
        audit["seasons"][key] = {
            "labeled_incident_rows": 0,
            "flattened_rows": 0,
            "rows_with_text": 0,
            "missing_meta": 0,
            "missing_document_id": 0,
            "missing_interim_doc": 0,
            "text_build_errors": 0,
            "misaligned_skipped": 0,
            "missing_interim_incident_ids": [],
            "text_build_error_incident_ids": [],
        }
    return audit["seasons"][key]


def bump_total(audit: dict[str, Any], field: str, amount: int = 1) -> None:
    audit["totals"][field] = int(audit["totals"].get(field, 0)) + amount


def finalize_audit(audit: dict[str, Any]) -> dict[str, Any]:
    totals = audit["totals"]
    season_rows = sum(s["flattened_rows"] for s in audit["seasons"].values())
    totals["flattened_rows"] = season_rows
    totals["rows_with_text"] = sum(s["rows_with_text"] for s in audit["seasons"].values())
    if totals["labeled_incident_rows"]:
        totals["join_rate"] = round(
            totals["rows_with_text"] / max(totals["flattened_rows"], 1),
            4,
        )
    else:
        totals["join_rate"] = 0.0
    return audit
