"""Build incident-centric CSV rows from parsed FIA documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.reference_data import load_reference
from fia_ml.data.schema import SCHEMA_COLUMNS, empty_row
from fia_ml.paths import ensure_dir
from fia_ml.utils import secure_file_io as sio


def slugify_name(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def classify_incident_type(fact: str, offence: str, keywords: dict[str, list[str]]) -> str:
    combined = f"{fact} {offence}".lower()
    for incident_type, terms in keywords.items():
        if any(term in combined for term in terms):
            return incident_type
    return "other"


def infer_sector(turn_number: str, circuit_slug: str, circuits: dict[str, Any]) -> str:
    if not turn_number:
        return ""
    circuit = circuits.get(circuit_slug, {})
    corners = circuit.get("corners", {})
    return str(corners.get(turn_number, ""))


def resolve_circuit(event: str, circuits_ref: dict[str, Any]) -> tuple[str, str, str]:
    event_map = circuits_ref.get("event_to_circuit", {})
    circuit_slug = event_map.get(event, slugify_name(event.replace(" Grand Prix", "")))
    circuit_meta = circuits_ref.get(circuit_slug, {})
    country = circuit_meta.get("country", "")
    first_season = str(circuit_meta.get("first_season", "")) if circuit_meta.get("first_season") else ""
    return circuit_slug, country, first_season


def normalize_fact(fact: str) -> str:
    return re.sub(r"\s+", " ", fact.lower()).strip()


def make_incident_id(season: int, session: str, car_number: str, fact: str) -> str:
    key = f"{season}|{session}|{car_number}|{normalize_fact(fact)}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return f"{season}_{session}_{digest}"


def _doc_sort_key(doc: dict[str, Any]) -> tuple[int, int]:
    title = doc.get("title", "").lower()
    is_correction = 0 if "correction" in title else 1
    type_rank = {"decision": 0, "offence": 1, "infringement": 1, "summons": 2}.get(
        doc.get("document_type", ""), 3
    )
    return (is_correction, type_rank)


def select_primary_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one primary document per car/session/fact cluster."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        fields = doc.get("parsed_fields", {})
        if doc.get("document_type") == "summons" and not fields.get("decision"):
            key = f"{fields.get('car_number')}|{fields.get('session')}|{normalize_fact(fields.get('fact', ''))}"
        else:
            if doc.get("document_type") not in {"decision", "offence", "infringement"}:
                continue
            if not fields.get("fact") and not fields.get("decision"):
                continue
            key = f"{fields.get('car_number')}|{fields.get('session')}|{normalize_fact(fields.get('fact', ''))}"
        clusters.setdefault(key, []).append(doc)

    selected: list[dict[str, Any]] = []
    for cluster in clusters.values():
        cluster.sort(key=_doc_sort_key)
        selected.append(cluster[0])
    return selected


def document_to_row(doc: dict[str, Any], refs: dict[str, Any], cfg: PipelineConfig) -> dict[str, str]:
    row = empty_row()
    fields = doc.get("parsed_fields", {})
    event = doc.get("event", "")
    season = int(doc.get("season", cfg.season))
    session = fields.get("session", "")
    fact = fields.get("fact", "")
    car_number = fields.get("car_number", "")

    circuit, country, first_season = resolve_circuit(event, refs["circuits"])
    incident_type = classify_incident_type(
        fact, fields.get("offence", ""), refs["incident_types"]
    )
    turn = fields.get("turn_number", "")
    sector = infer_sector(turn, circuit, refs["circuits"])

    driver_slug = slugify_name(fields.get("driver_name", ""))
    opponent_numbers = fields.get("opponent_car_numbers", [])
    opponent_slugs = [f"car_{num}" for num in opponent_numbers]

    drivers = [d for d in [driver_slug, *opponent_slugs] if d]
    teams = [fields.get("competitor", "").lower().replace(" ", "_") if fields.get("competitor") else ""]
    if len(drivers) > 1:
        teams = [teams[0] if teams else "", ""]

    row.update(
        {
            "incident_id": make_incident_id(season, session, car_number, fact or doc["document_id"]),
            "circuit": circuit,
            "country": country,
            "first_season": first_season,
            "season": str(season),
            "session": session,
            "incident_type": incident_type,
            "incident_classification": incident_type,
            "sector": sector,
            "num_drivers": str(len(drivers)),
            "drivers": ",".join(drivers),
            "respective_teams": ",".join(t for t in teams if t),
            "investigation": "TRUE"
            if doc.get("document_type") in {"decision", "offence", "infringement", "summons"}
            else "FALSE",
            "driver_at_fault": fields.get("driver_at_fault", ""),
            "penalty": fields.get("penalty", ""),
            "superlicense_points_added": fields.get("superlicense_points_added", ""),
            "mentioned_article": fields.get("mentioned_article", ""),
            "_document_id": doc.get("document_id", ""),
            "_parse_confidence": str(doc.get("parse_confidence", 0)),
            "_event": event,
            "_car_number": car_number,
        }
    )
    return row


def write_raw_incidents(df: pd.DataFrame, cfg: PipelineConfig, meta_rows: list[dict[str, Any]] | None = None) -> Path:
    out_dir = ensure_dir(cfg.path("csv_out"))
    path = out_dir / f"raw_incidents_{cfg.season}.csv"
    df.to_csv(path, index=False)
    if meta_rows is not None:
        sio.write_json(out_dir / f"raw_incidents_{cfg.season}.meta.json", meta_rows)
    return path


def build_incidents(parsed_docs: list[dict[str, Any]], cfg: PipelineConfig) -> pd.DataFrame:
    refs = load_reference(cfg)
    primary = select_primary_documents(parsed_docs)
    rows: list[dict[str, str]] = []
    meta_rows: list[dict[str, Any]] = []
    for doc in primary:
        row = document_to_row(doc, refs, cfg)
        fields = doc.get("parsed_fields", {})
        meta_rows.append(
            {
                "incident_id": row["incident_id"],
                "event": doc.get("event", ""),
                "car_number": fields.get("car_number", ""),
                "parse_confidence": doc.get("parse_confidence", 0),
                "document_id": doc.get("document_id", ""),
                "time": fields.get("time", ""),
                "date": fields.get("date", ""),
            }
        )
        rows.append({k: v for k, v in row.items() if not k.startswith("_")})

    df = pd.DataFrame(rows)
    if df.empty:
        write_raw_incidents(df, cfg, [])
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    out = df[SCHEMA_COLUMNS]
    write_raw_incidents(out, cfg, meta_rows)
    return out


def build_from_interim(cfg: PipelineConfig) -> pd.DataFrame:
    interim_dir = cfg.path("interim_docs") / str(cfg.season)
    docs: list[dict[str, Any]] = []
    if not interim_dir.exists():
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    for json_path in sorted(interim_dir.glob("*.json")):
        if json_path.name.endswith("_failures.json"):
            continue
        payload = sio.read_json(json_path)
        if not isinstance(payload, dict):
            continue
        docs.append(payload)
    return build_incidents(docs, cfg)
