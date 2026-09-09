"""Normalize FIA document times into session-relative offsets for race-state joins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import load_meta, map_event_to_round
from fia_ml.data.enrichment.ergast import fetch_with_cache, map_event_to_round as ergast_event_to_round
from fia_ml.data.reference_data import build_event_name_to_circuit_map, load_circuits, load_seasons
from fia_ml.utils import secure_file_io as sio

_DATE_FORMATS = (
    "%d %B %Y",
    "%d %b %Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
)


@dataclass(frozen=True)
class ParsedIncidentTime:
    method: str
    hours: int
    minutes: int
    seconds: int = 0
    lap_hint: int | None = None
    confidence: float = 0.0
    raw: str = ""


def parse_time_text(raw_time: str) -> ParsedIncidentTime | None:
    """Parse common FIA PDF time strings into a structured incident time."""
    text = str(raw_time or "").strip()
    if not text:
        return None

    lap_match = re.search(r"\blap\s*(\d{1,3})\b", text, re.IGNORECASE)
    if lap_match:
        return ParsedIncidentTime(
            method="lap_hint",
            hours=0,
            minutes=0,
            lap_hint=int(lap_match.group(1)),
            confidence=0.85,
            raw=text,
        )

    elapsed_match = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})\b", text)
    if elapsed_match:
        return ParsedIncidentTime(
            method="session_elapsed",
            hours=int(elapsed_match.group(1)),
            minutes=int(elapsed_match.group(2)),
            seconds=int(elapsed_match.group(3)),
            confidence=0.9,
            raw=text,
        )

    clock_match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text)
    if clock_match:
        seconds = int(clock_match.group(3) or 0)
        return ParsedIncidentTime(
            method="document_clock",
            hours=int(clock_match.group(1)),
            minutes=int(clock_match.group(2)),
            seconds=seconds,
            confidence=0.55,
            raw=text,
        )

    inline_elapsed = re.search(r"\bat\s+(\d{1,2}):(\d{2}):(\d{2})\b", text, re.IGNORECASE)
    if inline_elapsed:
        return ParsedIncidentTime(
            method="session_elapsed",
            hours=int(inline_elapsed.group(1)),
            minutes=int(inline_elapsed.group(2)),
            seconds=int(inline_elapsed.group(3)),
            confidence=0.75,
            raw=text,
        )

    return None


def parse_incident_date(raw_date: str) -> date | None:
    text = str(raw_date or "").strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_ergast_datetime(raw_date: str, raw_time: str) -> datetime | None:
    if not raw_date or not raw_time:
        return None
    time_value = str(raw_time).replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{raw_date}T{time_value}", fmt)
        except ValueError:
            continue
    return None


def _session_schedule_key(session: str) -> str | None:
    mapping = {
        "race": "Race",
        "qualifying": "Qualifying",
        "practice": "FirstPractice",
        "sprint": "Sprint",
    }
    return mapping.get(session.lower())


def load_weekend_schedule(cfg: PipelineConfig, round_num: int) -> dict[str, Any]:
    payload = fetch_with_cache(f"{cfg.season}/{round_num}.json", cfg)
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    return races[0]


def get_session_start_datetime(schedule: dict[str, Any], session: str) -> datetime | None:
    session_key = _session_schedule_key(session)
    if not session_key:
        return None

    if session_key == "Race":
        block = schedule
    else:
        block = schedule.get(session_key) or {}
        if not block:
            return None

    return _parse_ergast_datetime(str(block.get("date", "")), str(block.get("time", "")))


def _round_for_row(row: pd.Series, cfg: PipelineConfig, meta_row: dict[str, Any]) -> int | None:
    round_value = row.get("round", "")
    if round_value not in (None, ""):
        try:
            return int(float(round_value))
        except ValueError:
            pass

    seasons = load_seasons(cfg)
    season_data = seasons.get(str(cfg.season))
    if season_data:
        circuits = load_circuits(cfg)
        event_to_circuit = build_event_name_to_circuit_map(circuits)
        calendar = list(season_data.get("calendar_in_order", []))
        round_num, _ = map_event_to_round(str(meta_row.get("event", "")), calendar, event_to_circuit)
        if round_num:
            return round_num

    calendar = []
    try:
        from fia_ml.data.enrichment.ergast import load_season_calendar

        calendar = load_season_calendar(cfg)
    except RuntimeError:
        return None
    return ergast_event_to_round(str(meta_row.get("event", "")), calendar)


def compute_session_offset_seconds(
    parsed: ParsedIncidentTime,
    *,
    session_start: datetime | None,
    incident_date: date | None,
    max_match_error_seconds: float,
) -> tuple[float | None, float]:
    """Return (session_offset_seconds, confidence)."""
    if parsed.method == "session_elapsed":
        offset = parsed.hours * 3600 + parsed.minutes * 60 + parsed.seconds
        if offset < 0 or offset > max_match_error_seconds * 120:
            return None, parsed.confidence * 0.5
        return float(offset), parsed.confidence

    if parsed.method != "document_clock" or session_start is None or incident_date is None:
        return None, parsed.confidence * 0.25

    incident_dt = datetime.combine(
        incident_date,
        datetime.min.time(),
    ) + timedelta(hours=parsed.hours, minutes=parsed.minutes, seconds=parsed.seconds)

    # Assumption: FIA document header times are circuit-local; Ergast session times are UTC.
    # We compare naive datetimes on the same calendar day as a best-effort approximation.
    session_start_naive = session_start.replace(tzinfo=None)
    offset = (incident_dt - session_start_naive).total_seconds()
    if offset < 0 or offset > max_match_error_seconds * 120:
        return None, parsed.confidence * 0.35
    return offset, parsed.confidence


def enrich_meta_timestamp(
    meta_row: dict[str, Any],
    row: pd.Series,
    cfg: PipelineConfig,
    *,
    schedule_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(meta_row)
    parsed = parse_time_text(str(meta_row.get("time", "")))
    if parsed is None:
        updated["time_parse_method"] = "unparsed"
        updated["time_parse_confidence"] = 0.0
        updated["session_offset_seconds"] = None
        updated["incident_time_local"] = None
        updated["lap_hint"] = None
        return updated

    updated["time_parse_method"] = parsed.method
    updated["time_parse_confidence"] = parsed.confidence
    updated["lap_hint"] = parsed.lap_hint

    session = str(row.get("session", "")).lower()
    round_num = _round_for_row(row, cfg, meta_row)
    session_start = None
    if round_num is not None:
        schedule = schedule_cache.get(round_num)
        if schedule is None:
            try:
                schedule = load_weekend_schedule(cfg, round_num)
            except RuntimeError:
                schedule = {}
            schedule_cache[round_num] = schedule
        session_start = get_session_start_datetime(schedule, session)

    incident_date = parse_incident_date(str(meta_row.get("date", "")))
    offset, confidence = compute_session_offset_seconds(
        parsed,
        session_start=session_start,
        incident_date=incident_date,
        max_match_error_seconds=cfg.enrichment_settings.max_match_error_seconds,
    )
    updated["time_parse_confidence"] = confidence
    updated["session_offset_seconds"] = offset
    if offset is not None and incident_date is not None and parsed.method == "document_clock":
        incident_dt = datetime.combine(incident_date, datetime.min.time()) + timedelta(
            hours=parsed.hours,
            minutes=parsed.minutes,
            seconds=parsed.seconds,
        )
        updated["incident_time_local"] = incident_dt.isoformat(sep=" ")
    else:
        updated["incident_time_local"] = None
    return updated


def save_meta_rows(cfg: PipelineConfig, meta_rows: list[dict[str, Any]]) -> None:
    meta_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.meta.json"
    sio.write_json(meta_path, meta_rows)


def timestamp_audit(meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(meta)
    if total == 0:
        return {"rows": 0, "parsed_rate": 0.0, "offset_rate": 0.0, "methods": {}}

    parsed = 0
    with_offset = 0
    methods: dict[str, int] = {}
    for row in meta.values():
        method = str(row.get("time_parse_method", "unparsed"))
        methods[method] = methods.get(method, 0) + 1
        if method != "unparsed":
            parsed += 1
        if row.get("session_offset_seconds") is not None:
            with_offset += 1

    return {
        "rows": total,
        "parsed_rate": round(parsed / total, 4),
        "offset_rate": round(with_offset / total, 4),
        "methods": methods,
    }


def enrich_timestamps(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df, {"rows": 0, "parsed_rate": 0.0, "offset_rate": 0.0, "methods": {}}

    meta_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.meta.json"
    meta_rows: list[dict[str, Any]] = sio.read_json(meta_path) if meta_path.exists() else []
    meta = {row["incident_id"]: row for row in meta_rows}
    schedule_cache: dict[int, dict[str, Any]] = {}

    for idx, row in df.iterrows():
        incident_id = str(row.get("incident_id", ""))
        if not incident_id:
            continue
        meta_row = meta.get(incident_id, {"incident_id": incident_id})
        meta[incident_id] = enrich_meta_timestamp(
            meta_row,
            row,
            cfg,
            schedule_cache=schedule_cache,
        )

    updated_rows = [meta[row["incident_id"]] for row in meta_rows if row["incident_id"] in meta]
    for incident_id, row in meta.items():
        if incident_id not in {item["incident_id"] for item in updated_rows}:
            updated_rows.append(row)
    save_meta_rows(cfg, updated_rows)
    return df, timestamp_audit(meta)
