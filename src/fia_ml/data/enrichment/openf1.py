"""OpenF1 API enrichment for 2023+ race-state columns."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta
from fia_ml.data.enrichment.ergast import build_car_to_driver_map, load_race_results
from fia_ml.data.enrichment.provenance import EnrichmentProvenance
from fia_ml.data.enrichment.timestamp import _round_for_row
from fia_ml.paths import ensure_dir
from fia_ml.utils import secure_file_io as sio

_SESSION_TYPE_MAP = {
    "race": "Race",
    "qualifying": "Qualifying",
    "practice": "Practice 1",
    "sprint": "Sprint",
}


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def fetch_openf1(
    endpoint: str,
    params: dict[str, Any],
    cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    cache_root = ensure_dir(
        cfg.enrichment_settings.openf1_cache_dir / str(cfg.season) / endpoint
    )
    cache_file = cache_root / f"{_cache_key(endpoint, params)}.json"
    if cache_file.exists():
        return sio.read_json(cache_file)

    base_url = cfg.enrichment_settings.openf1_base_url.rstrip("/")
    url = f"{base_url}/{endpoint}?{urlencode(params)}"
    response = requests.get(url, timeout=60, headers={"User-Agent": "f1-penalty-predictor/1.0"})
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected OpenF1 response for {endpoint}")
    sio.write_json(cache_file, data)
    time.sleep(0.25)
    return data


def load_sessions_for_year(cfg: PipelineConfig) -> list[dict[str, Any]]:
    index_path = cfg.path("reference") / f"openf1_sessions_{cfg.season}.json"
    if index_path.exists():
        cached = sio.read_json(index_path)
        if isinstance(cached, list):
            return cached

    sessions = fetch_openf1("sessions", {"year": cfg.season}, cfg)
    sio.write_json(index_path, sessions)
    return sessions


def _normalize_event_token(event: str) -> str:
    return re.sub(r"\s+", " ", event.lower().replace("grand prix", "")).strip()


def resolve_meeting_key(event: str, sessions: list[dict[str, Any]]) -> int | None:
    if not event or not sessions:
        return None

    meeting_samples: dict[int, dict[str, Any]] = {}
    for session in sessions:
        meeting_samples.setdefault(int(session["meeting_key"]), session)

    event_norm = _normalize_event_token(event)
    for sample in meeting_samples.values():
        tokens = [
            str(sample.get("country_name", "")),
            str(sample.get("location", "")),
            str(sample.get("circuit_short_name", "")),
        ]
        for token in tokens:
            token_norm = token.lower().strip()
            if token_norm and (token_norm in event_norm or event_norm in token_norm):
                return int(sample["meeting_key"])
    return None


def resolve_session_key(
    event: str,
    session: str,
    sessions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    meeting_key = resolve_meeting_key(event, sessions)
    if meeting_key is None:
        return None

    target_type = _SESSION_TYPE_MAP.get(session.lower(), session)
    meeting_sessions = [item for item in sessions if int(item["meeting_key"]) == meeting_key]
    for item in meeting_sessions:
        if item.get("session_type") == target_type or item.get("session_name") == target_type:
            return item

    if session.lower() == "practice":
        practice_sessions = [
            item for item in meeting_sessions if str(item.get("session_type", "")).startswith("Practice")
        ]
        if practice_sessions:
            return sorted(practice_sessions, key=lambda row: str(row.get("session_name", "")))[0]
    return None


def build_driver_number_map(
    cfg: PipelineConfig,
    session_key: int,
    round_num: int,
) -> dict[str, int]:
    numbers: dict[str, int] = {}
    try:
        results = load_race_results(cfg, round_num)
        for car_number, slug in build_car_to_driver_map(results).items():
            numbers[slug] = int(car_number)
    except RuntimeError:
        pass

    try:
        drivers = fetch_openf1("drivers", {"session_key": session_key}, cfg)
    except (requests.RequestException, RuntimeError):
        return numbers

    for driver in drivers:
        number = driver.get("driver_number")
        if number in (None, ""):
            continue
        full_name = str(driver.get("full_name", "")).lower().replace(" ", "_")
        last_name = full_name.split("_")[-1] if full_name else ""
        acronym = str(driver.get("name_acronym", "")).lower()
        if full_name:
            numbers.setdefault(full_name, int(number))
        if last_name:
            numbers.setdefault(last_name, int(number))
        if acronym:
            numbers.setdefault(acronym, int(number))
    return numbers


def resolve_driver_numbers(
    row: pd.Series,
    meta_row: dict[str, Any],
    driver_number_map: dict[str, int],
) -> list[int | None]:
    driver_ids = [part.strip() for part in str(row.get("drivers", "")).split(",") if part.strip()]
    if not driver_ids:
        return []

    numbers: list[int | None] = []
    primary_car = str(meta_row.get("car_number", "")).strip()
    for index, driver_id in enumerate(driver_ids):
        if driver_id.startswith("car_"):
            numbers.append(None)
            continue
        if index == 0 and primary_car.isdigit():
            numbers.append(int(primary_car))
            continue
        number = driver_number_map.get(driver_id)
        if number is None:
            number = driver_number_map.get(driver_id.split("_")[-1])
        numbers.append(number)
    return numbers


def incident_datetime_utc(
    meta_row: dict[str, Any],
    session_info: dict[str, Any],
) -> datetime | None:
    offset = meta_row.get("session_offset_seconds")
    session_start = _parse_iso_datetime(str(session_info.get("date_start", "")))
    if offset is not None and session_start is not None:
        try:
            return session_start + timedelta(seconds=float(offset))
        except (TypeError, ValueError):
            pass
    return None


def lap_at_time(laps: list[dict[str, Any]], incident_dt: datetime) -> int | None:
    lap_number, _ = lap_at_time_with_error(laps, incident_dt)
    return lap_number


def lap_at_time_with_error(
    laps: list[dict[str, Any]],
    incident_dt: datetime,
) -> tuple[int | None, float | None]:
    best_lap = None
    best_start: datetime | None = None
    for lap in laps:
        start = _parse_iso_datetime(str(lap.get("date_start", "")))
        if start is None:
            continue
        if start <= incident_dt and (best_start is None or start > best_start):
            best_start = start
            best_lap = int(lap["lap_number"])
    if best_lap is None or best_start is None:
        return None, None
    return best_lap, abs((incident_dt - best_start).total_seconds())


def max_lap_number(laps: list[dict[str, Any]]) -> int | None:
    if not laps:
        return None
    return max(int(item["lap_number"]) for item in laps if item.get("lap_number") is not None)


def normalize_flag(flag_value: str | None, message: str = "") -> str:
    text = f"{flag_value or ''} {message}".upper()
    if "DOUBLE YELLOW" in text:
        return "double_yellow"
    if "YELLOW" in text:
        return "yellow"
    if "RED" in text:
        return "red"
    if "VIRTUAL SAFETY CAR" in text or "VSC" in text:
        return "vsc"
    if "SAFETY CAR" in text or text.strip() == "SC":
        return "safety_car"
    if "BLUE" in text:
        return "blue"
    if "GREEN" in text:
        return "green"
    if "CHEQUERED" in text or "CHECKERED" in text:
        return "chequered"
    return ""


def normalize_track_sector(sector_value: Any, message: str = "") -> str:
    if sector_value in (1, 2, 3, "1", "2", "3"):
        return str(int(sector_value))
    message_upper = str(message).upper()
    for sector in (1, 2, 3):
        if re.search(rf"\bSECTOR\s*{sector}\b", message_upper):
            return str(sector)
    try:
        mini_sector = int(sector_value)
    except (TypeError, ValueError):
        return ""
    if 1 <= mini_sector <= 3:
        return str(mini_sector)
    return str(min(((mini_sector - 1) // 8) + 1, 3))


def nearest_race_control_message(
    messages: list[dict[str, Any]],
    incident_dt: datetime,
    tolerance_seconds: float,
) -> tuple[dict[str, Any] | None, float | None]:
    best = None
    best_diff = None
    for message in messages:
        message_dt = _parse_iso_datetime(str(message.get("date", "")))
        if message_dt is None:
            continue
        diff = abs((message_dt - incident_dt).total_seconds())
        if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
            best = message
            best_diff = diff
    return best, best_diff


def positions_at_time(
    positions: list[dict[str, Any]],
    driver_numbers: list[int | None],
    incident_dt: datetime,
    tolerance_seconds: float,
) -> list[str]:
    by_driver: dict[int, list[dict[str, Any]]] = {}
    for item in positions:
        number = item.get("driver_number")
        if number is None:
            continue
        by_driver.setdefault(int(number), []).append(item)

    values: list[str] = []
    for number in driver_numbers:
        if number is None:
            values.append("")
            continue
        best_position = None
        best_diff = None
        for item in by_driver.get(number, []):
            item_dt = _parse_iso_datetime(str(item.get("date", "")))
            if item_dt is None:
                continue
            diff = abs((item_dt - incident_dt).total_seconds())
            if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
                best_position = str(item.get("position", ""))
                best_diff = diff
        values.append(best_position or "")
    return values


def _should_write(value: Any, row_value: Any, fill_gaps_only: bool) -> bool:
    return not fill_gaps_only or is_blank(row_value)


def enrich_with_openf1(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    fill_gaps_only: bool = True,
    provenance: EnrichmentProvenance | None = None,
) -> pd.DataFrame:
    if df.empty or cfg.season < cfg.enrichment_settings.openf1_enabled_from_season:
        return df

    out = df.copy()
    meta = load_meta(cfg)
    tolerance = cfg.enrichment_settings.max_match_error_seconds
    sessions = load_sessions_for_year(cfg)
    session_data_cache: dict[int, dict[str, list[dict[str, Any]]]] = {}
    driver_number_cache: dict[int, dict[str, int]] = {}

    for idx, row in out.iterrows():
        incident_id = str(row.get("incident_id", ""))
        meta_row = meta.get(incident_id, {})
        event = str(meta_row.get("event", ""))
        session = str(row.get("session", "")).lower()
        session_info = resolve_session_key(event, session, sessions)
        if not session_info:
            continue

        session_key = int(session_info["session_key"])
        round_num = _round_for_row(row, cfg, meta_row)

        if session_key not in session_data_cache:
            try:
                session_data_cache[session_key] = {
                    "laps": fetch_openf1("laps", {"session_key": session_key}, cfg),
                    "position": fetch_openf1("position", {"session_key": session_key}, cfg),
                    "race_control": fetch_openf1("race_control", {"session_key": session_key}, cfg),
                }
            except requests.RequestException:
                session_data_cache[session_key] = {"laps": [], "position": [], "race_control": []}

        session_data = session_data_cache[session_key]
        laps = session_data["laps"]
        positions = session_data["position"]
        race_control = session_data["race_control"]

        if round_num is not None and session_key not in driver_number_cache:
            driver_number_cache[session_key] = build_driver_number_map(cfg, session_key, round_num)
        driver_number_map = driver_number_cache.get(session_key, {})
        driver_numbers = resolve_driver_numbers(row, meta_row, driver_number_map)

        lap_hint = meta_row.get("lap_hint")
        incident_dt = incident_datetime_utc(meta_row, session_info)

        if lap_hint and _should_write(True, row.get("lap"), fill_gaps_only):
            out.at[idx, "lap"] = str(lap_hint)
            if provenance:
                provenance.record(incident_id, "lap", "openf1", value=str(lap_hint), via="lap_hint")
        elif incident_dt is not None and _should_write(True, row.get("lap"), fill_gaps_only):
            lap_number, match_error = lap_at_time_with_error(laps, incident_dt)
            if lap_number is not None:
                out.at[idx, "lap"] = str(lap_number)
                if provenance:
                    provenance.record(
                        incident_id,
                        "lap",
                        "openf1",
                        value=str(lap_number),
                        session_key=session_key,
                        match_error_seconds=match_error,
                    )

        if _should_write(True, row.get("full_laps"), fill_gaps_only):
            max_lap = max_lap_number(laps)
            if max_lap is not None:
                out.at[idx, "full_laps"] = str(max_lap)
                if provenance:
                    provenance.record(
                        incident_id,
                        "full_laps",
                        "openf1",
                        value=str(max_lap),
                        session_key=session_key,
                    )

        if incident_dt is not None:
            rc_message, rc_error = nearest_race_control_message(race_control, incident_dt, tolerance)
            if rc_message:
                if _should_write(True, row.get("flag"), fill_gaps_only):
                    flag = normalize_flag(rc_message.get("flag"), str(rc_message.get("message", "")))
                    if flag:
                        out.at[idx, "flag"] = flag
                        if provenance:
                            provenance.record(
                                incident_id,
                                "flag",
                                "openf1",
                                value=flag,
                                message=str(rc_message.get("message", "")),
                                match_error_seconds=rc_error,
                            )
                if _should_write(True, row.get("sector"), fill_gaps_only):
                    sector = normalize_track_sector(
                        rc_message.get("sector"),
                        str(rc_message.get("message", "")),
                    )
                    if sector:
                        out.at[idx, "sector"] = sector
                        if provenance:
                            provenance.record(
                                incident_id,
                                "sector",
                                "openf1",
                                value=sector,
                                message=str(rc_message.get("message", "")),
                                match_error_seconds=rc_error,
                            )

            if driver_numbers and _should_write(True, row.get("positions_of_involved parties"), fill_gaps_only):
                position_values = positions_at_time(positions, driver_numbers, incident_dt, tolerance)
                if any(position_values):
                    joined = ",".join(position_values)
                    out.at[idx, "positions_of_involved parties"] = joined
                    if provenance:
                        provenance.record(
                            incident_id,
                            "positions_of_involved parties",
                            "openf1",
                            value=joined,
                            session_key=session_key,
                        )

    return out
