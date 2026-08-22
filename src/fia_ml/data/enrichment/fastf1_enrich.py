"""FastF1 session enrichment for lap, weather, and race control context."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta, map_event_to_round
from fia_ml.data.reference_data import build_event_name_to_circuit_map, load_circuits, load_seasons
from fia_ml.paths import ensure_dir


def _session_name(session: str) -> str:
    mapping = {
        "race": "R",
        "qualifying": "Q",
        "practice": "FP1",
        "sprint": "S",
    }
    if session == "practice":
        return "FP1"
    return mapping.get(session, session.upper()[:1])


def _parse_time_to_seconds(value: str) -> int | None:
    if not value:
        return None
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 3600 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    except ValueError:
        return None
    return None


def _round_from_reference(event: str, cfg: PipelineConfig) -> int | None:
    seasons = load_seasons(cfg)
    season_data = seasons.get(str(cfg.season))
    if not season_data:
        return None
    circuits = load_circuits(cfg)
    event_to_circuit = build_event_name_to_circuit_map(circuits)
    calendar = list(season_data.get("calendar_in_order", []))
    round_num, _ = map_event_to_round(event, calendar, event_to_circuit)
    return round_num


def enrich_with_fastf1(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    fill_gaps_only: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    try:
        import fastf1
    except ImportError:
        return df

    cache_dir = ensure_dir(cfg.path("fastf1_cache"))
    if cfg.enrichment.get("fastf1_cache_enabled", True):
        fastf1.Cache.enable_cache(str(cache_dir))

    out = df.copy()
    meta = load_meta(cfg)
    session_cache: dict[tuple[int, str], Any] = {}

    for idx, row in out.iterrows():
        incident_id = row.get("incident_id", "")
        meta_row = meta.get(incident_id, {})
        event = meta_row.get("event", "")
        session = str(row.get("session", "")).lower()
        round_num = _round_from_reference(event, cfg)
        if not round_num:
            continue

        cache_key = (round_num, session)
        if cache_key not in session_cache:
            try:
                event_obj = fastf1.get_session(cfg.season, round_num, _session_name(session))
                event_obj.load(telemetry=False, weather=True, messages=True)
                session_cache[cache_key] = event_obj
            except Exception:  # noqa: BLE001 - skip missing sessions
                session_cache[cache_key] = None
                continue

        event_obj = session_cache[cache_key]
        if event_obj is None:
            continue

        try:
            if (
                (not fill_gaps_only or is_blank(row.get("full_laps")))
                and hasattr(event_obj, "laps")
                and event_obj.laps is not None
                and len(event_obj.laps) > 0
            ):
                out.at[idx, "full_laps"] = str(int(event_obj.laps["LapNumber"].max()))
        except Exception:  # noqa: BLE001
            pass

        try:
            if is_blank(row.get("track_conditions")) or is_blank(row.get("weather_conditions")) or not fill_gaps_only:
                weather = event_obj.weather_data
                if weather is not None and len(weather) > 0:
                    last = weather.iloc[-1]
                    rainfall = getattr(last, "Rainfall", False)
                    track = str(getattr(last, "TrackStatus", "Dry"))
                    if is_blank(row.get("track_conditions")) or not fill_gaps_only:
                        out.at[idx, "track_conditions"] = (
                            "wet" if rainfall or "wet" in track.lower() else "dry"
                        )
                    if is_blank(row.get("weather_conditions")) or not fill_gaps_only:
                        out.at[idx, "weather_conditions"] = "rain" if rainfall else "sunny"
        except Exception:  # noqa: BLE001
            pass

        incident_time = _parse_time_to_seconds(str(meta_row.get("time", "")))
        if incident_time and session == "race" and (is_blank(row.get("lap")) or not fill_gaps_only):
            try:
                laps = event_obj.laps
                for _, lap_row in laps.iterrows():
                    lap_time = lap_row.get("Time")
                    if lap_time is None:
                        continue
                    total_seconds = lap_time.total_seconds() if hasattr(lap_time, "total_seconds") else None
                    if total_seconds is not None and total_seconds >= incident_time:
                        out.at[idx, "lap"] = str(int(lap_row["LapNumber"]))
                        break
            except Exception:  # noqa: BLE001
                pass

        try:
            if is_blank(row.get("safety_car")) or not fill_gaps_only:
                messages = event_obj.race_control_messages
                if messages is not None and len(messages) > 0:
                    sc_active = any("SAFETY CAR" in str(m).upper() for m in messages["Message"].astype(str))
                    vsc_active = any(
                        "VIRTUAL SAFETY CAR" in str(m).upper() for m in messages["Message"].astype(str)
                    )
                    if sc_active:
                        out.at[idx, "safety_car"] = "safety_car"
                    elif vsc_active:
                        out.at[idx, "safety_car"] = "vsc"
                    else:
                        out.at[idx, "safety_car"] = "none"
        except Exception:  # noqa: BLE001
            pass

    return out
