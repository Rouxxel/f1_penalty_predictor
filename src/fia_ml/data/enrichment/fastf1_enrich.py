"""FastF1 session enrichment for lap, weather, and race control context."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta
from fia_ml.data.enrichment.ergast import build_car_to_driver_map, load_race_results
from fia_ml.data.enrichment.openf1 import normalize_flag, normalize_track_sector, resolve_driver_numbers
from fia_ml.data.enrichment.provenance import EnrichmentProvenance
from fia_ml.data.enrichment.timestamp import _round_for_row
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


def _parse_time_to_seconds(value: str) -> float | None:
    if not value:
        return None
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            return float(int(parts[0]) * 3600 + int(parts[1]))
        if len(parts) == 3:
            return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
    except ValueError:
        return None
    return None


def incident_session_seconds(meta_row: dict[str, Any]) -> float | None:
    offset = meta_row.get("session_offset_seconds")
    if offset is not None:
        try:
            return float(offset)
        except (TypeError, ValueError):
            pass
    return _parse_time_to_seconds(str(meta_row.get("time", "")))


def timedelta_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return None


def build_driver_number_map_from_ergast(cfg: PipelineConfig, round_num: int) -> dict[str, int]:
    try:
        results = load_race_results(cfg, round_num)
    except RuntimeError:
        return {}
    return {slug: int(number) for number, slug in build_car_to_driver_map(results).items()}


def lap_number_at_session_time(laps: pd.DataFrame, incident_seconds: float) -> int | None:
    if laps is None or laps.empty or "Time" not in laps.columns or "LapNumber" not in laps.columns:
        return None

    eligible = laps[laps["Time"].notna()].copy()
    if eligible.empty:
        return None

    eligible["_seconds"] = eligible["Time"].map(timedelta_seconds)
    eligible = eligible[eligible["_seconds"].notna() & (eligible["_seconds"] <= incident_seconds)]
    if eligible.empty:
        return None
    return int(eligible["LapNumber"].max())


def nearest_race_control_row(
    messages: pd.DataFrame,
    incident_seconds: float,
    tolerance_seconds: float,
) -> pd.Series | None:
    if messages is None or messages.empty or "Time" not in messages.columns:
        return None

    best_row = None
    best_diff = None
    for _, row in messages.iterrows():
        message_seconds = timedelta_seconds(row.get("Time"))
        if message_seconds is None:
            continue
        diff = abs(message_seconds - incident_seconds)
        if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
            best_row = row
            best_diff = diff
    return best_row


def flag_at_session_time(
    messages: pd.DataFrame,
    incident_seconds: float,
    tolerance_seconds: float,
) -> str:
    row = nearest_race_control_row(messages, incident_seconds, tolerance_seconds)
    if row is None:
        return ""
    message = str(row.get("Message", ""))
    flag_column = row.get("Flag", "")
    return normalize_flag(str(flag_column) if flag_column not in (None, "") else None, message)


def sector_at_session_time(
    messages: pd.DataFrame,
    incident_seconds: float,
    tolerance_seconds: float,
) -> str:
    row = nearest_race_control_row(messages, incident_seconds, tolerance_seconds)
    if row is None:
        return ""
    message = str(row.get("Message", ""))
    sector_value = row.get("Sector")
    if sector_value in (None, "") and "Sector" in messages.columns:
        sector_value = row.get("Sector")
    return normalize_track_sector(sector_value, message)


def safety_car_at_session_time(
    messages: pd.DataFrame,
    incident_seconds: float,
    tolerance_seconds: float,
) -> str:
    row = nearest_race_control_row(messages, incident_seconds, tolerance_seconds)
    if row is None:
        return ""
    message = str(row.get("Message", "")).upper()
    if "VIRTUAL SAFETY CAR" in message:
        return "vsc"
    if "SAFETY CAR" in message:
        return "safety_car"
    return "none"


def positions_at_session_time(
    laps: pd.DataFrame,
    driver_numbers: list[int | None],
    incident_seconds: float,
    tolerance_seconds: float,
) -> list[str]:
    if laps is None or laps.empty:
        return ["" for _ in driver_numbers]

    values: list[str] = []
    for number in driver_numbers:
        if number is None:
            values.append("")
            continue
        driver_laps = laps[laps.get("DriverNumber") == number] if "DriverNumber" in laps.columns else laps.iloc[0:0]
        best_position = None
        best_diff = None
        for _, lap_row in driver_laps.iterrows():
            lap_seconds = timedelta_seconds(lap_row.get("Time"))
            if lap_seconds is None:
                continue
            diff = abs(lap_seconds - incident_seconds)
            if diff <= tolerance_seconds and (best_diff is None or diff < best_diff):
                position = lap_row.get("Position")
                if position is not None and not pd.isna(position):
                    best_position = str(int(position))
                    best_diff = diff
        values.append(best_position or "")
    return values


def _should_write(row_value: Any, fill_gaps_only: bool) -> bool:
    return not fill_gaps_only or is_blank(row_value)


def enrich_with_fastf1(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    fill_gaps_only: bool = True,
    provenance: EnrichmentProvenance | None = None,
) -> pd.DataFrame:
    if df.empty or not cfg.enrichment_settings.fastf1_enabled:
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
    driver_number_cache: dict[int, dict[str, int]] = {}
    tolerance = cfg.enrichment_settings.max_match_error_seconds

    for idx, row in out.iterrows():
        incident_id = str(row.get("incident_id", ""))
        meta_row = meta.get(incident_id, {})
        event = str(meta_row.get("event", ""))
        session = str(row.get("session", "")).lower()
        round_num = _round_for_row(row, cfg, meta_row)
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

        event_obj = session_cache[cache_key]
        if event_obj is None:
            continue

        laps = getattr(event_obj, "laps", None)
        messages = getattr(event_obj, "race_control_messages", None)

        if round_num not in driver_number_cache:
            driver_number_cache[round_num] = build_driver_number_map_from_ergast(cfg, round_num)
        driver_numbers = resolve_driver_numbers(row, meta_row, driver_number_cache[round_num])

        try:
            if _should_write(row.get("full_laps"), fill_gaps_only) and laps is not None and len(laps) > 0:
                full_laps = str(int(laps["LapNumber"].max()))
                out.at[idx, "full_laps"] = full_laps
                if provenance:
                    provenance.record(incident_id, "full_laps", "fastf1", value=full_laps, round=round_num)
        except Exception:  # noqa: BLE001
            pass

        try:
            if is_blank(row.get("track_conditions")) or is_blank(row.get("weather_conditions")) or not fill_gaps_only:
                weather = event_obj.weather_data
                if weather is not None and len(weather) > 0:
                    last = weather.iloc[-1]
                    rainfall = getattr(last, "Rainfall", False)
                    track = str(getattr(last, "TrackStatus", "Dry"))
                    if _should_write(row.get("track_conditions"), fill_gaps_only):
                        track_value = "wet" if rainfall or "wet" in track.lower() else "dry"
                        out.at[idx, "track_conditions"] = track_value
                        if provenance:
                            provenance.record(
                                incident_id, "track_conditions", "fastf1", value=track_value, round=round_num
                            )
                    if _should_write(row.get("weather_conditions"), fill_gaps_only):
                        weather_value = "rain" if rainfall else "sunny"
                        out.at[idx, "weather_conditions"] = weather_value
                        if provenance:
                            provenance.record(
                                incident_id, "weather_conditions", "fastf1", value=weather_value, round=round_num
                            )
        except Exception:  # noqa: BLE001
            pass

        incident_seconds = incident_session_seconds(meta_row)
        lap_hint = meta_row.get("lap_hint")

        if lap_hint and _should_write(row.get("lap"), fill_gaps_only):
            out.at[idx, "lap"] = str(lap_hint)
            if provenance:
                provenance.record(incident_id, "lap", "fastf1", value=str(lap_hint), via="lap_hint", round=round_num)
        elif (
            incident_seconds is not None
            and incident_seconds >= 0
            and _should_write(row.get("lap"), fill_gaps_only)
            and laps is not None
        ):
            try:
                lap_number = lap_number_at_session_time(laps, incident_seconds)
                if lap_number is not None:
                    out.at[idx, "lap"] = str(lap_number)
                    if provenance:
                        provenance.record(
                            incident_id,
                            "lap",
                            "fastf1",
                            value=str(lap_number),
                            round=round_num,
                            session_offset_seconds=incident_seconds,
                        )
            except Exception:  # noqa: BLE001
                pass

        if incident_seconds is not None and incident_seconds >= 0:
            if _should_write(row.get("flag"), fill_gaps_only) and messages is not None:
                try:
                    flag = flag_at_session_time(messages, incident_seconds, tolerance)
                    if flag:
                        out.at[idx, "flag"] = flag
                        if provenance:
                            rc_row = nearest_race_control_row(messages, incident_seconds, tolerance)
                            message = str(rc_row.get("Message", "")) if rc_row is not None else ""
                            provenance.record(
                                incident_id,
                                "flag",
                                "fastf1",
                                value=flag,
                                message=message,
                                round=round_num,
                            )
                except Exception:  # noqa: BLE001
                    pass

            if _should_write(row.get("sector"), fill_gaps_only) and messages is not None:
                try:
                    sector = sector_at_session_time(messages, incident_seconds, tolerance)
                    if sector:
                        out.at[idx, "sector"] = sector
                        if provenance:
                            provenance.record(
                                incident_id,
                                "sector",
                                "fastf1",
                                value=sector,
                                round=round_num,
                            )
                except Exception:  # noqa: BLE001
                    pass

            if (
                driver_numbers
                and _should_write(row.get("positions_of_involved parties"), fill_gaps_only)
                and laps is not None
            ):
                try:
                    position_values = positions_at_session_time(
                        laps,
                        driver_numbers,
                        incident_seconds,
                        tolerance,
                    )
                    if any(position_values):
                        joined = ",".join(position_values)
                        out.at[idx, "positions_of_involved parties"] = joined
                        if provenance:
                            provenance.record(
                                incident_id,
                                "positions_of_involved parties",
                                "fastf1",
                                value=joined,
                                round=round_num,
                            )
                except Exception:  # noqa: BLE001
                    pass

        try:
            if _should_write(row.get("safety_car"), fill_gaps_only) and messages is not None:
                if incident_seconds is not None and incident_seconds >= 0:
                    state = safety_car_at_session_time(messages, incident_seconds, tolerance)
                    if state:
                        out.at[idx, "safety_car"] = state
                        if provenance:
                            provenance.record(
                                incident_id,
                                "safety_car",
                                "fastf1",
                                value=state,
                                round=round_num,
                            )
                else:
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
