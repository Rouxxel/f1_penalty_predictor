"""Enrich incident rows from manually curated reference JSON files."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import (
    is_blank,
    load_meta,
    map_event_to_round,
    resolve_team_id,
    slugify_nationality,
)
from fia_ml.data.reference_data import build_event_name_to_circuit_map, load_circuits, load_drivers, load_seasons, load_teams


def _driver_years_in_sport(driver_id: str, season_year: int, drivers: dict[str, Any]) -> str:
    profile = drivers.get(driver_id)
    if not profile:
        return ""
    debut = profile.get("debut")
    if debut in (None, ""):
        return ""
    try:
        return str(max(season_year - int(debut), 0))
    except ValueError:
        return ""


def _fill_driver_fields(
    driver_ids: list[str],
    season_year: int,
    drivers: dict[str, Any],
    driver_pos: dict[str, int],
    driver_pts: dict[str, int],
    driver_team: dict[str, str],
    team_pos: dict[str, int],
    team_pts: dict[str, int],
    teams: dict[str, Any],
) -> dict[str, str]:
    nationalities: list[str] = []
    d_standings: list[str] = []
    d_points: list[str] = []
    c_standings: list[str] = []
    c_points: list[str] = []
    team_slugs: list[str] = []
    years: list[str] = []

    for driver_id in driver_ids:
        if driver_id.startswith("car_"):
            continue

        profile = drivers.get(driver_id, {})
        nationality = profile.get("nationality", "")
        if nationality:
            nationalities.append(slugify_nationality(str(nationality)))

        if driver_id in driver_pos:
            d_standings.append(str(driver_pos[driver_id]))
        if driver_id in driver_pts:
            d_points.append(str(driver_pts[driver_id]))

        team_id = driver_team.get(driver_id, "")
        if team_id:
            canonical_team = resolve_team_id(team_id, teams)
            team_slugs.append(canonical_team)
            if canonical_team in team_pos:
                c_standings.append(str(team_pos[canonical_team]))
            if canonical_team in team_pts:
                c_points.append(str(team_pts[canonical_team]))

        years.append(_driver_years_in_sport(driver_id, season_year, drivers))

    return {
        "nationalities": ",".join(nationalities),
        "driver_standings": ",".join(d_standings),
        "driver_points": ",".join(d_points),
        "respective_teams": ",".join(team_slugs),
        "construct_standings": ",".join(c_standings),
        "construct_points": ",".join(c_points),
        "years_in_sport": ",".join(years),
    }


def enrich_with_reference(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Fill dataset columns from verified local reference files."""
    if df.empty:
        return df

    circuits = load_circuits(cfg)
    drivers = load_drivers(cfg)
    teams = load_teams(cfg)
    seasons = load_seasons(cfg)
    season_data = seasons.get(str(cfg.season))
    if not season_data:
        return df

    event_to_circuit = build_event_name_to_circuit_map(circuits)
    calendar: list[str] = list(season_data.get("calendar_in_order", []))
    driver_standings: list[dict[str, Any]] = list(season_data.get("drivers.standings_order", []))
    team_standings: list[dict[str, Any]] = list(season_data.get("teams.standings_order", []))

    driver_pos = {entry["id"]: index + 1 for index, entry in enumerate(driver_standings)}
    driver_pts = {entry["id"]: entry.get("total_points", "") for entry in driver_standings}
    driver_team = {entry["id"]: entry.get("team", "") for entry in driver_standings}
    team_pos = {entry["id"]: index + 1 for index, entry in enumerate(team_standings)}
    team_pts = {entry["id"]: entry.get("total_points", "") for entry in team_standings}
    top_four = ",".join(entry["id"] for entry in driver_standings[:4])

    out = df.copy()
    meta = load_meta(cfg)
    season_year = int(cfg.season)

    out["rounds"] = str(season_data.get("rounds", ""))
    out["num_teams"] = str(len(team_standings))

    for idx, row in out.iterrows():
        incident_id = str(row.get("incident_id", ""))
        meta_row = meta.get(incident_id, {})
        event = str(meta_row.get("event", ""))
        round_num, circuit_slug = map_event_to_round(event, calendar, event_to_circuit)

        if round_num is not None and is_blank(row.get("round")):
            out.at[idx, "round"] = str(round_num)

        if circuit_slug:
            circuit_meta = circuits.get(circuit_slug, {})
            if is_blank(row.get("circuit")):
                out.at[idx, "circuit"] = circuit_slug
            if is_blank(row.get("country")):
                out.at[idx, "country"] = str(circuit_meta.get("country", ""))
            if is_blank(row.get("first_season")) and circuit_meta.get("first_season") not in (None, ""):
                out.at[idx, "first_season"] = str(circuit_meta.get("first_season"))

            session = str(row.get("session", "")).lower()
            total_laps = circuit_meta.get("total_laps")
            if session == "race" and total_laps not in (None, "") and is_blank(row.get("full_laps")):
                out.at[idx, "full_laps"] = str(total_laps)

        if is_blank(row.get("current_top_4_drivers")) and top_four:
            out.at[idx, "current_top_4_drivers"] = top_four

        driver_ids = [part.strip() for part in str(row.get("drivers", "")).split(",") if part.strip()]
        if not driver_ids:
            continue

        filled = _fill_driver_fields(
            driver_ids,
            season_year,
            drivers,
            driver_pos,
            driver_pts,
            driver_team,
            team_pos,
            team_pts,
            teams,
        )
        for column, value in filled.items():
            if value and is_blank(row.get(column)):
                out.at[idx, column] = value

    return out
