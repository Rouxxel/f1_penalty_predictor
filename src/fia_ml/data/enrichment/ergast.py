"""Ergast / Jolpica API enrichment fallback for gaps not covered by reference data."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta, resolve_team_id, slugify_nationality
from fia_ml.data.enrichment.provenance import EnrichmentProvenance
from fia_ml.data.reference_data import load_teams
from fia_ml.paths import ensure_dir
from fia_ml.utils import secure_file_io as sio


def _fetch_json(url: str, cfg: PipelineConfig) -> dict[str, Any]:
    scraper = cfg.scraper
    req = urllib.request.Request(
        url,
        headers={"User-Agent": scraper.get("user_agent", "f1-penalty-predictor/1.0")},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_with_cache(relative_path: str, cfg: PipelineConfig) -> dict[str, Any]:
    cache_root = ensure_dir(cfg.path("ergast_cache") / str(cfg.season))
    cache_file = cache_root / relative_path.replace("/", "_")
    if cache_file.exists():
        return sio.read_json(cache_file)

    bases = [
        cfg.enrichment.get("ergast_base_url", "https://api.jolpi.ca/ergast/f1").rstrip("/"),
        cfg.enrichment.get("ergast_fallback_url", "https://ergast.com/api/f1").rstrip("/"),
    ]
    last_error: Exception | None = None
    for base in bases:
        url = f"{base}/{relative_path}"
        try:
            payload = _fetch_json(url, cfg)
            sio.write_json(cache_file, payload)
            time.sleep(0.5)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Ergast fetch failed for {relative_path}: {last_error}") from last_error


def load_season_calendar(cfg: PipelineConfig) -> list[dict[str, Any]]:
    payload = fetch_with_cache(f"{cfg.season}.json", cfg)
    races = payload["MRData"]["RaceTable"]["Races"]
    calendar: list[dict[str, Any]] = []
    for race in races:
        circuit = race["Circuit"]
        loc = circuit.get("Location", {})
        calendar.append(
            {
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "circuit_id": circuit["circuitId"],
                "country": loc.get("country", "").lower().replace(" ", "_"),
            }
        )
    return calendar


def load_driver_standings(cfg: PipelineConfig, round_num: int) -> list[dict[str, Any]]:
    if round_num < 1:
        return []
    payload = fetch_with_cache(f"{cfg.season}/{round_num}/driverStandings.json", cfg)
    standings = payload["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return []
    return standings[0]["DriverStandings"]


def load_constructor_standings(cfg: PipelineConfig, round_num: int) -> list[dict[str, Any]]:
    if round_num < 1:
        return []
    payload = fetch_with_cache(f"{cfg.season}/{round_num}/constructorStandings.json", cfg)
    standings = payload["MRData"]["StandingsTable"]["StandingsLists"]
    if not standings:
        return []
    return standings[0]["ConstructorStandings"]


def load_race_results(cfg: PipelineConfig, round_num: int) -> list[dict[str, Any]]:
    payload = fetch_with_cache(f"{cfg.season}/{round_num}/results.json", cfg)
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    return races[0].get("Results", [])


def slugify_driver_id(driver_id: str) -> str:
    return driver_id.lower().replace("-", "_")


def slugify_constructor_id(constructor_id: str) -> str:
    return constructor_id.lower().replace("-", "_")


STANDINGS_COLUMNS = frozenset(
    {
        "driver_standings",
        "driver_points",
        "construct_standings",
        "construct_points",
        "nationalities",
        "respective_teams",
        "current_top_4_drivers",
    }
)


def _build_driver_standing_lookup(driver_standings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for standing in driver_standings:
        driver = standing["Driver"]
        keys = {
            slugify_driver_id(driver["driverId"]),
            driver.get("familyName", "").lower().replace(" ", "_"),
            driver.get("code", "").lower(),
        }
        for key in keys:
            if key:
                lookup[key] = standing
    return lookup


def _standing_position(standing: dict[str, Any]) -> str:
    value = standing.get("position", standing.get("positionText", ""))
    return str(value) if value not in (None, "") else ""


def resolve_driver_standing(
    driver_slug: str,
    driver_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if driver_slug in driver_lookup:
        return driver_lookup[driver_slug]
    tail = driver_slug.split("_")[-1]
    if tail in driver_lookup:
        return driver_lookup[tail]
    return None


def build_per_driver_standing_fields(
    driver_ids: list[str],
    driver_standings: list[dict[str, Any]],
    constructor_standings: list[dict[str, Any]],
    teams: dict[str, Any],
) -> dict[str, str]:
    driver_lookup = _build_driver_standing_lookup(driver_standings)
    constructor_lookup = {
        slugify_constructor_id(item["Constructor"]["constructorId"]): item for item in constructor_standings
    }

    nationalities: list[str] = []
    team_slugs: list[str] = []
    d_standings: list[str] = []
    d_points: list[str] = []
    c_standings: list[str] = []
    c_points: list[str] = []

    for driver_slug in driver_ids:
        if driver_slug.startswith("car_"):
            nationalities.append("")
            team_slugs.append("")
            d_standings.append("")
            d_points.append("")
            c_standings.append("")
            c_points.append("")
            continue

        standing = resolve_driver_standing(driver_slug, driver_lookup)
        if not standing:
            nationalities.append("")
            team_slugs.append("")
            d_standings.append("")
            d_points.append("")
            c_standings.append("")
            c_points.append("")
            continue

        driver = standing["Driver"]
        nationalities.append(slugify_nationality(str(driver.get("nationality", ""))))
        d_standings.append(_standing_position(standing))
        d_points.append(str(standing.get("points", "")))

        constructor_id = slugify_constructor_id(standing["Constructors"][0]["constructorId"])
        team_slug = resolve_team_id(constructor_id, teams)
        team_slugs.append(team_slug)
        constructor_standing = constructor_lookup.get(constructor_id)
        if constructor_standing:
            c_standings.append(_standing_position(constructor_standing))
            c_points.append(str(constructor_standing.get("points", "")))
        else:
            c_standings.append("")
            c_points.append("")

    return {
        "nationalities": ",".join(nationalities),
        "respective_teams": ",".join(team_slugs),
        "driver_standings": ",".join(d_standings),
        "driver_points": ",".join(d_points),
        "construct_standings": ",".join(c_standings),
        "construct_points": ",".join(c_points),
    }


def _should_write_field(
    column: str,
    row_value: Any,
    *,
    fill_gaps_only: bool,
    overwrite_standings: bool,
) -> bool:
    if overwrite_standings and column in STANDINGS_COLUMNS:
        return True
    return is_blank(row_value) or not fill_gaps_only


def map_event_to_round(event: str, calendar: list[dict[str, Any]]) -> int | None:
    for race in calendar:
        if race["race_name"].lower() == event.lower():
            return race["round"]
    normalized = event.lower().replace("grand prix", "").strip()
    for race in calendar:
        if normalized in race["race_name"].lower():
            return race["round"]
    return None


def build_car_to_driver_map(results: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for result in results:
        driver = result["Driver"]
        number = str(result.get("number", ""))
        if number:
            mapping[number] = slugify_driver_id(driver["driverId"])
    return mapping


def enrich_with_ergast(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    fill_gaps_only: bool = True,
    provenance: EnrichmentProvenance | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if not cfg.enrichment.get("ergast_fallback_enabled", True):
        return df

    overwrite_standings = (
        cfg.enrichment_settings.overwrite_reference_standings
        and cfg.enrichment_settings.prefer_ergast_round_n_minus_1
    )
    teams = load_teams(cfg)

    out = df.copy()
    meta = load_meta(cfg)
    calendar = load_season_calendar(cfg)
    total_rounds = len(calendar)
    if fill_gaps_only and is_blank(out["rounds"].iloc[0] if len(out) else ""):
        out["rounds"] = str(total_rounds)

    if fill_gaps_only and is_blank(out["num_teams"].iloc[0] if len(out) else ""):
        constructors_seen: set[str] = set()
        for round_num in range(1, total_rounds + 1):
            for item in load_constructor_standings(cfg, round_num):
                constructors_seen.add(item["Constructor"]["constructorId"])
        out["num_teams"] = str(len(constructors_seen) or "")

    for idx, row in out.iterrows():
        incident_id = row.get("incident_id", "")
        meta_row = meta.get(incident_id, {})
        event = meta_row.get("event", "")
        round_num = map_event_to_round(event, calendar)
        if not round_num:
            continue

        race_info = next((r for r in calendar if r["round"] == round_num), None)
        if race_info:
            if is_blank(row.get("round")) or not fill_gaps_only:
                out.at[idx, "round"] = str(round_num)
                if provenance:
                    provenance.record(incident_id, "round", "ergast", value=str(round_num))
            if is_blank(row.get("circuit")) or not fill_gaps_only:
                out.at[idx, "circuit"] = race_info["circuit_id"]
                if provenance:
                    provenance.record(incident_id, "circuit", "ergast", value=race_info["circuit_id"])
            if is_blank(row.get("country")) or not fill_gaps_only:
                out.at[idx, "country"] = race_info["country"]
                if provenance:
                    provenance.record(incident_id, "country", "ergast", value=race_info["country"])

        standings_round = max(round_num - 1, 0)
        driver_standings = load_driver_standings(cfg, standings_round)
        constructor_standings = load_constructor_standings(cfg, standings_round)
        results = load_race_results(cfg, round_num)
        car_map = build_car_to_driver_map(results)

        car_number = str(meta_row.get("car_number", ""))
        if car_number and car_number in car_map and is_blank(row.get("drivers")):
            out.at[idx, "drivers"] = car_map[car_number]

        driver_ids = [d.strip() for d in str(out.at[idx, "drivers"]).split(",") if d.strip()]
        if driver_ids:
            standing_fields = build_per_driver_standing_fields(
                driver_ids,
                driver_standings,
                constructor_standings,
                teams,
            )
            for column, value in standing_fields.items():
                if _should_write_field(
                    column,
                    row.get(column),
                    fill_gaps_only=fill_gaps_only,
                    overwrite_standings=overwrite_standings,
                ):
                    out.at[idx, column] = value
                    if provenance and overwrite_standings and value:
                        provenance.record(
                            incident_id,
                            column,
                            "ergast",
                            value=value,
                            round_used=standings_round,
                        )

        top4 = [slugify_driver_id(s["Driver"]["driverId"]) for s in driver_standings[:4]]
        if _should_write_field(
            "current_top_4_drivers",
            row.get("current_top_4_drivers"),
            fill_gaps_only=fill_gaps_only,
            overwrite_standings=overwrite_standings,
        ):
            out.at[idx, "current_top_4_drivers"] = ",".join(top4)
            if provenance and overwrite_standings and top4:
                provenance.record(
                    incident_id,
                    "current_top_4_drivers",
                    "ergast",
                    value=",".join(top4),
                    round_used=standings_round,
                )

    return out
