"""Ergast / Jolpica API enrichment with local caching."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
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


def load_meta(cfg: PipelineConfig) -> dict[str, dict[str, Any]]:
    meta_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.meta.json"
    if not meta_path.exists():
        return {}
    rows = sio.read_json(meta_path)
    return {row["incident_id"]: row for row in rows}


def enrich_with_ergast(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    meta = load_meta(cfg)
    calendar = load_season_calendar(cfg)
    total_rounds = len(calendar)
    out["rounds"] = str(total_rounds)

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
            out.at[idx, "round"] = str(round_num)
            if not row.get("circuit"):
                out.at[idx, "circuit"] = race_info["circuit_id"]
            if not row.get("country"):
                out.at[idx, "country"] = race_info["country"]

        standings_round = max(round_num - 1, 0)
        driver_standings = load_driver_standings(cfg, standings_round)
        constructor_standings = load_constructor_standings(cfg, standings_round)
        results = load_race_results(cfg, round_num)
        car_map = build_car_to_driver_map(results)

        car_number = str(meta_row.get("car_number", ""))
        if car_number and car_number in car_map:
            out.at[idx, "drivers"] = car_map[car_number]

        driver_ids = [d.strip() for d in str(out.at[idx, "drivers"]).split(",") if d.strip()]
        nat_list: list[str] = []
        d_standings: list[str] = []
        d_points: list[str] = []
        c_standings: list[str] = []
        c_points: list[str] = []
        years_list: list[str] = []
        sl_points_list: list[str] = []

        driver_lookup = {slugify_driver_id(s["Driver"]["driverId"]): s for s in driver_standings}
        constructor_lookup = {
            s["Constructor"]["constructorId"].lower(): s for s in constructor_standings
        }

        for driver_slug in driver_ids:
            if driver_slug.startswith("car_"):
                continue
            standing = driver_lookup.get(driver_slug)
            if standing:
                d_standings.append(str(standing["position"]))
                d_points.append(str(standing["points"]))
                nat = standing["Driver"].get("nationality", "")
                nat_list.append(nat.lower().replace(" ", "_"))
                constructor_id = standing["Constructors"][0]["constructorId"].lower()
                c_st = constructor_lookup.get(constructor_id)
                if c_st:
                    c_standings.append(str(c_st["position"]))
                    c_points.append(str(c_st["points"]))

            sl_points_list.append("0")
            years_list.append("")

        if driver_ids:
            out.at[idx, "nationalities"] = ",".join(nat_list)
            out.at[idx, "driver_standings"] = ",".join(d_standings)
            out.at[idx, "driver_points"] = ",".join(d_points)
            out.at[idx, "construct_standings"] = ",".join(c_standings)
            out.at[idx, "construct_points"] = ",".join(c_points)
            out.at[idx, "years_in_sport"] = ",".join(years_list)
            out.at[idx, "superlicense_points_before_incident"] = ",".join(sl_points_list)

        top4 = [slugify_driver_id(s["Driver"]["driverId"]) for s in driver_standings[:4]]
        out.at[idx, "current_top_4_drivers"] = ",".join(top4)

    return out
