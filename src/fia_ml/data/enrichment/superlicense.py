"""Rolling superlicense points before each incident within a season."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta
from fia_ml.data.enrichment.provenance import EnrichmentProvenance
from fia_ml.data.enrichment.timestamp import parse_incident_date


def _split_multi(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_superlicense_points(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def incident_sort_key(
    row: pd.Series,
    meta_row: dict[str, Any],
) -> tuple[int, date, str]:
    round_value = row.get("round", "")
    try:
        round_num = int(float(round_value)) if not is_blank(round_value) else 999
    except ValueError:
        round_num = 999

    incident_date = parse_incident_date(str(meta_row.get("date", "")))
    if incident_date is None:
        incident_date = date.max

    return (round_num, incident_date, str(row.get("incident_id", "")))


def primary_penalized_driver(driver_ids: list[str]) -> str | None:
    for driver_id in driver_ids:
        if not driver_id.startswith("car_"):
            return driver_id
    return None


def compute_superlicense_before_values(
    df: pd.DataFrame,
    meta: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if df.empty:
        return {}

    ordered_indices = sorted(
        df.index.tolist(),
        key=lambda index: incident_sort_key(df.loc[index], meta.get(str(df.loc[index, "incident_id"]), {})),
    )

    driver_totals: dict[str, int] = {}
    before_by_incident: dict[str, str] = {}

    for index in ordered_indices:
        row = df.loc[index]
        incident_id = str(row.get("incident_id", ""))
        driver_ids = _split_multi(row.get("drivers", ""))
        if not driver_ids:
            continue

        before_values: list[str] = []
        for driver_id in driver_ids:
            if driver_id.startswith("car_"):
                before_values.append("")
            else:
                before_values.append(str(driver_totals.get(driver_id, 0)))

        before_by_incident[incident_id] = ",".join(before_values)

        points_added = parse_superlicense_points(row.get("superlicense_points_added"))
        if points_added > 0:
            penalized_driver = primary_penalized_driver(driver_ids)
            if penalized_driver:
                driver_totals[penalized_driver] = driver_totals.get(penalized_driver, 0) + points_added

    return before_by_incident


def enrich_superlicense_points(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    provenance: EnrichmentProvenance | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if cfg.enrichment_settings.superlicense_rolling_by != "driver":
        return df

    out = df.copy()
    meta = load_meta(cfg)
    before_values = compute_superlicense_before_values(out, meta)

    for index, row in out.iterrows():
        incident_id = str(row.get("incident_id", ""))
        value = before_values.get(incident_id)
        if value is None:
            continue
        out.at[index, "superlicense_points_before_incident"] = value
        if provenance and value != "":
            provenance.record(
                incident_id,
                "superlicense_points_before_incident",
                "corpus_rolling",
                value=value,
                season=cfg.season,
            )

    return out
