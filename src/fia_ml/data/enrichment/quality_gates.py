"""Enrichment quality targets and reporting."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.common import is_blank, load_meta
from fia_ml.data.enrichment.provenance import enrichment_meta_path
from fia_ml.data.enrichment.superlicense import (
    incident_sort_key,
    parse_superlicense_points,
    primary_penalized_driver,
)
from fia_ml.data.enrichment.timestamp import timestamp_audit
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.utils import secure_file_io as sio


def _split_multi(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _is_filled(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


def column_fill_rate(df: pd.DataFrame, column: str, mask: pd.Series | None = None) -> float:
    if column not in df.columns:
        return 0.0
    subset = df[mask] if mask is not None else df
    if subset.empty:
        return 0.0
    filled = subset[column].map(_is_filled).sum()
    return round(float(filled) / len(subset), 4)


def incidents_with_prior_superlicense(df: pd.DataFrame, meta: dict[str, dict[str, Any]]) -> set[str]:
    ordered_indices = sorted(
        df.index.tolist(),
        key=lambda index: incident_sort_key(df.loc[index], meta.get(str(df.loc[index, "incident_id"]), {})),
    )
    driver_totals: dict[str, int] = {}
    prior_incidents: set[str] = set()

    for index in ordered_indices:
        row = df.loc[index]
        incident_id = str(row.get("incident_id", ""))
        driver_ids = [driver for driver in _split_multi(row.get("drivers", "")) if not driver.startswith("car_")]
        if any(driver_totals.get(driver_id, 0) > 0 for driver_id in driver_ids):
            prior_incidents.add(incident_id)

        points_added = parse_superlicense_points(row.get("superlicense_points_added"))
        if points_added > 0:
            penalized_driver = primary_penalized_driver(_split_multi(row.get("drivers", "")))
            if penalized_driver:
                driver_totals[penalized_driver] = driver_totals.get(penalized_driver, 0) + points_added

    return prior_incidents


def ergast_standings_provenance_rate(
    df: pd.DataFrame,
    enrichment_meta: list[dict[str, Any]],
) -> float:
    rows_with_round = df[~df["round"].map(is_blank)] if "round" in df.columns else df.iloc[0:0]
    if rows_with_round.empty:
        return 0.0

    provenance_by_id = {entry["incident_id"]: entry.get("fields", {}) for entry in enrichment_meta}
    matched = 0
    for _, row in rows_with_round.iterrows():
        incident_id = str(row.get("incident_id", ""))
        fields = provenance_by_id.get(incident_id, {})
        standing = fields.get("driver_standings", {})
        if standing.get("source") == "ergast" and standing.get("round_used") is not None:
            matched += 1
    return round(float(matched) / len(rows_with_round), 4)


def load_enrichment_meta(cfg: PipelineConfig) -> list[dict[str, Any]]:
    path = enrichment_meta_path(cfg)
    if not path.exists():
        return []
    payload = sio.read_json(path)
    return payload if isinstance(payload, list) else []


def evaluate_quality_gates(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    enrichment_meta: list[dict[str, Any]] | None = None,
    timestamp_stats: dict[str, Any] | None = None,
    provenance_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = load_meta(cfg)
    enrichment_meta = enrichment_meta if enrichment_meta is not None else load_enrichment_meta(cfg)
    targets = cfg.enrichment_settings.quality_targets

    race_mask = df["session"].astype(str).str.lower() == "race" if "session" in df.columns else pd.Series(True, index=df.index)
    drivers_mask = df["drivers"].map(lambda value: len(_split_multi(value)) > 0) if "drivers" in df.columns else pd.Series(True, index=df.index)

    metrics: dict[str, Any] = {
        "lap_race_fill_rate": column_fill_rate(df, "lap", race_mask),
        "flag_race_fill_rate": column_fill_rate(df, "flag", race_mask),
        "positions_with_drivers_fill_rate": column_fill_rate(df, "positions_of_involved parties", drivers_mask),
        "sector_fill_rate": column_fill_rate(df, "sector"),
        "superlicense_before_fill_rate": column_fill_rate(df, "superlicense_points_before_incident"),
        "ergast_standings_provenance_rate": ergast_standings_provenance_rate(df, enrichment_meta),
        "timestamp_audit": timestamp_stats or timestamp_audit(meta),
        "provenance_summary": provenance_summary or {},
    }

    prior_sl_ids = incidents_with_prior_superlicense(df, meta)
    if prior_sl_ids:
        prior_mask = df["incident_id"].astype(str).isin(prior_sl_ids)
        metrics["superlicense_before_given_prior_sl_rate"] = column_fill_rate(
            df,
            "superlicense_points_before_incident",
            prior_mask,
        )
        metrics["rows_with_prior_superlicense"] = int(prior_mask.sum())
    else:
        metrics["superlicense_before_given_prior_sl_rate"] = 0.0
        metrics["rows_with_prior_superlicense"] = 0

    gate_results: list[dict[str, Any]] = []

    def add_gate(gate_id: str, actual: float, target: float, passed: bool, detail: str = "") -> None:
        gate_results.append(
            {
                "id": gate_id,
                "actual": actual,
                "target": target,
                "passed": passed,
                "detail": detail,
            }
        )

    lap_target = float(targets.get("lap_race_min_fill_rate", 0.5))
    add_gate("lap_race", metrics["lap_race_fill_rate"], lap_target, metrics["lap_race_fill_rate"] > lap_target)

    flag_target = float(targets.get("flag_race_min_fill_rate", 0.5))
    add_gate("flag_race", metrics["flag_race_fill_rate"], flag_target, metrics["flag_race_fill_rate"] > flag_target)

    positions_target = float(targets.get("positions_with_drivers_min_fill_rate", 0.4))
    add_gate(
        "positions_with_drivers",
        metrics["positions_with_drivers_fill_rate"],
        positions_target,
        metrics["positions_with_drivers_fill_rate"] > positions_target,
    )

    sector_targets = targets.get("sector_min_fill_rate_by_season", {})
    sector_target = float(sector_targets.get(str(cfg.season), sector_targets.get("default", 0.5)))
    add_gate("sector", metrics["sector_fill_rate"], sector_target, metrics["sector_fill_rate"] >= sector_target)

    standings_target = float(targets.get("ergast_round_n_minus_1_rate", 0.9))
    add_gate(
        "ergast_standings_round_n_minus_1",
        metrics["ergast_standings_provenance_rate"],
        standings_target,
        metrics["ergast_standings_provenance_rate"] >= standings_target,
    )

    sl_target = float(targets.get("superlicense_before_given_prior_sl_rate", 0.8))
    if metrics["rows_with_prior_superlicense"] > 0:
        add_gate(
            "superlicense_before_given_prior_sl",
            metrics["superlicense_before_given_prior_sl_rate"],
            sl_target,
            metrics["superlicense_before_given_prior_sl_rate"] >= sl_target,
        )

    passed_count = sum(1 for gate in gate_results if gate["passed"])
    return {
        "season": cfg.season,
        "metrics": metrics,
        "gates": gate_results,
        "gates_passed": passed_count,
        "gates_total": len(gate_results),
        "all_passed": passed_count == len(gate_results),
    }


def write_enrichment_reports(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    timestamp_stats: dict[str, Any] | None = None,
    provenance_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enrichment_meta = load_enrichment_meta(cfg)
    report = evaluate_quality_gates(
        df,
        cfg,
        enrichment_meta=enrichment_meta,
        timestamp_stats=timestamp_stats,
        provenance_summary=provenance_summary,
    )

    tables_dir = ensure_dir(cfg.path("reports"))
    sio.write_json(tables_dir / f"enrichment_report_{cfg.season}.json", report)

    report_date = date.today().isoformat()
    model_reports_dir = ensure_dir(PROJECT_ROOT / "reports" / "model_reports")
    markdown_path = model_reports_dir / f"enrichment_improvement_{report_date}.md"
    markdown_path.write_text(render_enrichment_markdown(report), encoding="utf-8")
    report["enrichment_report_path"] = str(tables_dir / f"enrichment_report_{cfg.season}.json")
    report["enrichment_markdown_path"] = str(markdown_path)
    return report


def render_enrichment_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Enrichment quality report — season {report['season']}",
        "",
        f"Gates passed: **{report['gates_passed']}/{report['gates_total']}**",
        "",
        "## Metrics",
        "",
    ]
    for key, value in report["metrics"].items():
        if isinstance(value, dict):
            lines.append(f"- **{key}**: `{value}`")
        else:
            lines.append(f"- **{key}**: `{value}`")

    lines.extend(["", "## Gates", ""])
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(
            f"- [{status}] `{gate['id']}` — actual `{gate['actual']}` vs target `{gate['target']}`"
        )
    lines.append("")
    return "\n".join(lines)


def merge_enrichment_into_quality(quality: dict[str, Any], enrichment_report: dict[str, Any]) -> dict[str, Any]:
    quality["enrichment"] = enrichment_report
    return quality
