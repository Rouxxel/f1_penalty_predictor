"""Orchestrate dataset generation pipeline stages."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd

from fia_ml.data.config import PipelineConfig
from fia_ml.data.download import download_season
from fia_ml.data.enrichment import enrich_with_ergast, enrich_with_fastf1, enrich_with_reference
from fia_ml.data.incident_builder import build_from_interim, build_incidents
from fia_ml.data.parsing import parse_all_documents
from fia_ml.data.validation import validate_and_export
from fia_ml.paths import PROJECT_ROOT
from fia_ml.utils import secure_file_io as sio


class Stage(str, Enum):
    ALL = "all"
    DOWNLOAD = "download"
    PARSE = "parse"
    BUILD = "build"
    ENRICH = "enrich"
    VALIDATE = "validate"


def run_pipeline_for_seasons(
    cfg: PipelineConfig,
    stage: Stage = Stage.ALL,
    seasons: list[int] | None = None,
) -> dict:
    season_cfgs = cfg.iter_seasons(seasons)
    if len(season_cfgs) == 1:
        return run_pipeline(season_cfgs[0], stage)

    results: dict = {"stage": stage.value, "seasons": {}}
    for season_cfg in season_cfgs:
        results["seasons"][str(season_cfg.season)] = run_pipeline(season_cfg, stage)
    return results


def run_pipeline(cfg: PipelineConfig, stage: Stage = Stage.ALL) -> dict:
    sio.set_allowed_root(PROJECT_ROOT)
    results: dict = {"stage": stage.value, "season": cfg.season}

    entries = None
    parsed_docs = None
    df = None

    if stage in {Stage.ALL, Stage.DOWNLOAD}:
        entries = download_season(cfg)
        results["downloaded_documents"] = len(entries)

    if stage in {Stage.ALL, Stage.PARSE}:
        if entries is None:
            manifest = cfg.path("raw_fia") / str(cfg.season) / "manifest.json"
            raw_entries = sio.read_json(manifest)
            from fia_ml.data.download import DocumentEntry

            entries = [DocumentEntry(**item) for item in raw_entries]
        parsed_docs = parse_all_documents(entries, cfg)
        results["parsed_documents"] = len(parsed_docs)

    if stage in {Stage.ALL, Stage.BUILD}:
        if parsed_docs is None:
            df = build_from_interim(cfg)
        else:
            df = build_incidents(parsed_docs, cfg)
        results["raw_rows"] = len(df)

    if stage in {Stage.ALL, Stage.ENRICH, Stage.VALIDATE}:
        if df is None:
            raw_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.csv"
            df = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()

    if stage in {Stage.ALL, Stage.ENRICH}:
        df = enrich_with_reference(df, cfg)
        df = enrich_with_ergast(df, cfg, fill_gaps_only=True)
        df = enrich_with_fastf1(df, cfg, fill_gaps_only=True)
        raw_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.csv"
        df.to_csv(raw_path, index=False)
        results["enriched_rows"] = len(df)

    if stage in {Stage.ALL, Stage.VALIDATE}:
        if df is None:
            raw_path = cfg.path("csv_out") / f"raw_incidents_{cfg.season}.csv"
            df = pd.read_csv(raw_path)
        processed_path, review_path, quality = validate_and_export(df, cfg)
        results["processed_csv"] = str(processed_path)
        results["review_csv"] = str(review_path)
        results["quality"] = quality

    return results
