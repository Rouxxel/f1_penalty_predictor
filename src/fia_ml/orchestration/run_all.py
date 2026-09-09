"""Run dataset, tabular ML, NLP, and normative pipelines in sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

from fia_ml.data.config import PipelineConfig
from fia_ml.data.pipeline import Stage as DatasetStage
from fia_ml.data.pipeline import run_pipeline_for_seasons
from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.run_normative import run_compare_report, run_predict
from fia_ml.paths import (
    DEFAULT_NLP_CONFIG,
    DEFAULT_NORMATIVE_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_TRAINING_V2_CONFIG,
    PROJECT_ROOT,
)
from fia_ml.training.config import TrainingConfig
from fia_ml.training.evaluate_nlp import evaluate_nlp
from fia_ml.training.fusion_nlp import run_fusion_nlp
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.training.pipeline import Stage as TrainingStage
from fia_ml.training.pipeline import run_training
from fia_ml.training.train_nlp import prepare_nlp_data, train_nlp


class PipelinePhase(str, Enum):
    DATASET = "dataset"
    V1 = "v1"
    V2 = "v2"
    NLP = "nlp"
    NORMATIVE = "normative"


PHASE_ORDER = [
    PipelinePhase.DATASET,
    PipelinePhase.V1,
    PipelinePhase.V2,
    PipelinePhase.NLP,
    PipelinePhase.NORMATIVE,
]


def _default_seasons(v1_cfg: TrainingConfig) -> list[int]:
    seasons = v1_cfg.inputs.get("seasons")
    if seasons:
        return [int(s) for s in seasons]
    return [2019, 2025]


def _run_dataset(
    *,
    data_config: Path,
    enrichment_config: Path,
    seasons: list[int],
    skip_download: bool,
) -> dict[str, Any]:
    cfg = PipelineConfig.from_yaml(data_config, enrichment_config_path=enrichment_config)
    if skip_download:
        results: dict[str, Any] = {"skip_download": True, "seasons": {}}
        for stage in (
            DatasetStage.PARSE,
            DatasetStage.BUILD,
            DatasetStage.ENRICH,
            DatasetStage.VALIDATE,
        ):
            stage_result = run_pipeline_for_seasons(cfg, stage, seasons)
            results["seasons"][stage.value] = stage_result
        return results
    return run_pipeline_for_seasons(cfg, DatasetStage.ALL, seasons)


def _run_v1(config: Path) -> dict[str, Any]:
    cfg = TrainingConfig.from_yaml(config)
    return run_training(cfg, TrainingStage.ALL)


def _run_v2(config: Path) -> dict[str, Any]:
    cfg = TrainingConfig.from_yaml(config)
    return run_training(cfg, TrainingStage.ALL)


def _run_nlp(config: Path, *, fusion: bool) -> dict[str, Any]:
    cfg = NlpTrainingConfig.from_yaml(config)
    results: dict[str, Any] = {}
    results["prepare"] = prepare_nlp_data(cfg)
    results["train"] = train_nlp(cfg)
    results["evaluate"] = evaluate_nlp(cfg)
    if fusion or bool(cfg.fusion.get("enabled")):
        results["fusion"] = run_fusion_nlp(cfg)
    return results


def _run_normative(
    config: Path,
    *,
    ml_predictions: Path | None,
) -> dict[str, Any]:
    cfg = NormativeConfig.from_yaml(config)
    rules_path = cfg.rules_file_path()
    input_path = cfg.resolve_path("incidents").resolve()
    output_path = cfg.resolve_path("output").resolve()
    report_dir = cfg.resolve_path("reports_dir").resolve()

    predict_result = run_predict(
        cfg,
        rules_path=rules_path,
        input_path=input_path,
        output_path=output_path,
    )
    compare_result = run_compare_report(
        cfg,
        rules_path=rules_path,
        input_path=output_path,
        report_dir=report_dir,
        ml_predictions_path=ml_predictions.resolve() if ml_predictions else None,
        output_path=output_path,
    )
    return {"predict": predict_result, "compare": compare_result}


def _phases_to_run(
    *,
    start_from: PipelinePhase | None,
    skip_v2: bool,
    skip_nlp: bool,
    skip_normative: bool,
    dataset_only: bool,
) -> list[PipelinePhase]:
    if dataset_only:
        return [PipelinePhase.DATASET]

    phases = list(PHASE_ORDER)
    if skip_v2:
        phases = [p for p in phases if p != PipelinePhase.V2]
    if skip_nlp:
        phases = [p for p in phases if p != PipelinePhase.NLP]
    if skip_normative:
        phases = [p for p in phases if p != PipelinePhase.NORMATIVE]

    if start_from is None:
        return phases
    if start_from not in phases:
        raise ValueError(f"Cannot start from {start_from.value}: phase is skipped or unknown")
    start_index = phases.index(start_from)
    return phases[start_index:]


def run_full_pipeline(
    *,
    seasons: list[int] | None = None,
    data_config: Path = PROJECT_ROOT / "configs" / "data.yaml",
    enrichment_config: Path = PROJECT_ROOT / "configs" / "enrichment.yaml",
    v1_config: Path = DEFAULT_TRAINING_CONFIG,
    v2_config: Path = DEFAULT_TRAINING_V2_CONFIG,
    nlp_config: Path = DEFAULT_NLP_CONFIG,
    normative_config: Path = DEFAULT_NORMATIVE_CONFIG,
    skip_download: bool = False,
    skip_v2: bool = False,
    skip_nlp: bool = False,
    skip_normative: bool = False,
    nlp_fusion: bool = True,
    dataset_only: bool = False,
    start_from: PipelinePhase | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute configured pipeline phases and return a summary dict."""
    v1_cfg = TrainingConfig.from_yaml(v1_config)
    resolved_seasons = seasons or _default_seasons(v1_cfg)
    phases = _phases_to_run(
        start_from=start_from,
        skip_v2=skip_v2,
        skip_nlp=skip_nlp,
        skip_normative=skip_normative,
        dataset_only=dataset_only,
    )

    plan = {
        "seasons": resolved_seasons,
        "phases": [phase.value for phase in phases],
        "skip_download": skip_download,
        "nlp_fusion": nlp_fusion,
    }
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    os.environ.setdefault("USE_TF", "0")

    results: dict[str, Any] = {"status": "ok", "plan": plan, "phases": {}}
    for phase in phases:
        started = time.perf_counter()
        print(f"\n=== [{phase.value}] starting ===", flush=True)
        if phase == PipelinePhase.DATASET:
            phase_result = _run_dataset(
                data_config=data_config,
                enrichment_config=enrichment_config,
                seasons=resolved_seasons,
                skip_download=skip_download,
            )
        elif phase == PipelinePhase.V1:
            phase_result = _run_v1(v1_config)
        elif phase == PipelinePhase.V2:
            phase_result = _run_v2(v2_config)
        elif phase == PipelinePhase.NLP:
            phase_result = _run_nlp(nlp_config, fusion=nlp_fusion)
        elif phase == PipelinePhase.NORMATIVE:
            ml_predictions = PROJECT_ROOT / "ml_models" / "xgboost" / "predictions_val.json"
            phase_result = _run_normative(
                normative_config,
                ml_predictions=ml_predictions if ml_predictions.exists() else None,
            )
        else:
            raise ValueError(f"Unhandled phase: {phase}")

        elapsed = time.perf_counter() - started
        results["phases"][phase.value] = {
            "elapsed_seconds": round(elapsed, 1),
            "result": phase_result,
        }
        print(f"=== [{phase.value}] done ({elapsed:.1f}s) ===", flush=True)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIA penalty predictor — run all pipelines end-to-end",
    )
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        dest="seasons",
        help="Season year(s) for dataset pipeline (default: from configs/xgboost.yaml)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Dataset: skip FIA download; run parse -> build -> enrich -> validate only",
    )
    parser.add_argument(
        "--skip-v2",
        action="store_true",
        help="Skip V2 tabular feature engineering and training",
    )
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip NLP text model training",
    )
    parser.add_argument(
        "--skip-normative",
        action="store_true",
        help="Skip normative rule engine and deviation report",
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="Skip NLP late-fusion vs V1 tabular",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Run only the dataset generation pipeline",
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=[phase.value for phase in PipelinePhase],
        help="Start from this phase (assumes earlier outputs already exist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without running pipelines",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute pipelines (required; bare 'python main.py' does not run anything)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.run and not args.dry_run:
        parser.print_help()
        print(
            "\nNo action taken. Pass --run to execute pipelines or --dry-run to preview the plan.",
            file=sys.stderr,
        )
        return 2

    start_from = PipelinePhase(args.start_from) if args.start_from else None
    results = run_full_pipeline(
        seasons=args.seasons,
        skip_download=args.skip_download,
        skip_v2=args.skip_v2,
        skip_nlp=args.skip_nlp,
        skip_normative=args.skip_normative,
        nlp_fusion=not args.no_fusion,
        dataset_only=args.dataset_only,
        start_from=start_from,
        dry_run=args.dry_run,
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
