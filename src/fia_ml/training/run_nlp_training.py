#!/usr/bin/env python3
"""CLI entry point for NLP (spec V2) model training."""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fia_ml.training.evaluate_nlp import evaluate_nlp
from fia_ml.training.nlp_config import NlpTrainingConfig
from fia_ml.training.train_nlp import prepare_nlp_data, train_nlp


class Stage(str, Enum):
    ALL = "all"
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIA penalty predictor — NLP text model training")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "bert.yaml",
        help="Path to bert.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=[s.value for s in Stage],
        default=Stage.ALL.value,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Run late-fusion experiment vs V1 tabular (evaluate stage only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = NlpTrainingConfig.from_yaml(args.config)
    stage = Stage(args.stage)
    results: dict[str, object] = {"stage": stage.value}

    if stage in {Stage.ALL, Stage.PREPARE}:
        results["prepare"] = prepare_nlp_data(cfg)

    if stage in {Stage.ALL, Stage.TRAIN}:
        results["train"] = train_nlp(cfg)

    if stage in {Stage.ALL, Stage.EVALUATE}:
        if args.fusion:
            results["fusion"] = {
                "skipped": True,
                "reason": "Late fusion is not implemented yet; run evaluate without --fusion",
            }
        results["evaluate"] = evaluate_nlp(cfg)

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
