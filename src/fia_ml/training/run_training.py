#!/usr/bin/env python3
"""CLI entry point for V1 model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fia_ml.training.config import TrainingConfig
from fia_ml.training.pipeline import Stage, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIA penalty predictor — model training (V1 and V2)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to xgboost.yaml (default: configs/xgboost.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=[s.value for s in Stage],
        default=Stage.ALL.value,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Explicit processed CSV path(s). Default: from config inputs.seasons or glob.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainingConfig.from_yaml(args.config)
    results = run_training(cfg, Stage(args.stage), input_paths=args.inputs)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
