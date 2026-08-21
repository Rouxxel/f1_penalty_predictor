#!/usr/bin/env python3
"""CLI entry point for the FIA dataset generation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.pipeline import Stage, run_pipeline_for_seasons


def main() -> int:
    parser = argparse.ArgumentParser(description="FIA dataset generation pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to data.yaml config file",
    )
    parser.add_argument(
        "--stage",
        choices=[s.value for s in Stage],
        default=Stage.ALL.value,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        dest="seasons",
        help="Season year to process (repeatable). Default: all seasons in config.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    results = run_pipeline_for_seasons(cfg, Stage(args.stage), args.seasons)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
