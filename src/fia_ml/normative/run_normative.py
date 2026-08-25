#!/usr/bin/env python3
"""CLI for the normative stewarding rule engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fia_ml.normative.config import NormativeConfig
from fia_ml.normative.rules_loader import load_rules
from fia_ml.paths import DEFAULT_NORMATIVE_CONFIG, PROJECT_ROOT, ensure_dir
from fia_ml.utils import secure_file_io as sio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FIA penalty predictor — normative rule engine"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_NORMATIVE_CONFIG,
        help="Path to normative.yaml runtime config",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=None,
        help="Override path to normative_rules.yaml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input incidents parquet (default from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output incidents_with_normative.parquet (default from config)",
    )
    parser.add_argument(
        "--validate-rules",
        action="store_true",
        help="Validate rule YAML schema and exit",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare normative outcomes vs FIA labels",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for deviation reports (default from config)",
    )
    parser.add_argument(
        "--ml-predictions",
        type=Path,
        default=None,
        help="Optional ML predictions JSON for three-way comparison",
    )
    return parser


def validate_rules(rules_path: Path) -> dict[str, Any]:
    loaded = load_rules(rules_path)
    return {
        "status": "ok",
        "rules_path": str(rules_path.relative_to(PROJECT_ROOT)),
        "version": loaded.version,
        "rule_count": loaded.rule_count,
        "content_hash": loaded.content_hash,
        "assumptions": list(loaded.document.assumptions),
        "rule_ids": [rule.id for rule in loaded.document.rules],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = NormativeConfig.from_yaml(args.config)
    rules_path = args.rules or cfg.rules_file_path()

    if args.validate_rules or not any(
        [args.input, args.output, args.compare, args.report_dir, args.ml_predictions]
    ):
        result = validate_rules(rules_path)
        print(json.dumps(result, indent=2))
        return 0

    if args.compare or args.report_dir or args.ml_predictions:
        raise NotImplementedError(
            "Comparison and reporting are not implemented yet."
        )

    raise NotImplementedError(
        "Batch normative prediction is not implemented yet."
    )


if __name__ == "__main__":
    raise SystemExit(main())
