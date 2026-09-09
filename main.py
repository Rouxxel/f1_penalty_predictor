#!/usr/bin/env python3
"""Run the full FIA penalty predictor pipeline from the project root.

Bare ``python main.py`` prints help and exits without running anything.
Pass ``--run`` to execute, or ``--dry-run`` to preview the plan only.

Examples:

    # See the plan without running anything
    python main.py --dry-run

    # Full run (downloads PDFs from FIA — slow, may hit WAF)
    python main.py --run

    # Recommended when PDFs already exist under data/raw/fia/
    python main.py --run --skip-download

    # Partial runs
    python main.py --run --dataset-only
    python main.py --run --from v1              # skip dataset if CSVs/parquet exist
    python main.py --run --skip-v2 --skip-nlp   # dataset + V1 + normative only
    python main.py --run --no-fusion            # skip NLP fusion step

Phase order: dataset -> V1 tabular -> V2 tabular -> NLP -> normative.
Implementation: ``src/fia_ml/orchestration/run_all.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("USE_TF", "0")

from fia_ml.orchestration.run_all import main

if __name__ == "__main__":
    raise SystemExit(main())
