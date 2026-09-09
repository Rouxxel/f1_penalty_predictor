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

    # Single phase (prerequisites checked before run)
    python main.py --run --only v1
    python main.py --run --only v2
    python main.py --run --only nlp --no-fusion

    # Multi-phase from a midpoint onward (--from runs this phase + all later)
    python main.py --run --from v1              # v1 -> v2 -> nlp -> normative
    python main.py --run --dataset-only
    python main.py --run --skip-v2 --skip-nlp   # dataset + v1 + normative only

Phase order: dataset -> V1 tabular -> V2 tabular -> NLP -> normative.
Normative requires V1 prepare (incidents.parquet), not NLP.
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
