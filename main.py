#!/usr/bin/env python3
"""Run the full FIA penalty predictor pipeline from the project root.

Bare ``python main.py`` prints help and exits without running anything.
Pass ``--run`` to execute, or ``--dry-run`` to preview the plan only.
If a phase already completed, you are prompted ``Run it again? [Y/N]``.
Use ``--force`` to skip prompts (e.g. after backfilling 2020-2024 data).

Examples:

    # See the plan without running anything
    python main.py --dry-run

    # Full run (downloads PDFs from FIA — slow, may hit WAF)
    python main.py --run

    # Recommended when PDFs already exist under data/raw/fia/
    python main.py --run --skip-download

    # Re-train everything after new seasons (no prompts)
    python main.py --run --skip-download --force

    # Single phase (prerequisites checked before run)
    python main.py --run --only v1
    python main.py --run --only v2
    python main.py --run --only nlp --no-fusion

    # Multi-phase from a midpoint onward (--from runs this phase + all later)
    python main.py --run --from v1              # v1 -> v2 -> nlp -> normative
    python main.py --run --dataset-only
    python main.py --run --skip-v2 --skip-nlp   # dataset + v1 + normative only

Phase order:

    dataset -> V1 tabular -> V2 tabular -> NLP -> normative

Prerequisites (dataset assumed for all downstream phases):

    - V1: ``dataset/csv/processed_{season}.csv`` per configured season
    - V2: ``data/processed/incidents.parquet`` (V1 prepare)
    - NLP: ``processed_{season}.csv``, interim JSON under
      ``data/interim/extracted_documents/{season}/``; fusion also needs
      ``ml_models/xgboost/predictions_val.json`` (V1) — use ``--no-fusion`` to skip
    - normative: ``data/processed/incidents.parquet`` only (not NLP); uses V1
      predictions when present, but they are optional

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
