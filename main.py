#!/usr/bin/env python3
"""Run the full FIA penalty predictor pipeline from the project root.

Example:
    python main.py --dry-run                # preview only (safe default)
    python main.py --run --skip-download    # execute (PDFs already on disk)
    python main.py --run --from v1          # ML only (dataset artifacts exist)

Bare `python main.py` prints help and exits without running anything.
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
