#!/usr/bin/env python3
"""Thin wrapper for V1 model training CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fia_ml.training.run_training import main

if __name__ == "__main__":
    raise SystemExit(main())
