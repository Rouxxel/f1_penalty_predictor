import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.schema import SCHEMA_COLUMNS
from fia_ml.data.validation import compute_derived_fields, validate_schema


def test_validate_schema_columns():
    df = pd.DataFrame(columns=SCHEMA_COLUMNS)
    assert validate_schema(df) == []


def test_compute_derived_fields():
    df = pd.DataFrame(
        [
            {
                "drivers": "a,b",
                "lap": "27",
                "full_laps": "53",
            }
        ]
    )
    out = compute_derived_fields(df)
    assert out.at[0, "num_drivers"] == "2"
    assert out.at[0, "lap_remaining"] == "26"
