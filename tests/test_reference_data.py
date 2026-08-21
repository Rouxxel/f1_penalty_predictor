import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.reference_data import load_circuits, load_reference
from fia_ml.utils import secure_file_io as sio
from fia_ml.utils.secure_file_io import ReadOnlyPathError


@pytest.fixture
def cfg():
    return PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml")


def test_load_circuits_has_event_map(cfg):
    circuits = load_circuits(cfg)
    assert "event_to_circuit" in circuits
    assert circuits["event_to_circuit"]["Italian Grand Prix"] == "monza"
    assert "monza" in circuits


def test_reference_dir_is_read_only(cfg):
    circuits_path = cfg.path("reference") / "circuits.json"
    with pytest.raises(ReadOnlyPathError, match="read-only"):
        sio.write_json(circuits_path, {"test": True})


def test_load_reference_returns_expected_keys(cfg):
    refs = load_reference(cfg)
    assert "circuits" in refs
    assert "incident_types" in refs
