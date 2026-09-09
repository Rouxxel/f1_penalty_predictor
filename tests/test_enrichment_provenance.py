import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.enrichment.provenance import EnrichmentProvenance, enrichment_meta_path, write_enrichment_meta


def test_enrichment_provenance_record_and_summary():
    prov = EnrichmentProvenance()
    prov.record("inc_1", "lap", "openf1", value="37", match_error_seconds=0.8)
    prov.record("inc_1", "driver_standings", "ergast", value="3,7", round_used=7)
    prov.add_timestamp_provenance(
        {
            "inc_1": {
                "session_offset_seconds": 1800.0,
                "time_parse_method": "session_elapsed",
                "time_parse_confidence": 0.9,
            }
        }
    )

    summary = prov.summary()
    assert summary["incidents"] == 1
    assert summary["fields_recorded"] == 3
    assert summary["by_source"]["openf1"] == 1
    assert summary["by_source"]["ergast"] == 1
    assert summary["by_source"]["timestamp"] == 1

    entries = prov.to_list()
    assert entries[0]["incident_id"] == "inc_1"
    assert entries[0]["fields"]["lap"]["source"] == "openf1"


def test_write_enrichment_meta(monkeypatch):
    cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(2025)
    prov = EnrichmentProvenance()
    prov.record("inc_1", "flag", "openf1", value="yellow")
    written: dict[str, object] = {}

    def _capture_write(path, payload):
        written["path"] = path
        written["payload"] = payload

    monkeypatch.setattr(
        "fia_ml.data.enrichment.provenance.enrichment_meta_path",
        lambda _cfg: ROOT / "data" / "interim" / "enrichment_meta" / "2025.json",
    )
    monkeypatch.setattr("fia_ml.data.enrichment.provenance.sio.write_json", _capture_write)

    summary = write_enrichment_meta(cfg, prov, meta={})
    assert summary["incidents"] == 1
    assert written["payload"][0]["fields"]["flag"]["source"] == "openf1"
