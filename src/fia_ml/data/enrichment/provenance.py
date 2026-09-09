"""Track enrichment field provenance in an interim JSON sidecar."""

from __future__ import annotations

from typing import Any

from fia_ml.data.config import PipelineConfig
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.utils import secure_file_io as sio


class EnrichmentProvenance:
    def __init__(self) -> None:
        self._incidents: dict[str, dict[str, Any]] = {}

    def record(self, incident_id: str, field: str, source: str, **details: Any) -> None:
        if not incident_id or not field or not source:
            return
        entry = self._incidents.setdefault(
            str(incident_id),
            {"incident_id": str(incident_id), "fields": {}},
        )
        payload: dict[str, Any] = {"source": source}
        for key, value in details.items():
            if value is not None and value != "":
                payload[key] = value
        entry["fields"][field] = payload

    def add_timestamp_provenance(self, meta: dict[str, dict[str, Any]]) -> None:
        for incident_id, meta_row in meta.items():
            offset = meta_row.get("session_offset_seconds")
            if offset is not None:
                self.record(
                    incident_id,
                    "session_offset_seconds",
                    "timestamp",
                    value=offset,
                    method=meta_row.get("time_parse_method"),
                    confidence=meta_row.get("time_parse_confidence"),
                )
            lap_hint = meta_row.get("lap_hint")
            if lap_hint not in (None, ""):
                self.record(
                    incident_id,
                    "lap",
                    "timestamp",
                    value=str(lap_hint),
                    via="lap_hint",
                )

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._incidents.values())

    def summary(self) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        field_counts: dict[str, int] = {}
        for entry in self._incidents.values():
            for field, payload in entry.get("fields", {}).items():
                field_counts[field] = field_counts.get(field, 0) + 1
                source = str(payload.get("source", "unknown"))
                source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "incidents": len(self._incidents),
            "fields_recorded": sum(field_counts.values()),
            "by_source": source_counts,
            "by_field": field_counts,
        }


def enrichment_meta_path(cfg: PipelineConfig):
    return ensure_dir(PROJECT_ROOT / "data" / "interim" / "enrichment_meta") / f"{cfg.season}.json"


def write_enrichment_meta(
    cfg: PipelineConfig,
    provenance: EnrichmentProvenance,
    *,
    meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if meta is not None:
        provenance.add_timestamp_provenance(meta)
    payload = provenance.to_list()
    sio.write_json(enrichment_meta_path(cfg), payload)
    return provenance.summary()
