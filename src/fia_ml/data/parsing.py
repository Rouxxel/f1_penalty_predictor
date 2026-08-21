"""PDF text extraction and FIA decision field parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fia_ml.data.config import PipelineConfig
from fia_ml.data.download import DocumentEntry
from fia_ml.paths import PROJECT_ROOT, ensure_dir
from fia_ml.utils import secure_file_io as sio


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufffd", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> str:
    data = sio.read_bytes(pdf_path)
    try:
        import pymupdf

        doc = pymupdf.open(stream=data, filetype="pdf")
        parts = [page.get_text() for page in doc]
        doc.close()
        return normalize_text("\n".join(parts))
    except ImportError:
        import io

        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            parts = [page.extract_text() or "" for page in pdf.pages]
        return normalize_text("\n".join(parts))


def classify_document_type(title: str) -> str:
    lowered = title.lower()
    if re.search(r"\bsummons\b", lowered):
        return "summons"
    if re.search(r"\binfringement\b", lowered):
        return "infringement"
    if re.search(r"\boffence\b", lowered):
        return "offence"
    if re.search(r"\bdecision\b", lowered):
        return "decision"
    return "unknown"


def normalize_session(value: str) -> str:
    lowered = value.lower().strip()
    mapping = {
        "race": "race",
        "qualifying": "qualifying",
        "sprint": "sprint",
        "practice 1": "practice",
        "practice 2": "practice",
        "practice 3": "practice",
        "free practice 1": "practice",
        "free practice 2": "practice",
        "free practice 3": "practice",
    }
    for key, normalized in mapping.items():
        if key in lowered:
            return normalized
    return lowered or ""


def _extract_field(text: str, label: str) -> str:
    if label.lower() == "reason":
        match = re.search(r"Reason\s*\n(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            chunk = match.group(1)
            chunk = re.split(r"\nCompetitors are reminded", chunk, maxsplit=1)[0]
            return normalize_text(chunk)
    pattern = rf"{re.escape(label)}\s*\n(.+?)(?=\n(?:No / Driver|Competitor|Time|Session|Fact|Offence|Decision|Reason|Document|Date)\s*\n|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return normalize_text(match.group(1))


def parse_car_driver(line: str) -> tuple[str, str]:
    match = re.match(r"(\d+)\s*[-–]\s*(.+)", line.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "", line.strip()


def parse_opponent_cars(fact: str) -> list[str]:
    return sorted(set(re.findall(r"\bcar\s+(\d+)\b", fact, re.IGNORECASE)))


def parse_turn_number(fact: str) -> str:
    match = re.search(r"\bturn\s+(\d+)\b", fact, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_penalty_points(decision: str) -> str:
    match = re.search(r"(\d+)\s+penalty\s+points?", decision, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_penalty_label(decision: str) -> str:
    decision = normalize_text(decision)
    if not decision:
        return ""
    lowered = decision.lower()
    if "no further action" in lowered or "no action" in lowered:
        return "no_further_action"
    if "reprimand" in lowered:
        return "reprimand"
    if "warning" in lowered:
        return "warning"
    if "disqualified" in lowered or "disqualification" in lowered:
        return "disqualification"
    if "grid" in lowered and "place" in lowered:
        match = re.search(r"(\d+)\s+place", lowered)
        if match:
            return f"{match.group(1)}_place_grid_drop"
        return "grid_penalty"
    if "time penalty" in lowered or re.search(r"\b\d+\s+second", lowered):
        match = re.search(r"(\d+)\s+second", lowered)
        if match:
            return f"{match.group(1)}s_time_penalty"
        return "time_penalty"
    if "drive through" in lowered:
        return "drive_through"
    return decision.lower().replace(" ", "_")[:80]


def parse_driver_at_fault(reason: str, fact: str) -> str:
    combined = normalize_text(f"{reason} {fact}").lower()
    if "no driver was wholly or predominantly to blame" in combined:
        return "none"
    if "wholly to blame" in combined or "predominantly to blame" in combined:
        driver_match = re.search(r"car\s+(\d+)", reason, re.IGNORECASE)
        if driver_match:
            return f"car_{driver_match.group(1)}"
    return ""


def parse_fia_document(text: str, title: str) -> dict[str, Any]:
    text = text.replace("\u00a0", " ").replace("\ufffd", " ")
    driver_line = _extract_field(text, "No / Driver")
    car_number, driver_name = parse_car_driver(driver_line.split("\n")[0] if driver_line else "")
    fact = _extract_field(text, "Fact")
    offence = _extract_field(text, "Offence")
    if not offence:
        offence = _extract_field(text, "Infringement")
    decision = _extract_field(text, "Decision")
    reason = _extract_field(text, "Reason")
    session = normalize_session(_extract_field(text, "Session"))
    competitor = _extract_field(text, "Competitor")
    document_no = _extract_field(text, "Document")
    if not document_no:
        document_no = _extract_field(text, "No")

    opponent_cars = parse_opponent_cars(fact)
    if car_number in opponent_cars:
        opponent_cars = [c for c in opponent_cars if c != car_number]

    fields = {
        "document_no": document_no,
        "date": _extract_field(text, "Date"),
        "time": _extract_field(text, "Time"),
        "grand_prix": "",
        "car_number": car_number,
        "driver_name": driver_name,
        "competitor": competitor,
        "session": session,
        "fact": fact,
        "offence": offence,
        "decision": decision,
        "reason": reason,
        "opponent_car_numbers": opponent_cars,
        "turn_number": parse_turn_number(fact),
        "penalty": parse_penalty_label(decision),
        "superlicense_points_added": parse_penalty_points(decision),
        "mentioned_article": offence,
        "driver_at_fault": parse_driver_at_fault(reason, fact),
    }

    gp_match = re.search(r"(\d{4})\s+(.+?\sGRAND PRIX)", text, re.IGNORECASE)
    if gp_match:
        fields["grand_prix"] = gp_match.group(2).strip()

    required = ["session", "fact", "decision"] if classify_document_type(title) == "decision" else ["session", "fact"]
    present = sum(1 for key in required if fields.get(key))
    confidence = present / max(len(required), 1)
    errors = [key for key in required if not fields.get(key)]

    return {
        "parsed_fields": fields,
        "parse_confidence": round(confidence, 3),
        "parse_errors": errors,
    }


def parse_document(entry: DocumentEntry, cfg: PipelineConfig) -> dict[str, Any]:
    pdf_path = PROJECT_ROOT / entry.local_path
    raw_text = extract_pdf_text(pdf_path)
    parsed = parse_fia_document(raw_text, entry.title)

    return {
        "document_id": entry.document_id,
        "source_pdf": entry.local_path,
        "document_type": classify_document_type(entry.title),
        "event": entry.event,
        "event_slug": entry.event_slug,
        "season": entry.season,
        "title": entry.title,
        "raw_text": raw_text,
        **parsed,
    }


def parse_all_documents(entries: list[DocumentEntry], cfg: PipelineConfig) -> list[dict[str, Any]]:
    out_dir = ensure_dir(cfg.path("interim_docs") / str(cfg.season))
    parsed_docs: list[dict[str, Any]] = []
    for entry in entries:
        doc = parse_document(entry, cfg)
        sio.write_json(out_dir / f"{entry.document_id}.json", doc)
        parsed_docs.append(doc)
    return parsed_docs
