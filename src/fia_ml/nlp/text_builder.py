"""Assemble leakage-safe training text from parsed FIA interim documents."""

from __future__ import annotations

from typing import Any

from fia_ml.data.parsing import normalize_text
from fia_ml.training.nlp_config import ALLOWED_TEXT_PROFILES

LEAKY_SECTION_TAGS = ("DECISION", "REASON")
LEAKY_FIELD_KEYS = ("decision", "reason")


def _field(fields: dict[str, Any], key: str) -> str:
    value = fields.get(key, "")
    return normalize_text(str(value)) if value not in (None, "") else ""


def _section(tag: str, body: str) -> str:
    body = normalize_text(body)
    if not body:
        return ""
    return f"[{tag}] {body}"


def _metadata_lines(doc: dict[str, Any], fields: dict[str, Any]) -> list[str]:
    session = _field(fields, "session")
    event = normalize_text(str(doc.get("event", "")))
    document_type = normalize_text(str(doc.get("document_type", "")))
    lines: list[str] = []
    if session:
        lines.append(_section("SESSION", session))
    if event:
        lines.append(_section("EVENT", event))
    if document_type:
        lines.append(_section("DOCTYPE", document_type))
    return lines


def build_text(
    doc: dict[str, Any],
    *,
    profile: str = "fact_offence",
    include_metadata: bool = True,
) -> str:
    """Build model input text from an interim extracted-document payload."""
    if profile not in ALLOWED_TEXT_PROFILES:
        raise ValueError(
            f"Invalid profile '{profile}'. Allowed: {sorted(ALLOWED_TEXT_PROFILES)}"
        )

    fields = doc.get("parsed_fields", {})
    if not isinstance(fields, dict):
        fields = {}

    parts: list[str] = []
    if include_metadata:
        parts.extend(_metadata_lines(doc, fields))

    fact = _field(fields, "fact")
    offence = _field(fields, "offence")
    decision = _field(fields, "decision")
    reason = _field(fields, "reason")

    if profile == "fact_only":
        if fact:
            parts.append(_section("FACT", fact))
    elif profile == "fact_offence":
        if fact:
            parts.append(_section("FACT", fact))
        if offence:
            parts.append(_section("OFFENCE", offence))
    elif profile == "leaky_full":
        if fact:
            parts.append(_section("FACT", fact))
        if offence:
            parts.append(_section("OFFENCE", offence))
        if decision:
            parts.append(_section("DECISION", decision))
        if reason:
            parts.append(_section("REASON", reason))
    else:
        raise ValueError(f"Unhandled profile: {profile}")

    text = "\n".join(part for part in parts if part).strip()
    violations = leakage_violations(text, fields, profile=profile)
    if violations:
        raise ValueError(f"Text profile '{profile}' produced leakage: {violations}")
    return text


def leakage_violations(
    text: str,
    fields: dict[str, Any],
    *,
    profile: str,
) -> list[str]:
    """Return human-readable leakage violations for safe profiles."""
    if profile not in {"fact_only", "fact_offence"}:
        return []

    violations: list[str] = []
    normalized = normalize_text(text).lower()

    for tag in LEAKY_SECTION_TAGS:
        if f"[{tag}]" in text.upper():
            violations.append(f"contains [{tag}] section")

    for key in LEAKY_FIELD_KEYS:
        value = _field(fields, key)
        if not value:
            continue
        if len(value) >= 12 and value.lower() in normalized:
            violations.append(f"contains {key} field body")

    return violations


def truncate_text(
    text: str,
    max_length: int,
    tokenizer: Any | None = None,
) -> str:
    """Truncate text to fit within a token budget (tokenizer-aware when provided)."""
    if not text or max_length <= 0:
        return ""
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_attention_mask=False,
        )
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

    char_limit = max(max_length * 4, 1)
    if len(text) <= char_limit:
        return text
    head = char_limit // 2
    tail = max(char_limit - head - 3, 0)
    return f"{text[:head]}...{text[-tail:]}" if tail else text[:char_limit]
