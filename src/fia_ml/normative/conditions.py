"""Evaluate condition trees against an incident row."""

from __future__ import annotations

from typing import Any

FACT_TEXT_FIELDS = ("fact", "Fact", "fact_text", "reason", "investigation")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _normalize_token(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip().lower()


def normalize_session(value: object) -> str:
    """Map dataset session strings to canonical race / qualifying / sprint tokens."""
    text = _normalize_token(value)
    if not text:
        return ""
    if "qualifying" in text:
        return "qualifying"
    if "sprint" in text:
        return "sprint"
    if text == "race" or text.startswith("race "):
        return "race"
    if "practice" in text:
        return "practice"
    return text


def _fact_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in FACT_TEXT_FIELDS:
        value = row.get(field)
        if not _is_missing(value):
            parts.append(str(value))
    return " ".join(parts)


def _to_float(value: object) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _compare_eq(actual: object, expected: object) -> bool:
    if isinstance(expected, str):
        return _normalize_token(actual) == _normalize_token(expected)
    if _is_missing(actual) and _is_missing(expected):
        return True
    return actual == expected


def _match_leaf(key: str, expected: Any, row: dict[str, Any]) -> bool:
    if key == "default":
        return expected is True

    if key == "fact_contains_any":
        if not isinstance(expected, list):
            return False
        haystack = _fact_text(row).lower()
        if not haystack:
            return False
        return any(str(needle).lower() in haystack for needle in expected)

    if key == "session_in":
        if not isinstance(expected, list):
            return False
        session = normalize_session(row.get("session"))
        allowed = {normalize_session(item) for item in expected}
        return session in allowed

    actual = row.get(key)

    if isinstance(expected, dict):
        if "eq" in expected:
            return _compare_eq(actual, expected["eq"])
        if "in" in expected:
            allowed = expected["in"]
            if isinstance(allowed, list) and all(isinstance(v, str) for v in allowed):
                return _normalize_token(actual) in {_normalize_token(v) for v in allowed}
            return actual in allowed
        if "contains" in expected:
            if _is_missing(actual):
                return False
            return str(expected["contains"]).lower() in str(actual).lower()
        if "gte" in expected:
            actual_num = _to_float(actual)
            expected_num = _to_float(expected["gte"])
            if actual_num != actual_num or expected_num != expected_num:
                return False
            return actual_num >= expected_num
        if "lt" in expected:
            actual_num = _to_float(actual)
            expected_num = _to_float(expected["lt"])
            if actual_num != actual_num or expected_num != expected_num:
                return False
            return actual_num < expected_num
        return False

    if isinstance(expected, list):
        if all(isinstance(item, str) for item in expected):
            return _normalize_token(actual) in {_normalize_token(item) for item in expected}
        return actual in expected

    return _compare_eq(actual, expected)


def evaluate_conditions(conditions: dict[str, Any], row: dict[str, Any]) -> bool:
    """Return whether a condition tree matches the given incident row."""
    if not conditions:
        return True

    if conditions.get("default") is True:
        return True

    if "and" in conditions:
        children = conditions["and"]
        if not isinstance(children, list) or not all(
            evaluate_conditions(child, row) for child in children
        ):
            return False

    if "or" in conditions:
        children = conditions["or"]
        if not isinstance(children, list) or not any(
            evaluate_conditions(child, row) for child in children
        ):
            return False

    leaf_items = (
        (key, value)
        for key, value in conditions.items()
        if key not in {"and", "or", "default"}
    )
    return all(_match_leaf(key, value, row) for key, value in leaf_items)
