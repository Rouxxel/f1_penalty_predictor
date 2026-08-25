"""YAML rule schema types and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_PENALTY_DETAILS = frozenset(
    {
        "no_action",
        "warning",
        "reprimand",
        "5s",
        "10s",
        "grid_drop",
        "dsq",
        "licence_points",
        "manual_review",
    }
)

ALLOWED_SEVERITIES = frozenset({0, 1, 2})

SUPPORTED_CONDITION_OPERATORS = frozenset(
    {
        "eq",
        "in",
        "contains",
        "fact_contains_any",
        "session_in",
        "gte",
        "lt",
        "and",
        "or",
        "default",
    }
)


@dataclass(frozen=True)
class RuleOutcome:
    penalty_detail: str
    penalty_severity: int
    cited_regulation: str | None = None


@dataclass(frozen=True)
class NormativeRule:
    id: str
    priority: int
    conditions: dict[str, Any]
    outcome: RuleOutcome
    reason: str | None = None
    continue_after_match: bool = False


@dataclass(frozen=True)
class NormativeRulesDocument:
    version: str
    description: str
    assumptions: tuple[str, ...]
    rules: tuple[NormativeRule, ...]


class RulesValidationError(ValueError):
    """Raised when normative_rules.yaml fails schema validation."""


def _parse_outcome(raw: dict[str, Any], *, context: str) -> RuleOutcome:
    if not isinstance(raw, dict):
        raise RulesValidationError(f"{context}: outcome must be a mapping")
    detail = raw.get("penalty_detail")
    severity = raw.get("penalty_severity")
    if detail not in ALLOWED_PENALTY_DETAILS:
        raise RulesValidationError(
            f"{context}: invalid penalty_detail '{detail}' "
            f"(allowed: {sorted(ALLOWED_PENALTY_DETAILS)})"
        )
    if severity not in ALLOWED_SEVERITIES:
        raise RulesValidationError(
            f"{context}: penalty_severity must be 0, 1, or 2 (got {severity!r})"
        )
    cited = raw.get("cited_regulation")
    if cited is not None and not isinstance(cited, str):
        raise RulesValidationError(f"{context}: cited_regulation must be a string")
    return RuleOutcome(
        penalty_detail=str(detail),
        penalty_severity=int(severity),
        cited_regulation=cited,
    )


def _validate_conditions(conditions: dict[str, Any], *, context: str) -> None:
    if not isinstance(conditions, dict):
        raise RulesValidationError(f"{context}: conditions must be a mapping")
    if not conditions:
        raise RulesValidationError(f"{context}: conditions must not be empty")

    for key, value in conditions.items():
        if key in {"and", "or"}:
            if not isinstance(value, list) or not value:
                raise RulesValidationError(f"{context}: '{key}' must be a non-empty list")
            for idx, child in enumerate(value):
                if not isinstance(child, dict):
                    raise RulesValidationError(
                        f"{context}: '{key}[{idx}]' must be a condition mapping"
                    )
                _validate_conditions(child, context=f"{context}.{key}[{idx}]")
            continue

        if key == "default":
            if value is not True:
                raise RulesValidationError(f"{context}: 'default' must be true when present")
            continue

        if key == "fact_contains_any":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise RulesValidationError(
                    f"{context}: fact_contains_any must be a list of strings"
                )
            continue

        if key == "session_in":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise RulesValidationError(f"{context}: session_in must be a list of strings")
            continue

        if isinstance(value, dict):
            unknown_ops = set(value) - {"eq", "gte", "lt", "in"}
            if unknown_ops:
                raise RulesValidationError(
                    f"{context}: unsupported operators on '{key}': {sorted(unknown_ops)}"
                )
            continue

        if isinstance(value, (str, int, float, bool)):
            continue

        raise RulesValidationError(
            f"{context}: unsupported condition value for '{key}': {type(value).__name__}"
        )


def parse_rules_document(raw: dict[str, Any]) -> NormativeRulesDocument:
    """Parse and validate a normative rules YAML document."""
    if not isinstance(raw, dict):
        raise RulesValidationError("Rule file root must be a mapping")

    version = raw.get("version")
    description = raw.get("description")
    assumptions = raw.get("assumptions")
    rules_raw = raw.get("rules")

    if not version or not isinstance(version, str):
        raise RulesValidationError("'version' must be a non-empty string")
    if not description or not isinstance(description, str):
        raise RulesValidationError("'description' must be a non-empty string")
    if not isinstance(assumptions, list) or not assumptions:
        raise RulesValidationError("'assumptions' must be a non-empty list of strings")
    if not all(isinstance(item, str) and item.strip() for item in assumptions):
        raise RulesValidationError("'assumptions' entries must be non-empty strings")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise RulesValidationError("'rules' must be a non-empty list")

    rules: list[NormativeRule] = []
    seen_ids: set[str] = set()

    for idx, rule_raw in enumerate(rules_raw):
        context = f"rules[{idx}]"
        if not isinstance(rule_raw, dict):
            raise RulesValidationError(f"{context}: each rule must be a mapping")

        rule_id = rule_raw.get("id")
        priority = rule_raw.get("priority")
        conditions = rule_raw.get("conditions")
        outcome_raw = rule_raw.get("outcome")

        if not rule_id or not isinstance(rule_id, str):
            raise RulesValidationError(f"{context}: 'id' must be a non-empty string")
        if rule_id in seen_ids:
            raise RulesValidationError(f"Duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)

        if not isinstance(priority, int):
            raise RulesValidationError(f"{context} ({rule_id}): 'priority' must be an integer")
        if not isinstance(conditions, dict):
            raise RulesValidationError(f"{context} ({rule_id}): 'conditions' must be a mapping")

        _validate_conditions(conditions, context=f"{context} ({rule_id})")
        outcome = _parse_outcome(outcome_raw, context=f"{context} ({rule_id})")

        reason = rule_raw.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise RulesValidationError(f"{context} ({rule_id}): 'reason' must be a string")

        continue_after = bool(rule_raw.get("continue", False))
        rules.append(
            NormativeRule(
                id=rule_id,
                priority=priority,
                conditions=conditions,
                outcome=outcome,
                reason=reason,
                continue_after_match=continue_after,
            )
        )

    default_rules = [rule for rule in rules if rule.conditions.get("default") is True]
    if len(default_rules) != 1:
        raise RulesValidationError(
            "Exactly one rule must have 'conditions: { default: true }' as catch-all"
        )
    if default_rules[0].priority != max(rule.priority for rule in rules):
        raise RulesValidationError(
            "Default catch-all rule must have the highest (last) priority value"
        )

    rules.sort(key=lambda rule: rule.priority)
    return NormativeRulesDocument(
        version=version,
        description=description,
        assumptions=tuple(assumptions),
        rules=tuple(rules),
    )
