"""Priority-ordered normative rule matching."""

from __future__ import annotations

from typing import Any

from fia_ml.normative.conditions import evaluate_conditions
from fia_ml.normative.schema import NormativeRule, RuleOutcome


def match_rule(
    rules: tuple[NormativeRule, ...],
    row: dict[str, Any],
) -> tuple[NormativeRule, RuleOutcome]:
    """Return the first matching rule and its outcome for a row."""
    matched_rule: NormativeRule | None = None
    matched_outcome: RuleOutcome | None = None

    for rule in rules:
        if not evaluate_conditions(rule.conditions, row):
            continue
        matched_rule = rule
        matched_outcome = rule.outcome
        if not rule.continue_after_match:
            break

    if matched_rule is None or matched_outcome is None:
        raise ValueError("No normative rule matched the incident row")
    return matched_rule, matched_outcome
