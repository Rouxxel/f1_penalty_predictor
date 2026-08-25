"""Priority-ordered normative rule matching. Implemented in Phase D."""

from __future__ import annotations

from typing import Any

from fia_ml.normative.schema import NormativeRule, RuleOutcome


def match_rule(rules: tuple[NormativeRule, ...], row: dict[str, Any]) -> tuple[NormativeRule, RuleOutcome]:
    """Return the first matching rule and its outcome for a row."""
    raise NotImplementedError(
        "Rule engine is not implemented yet (Normative Rules Phase D)."
    )
