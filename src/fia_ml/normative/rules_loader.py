"""Load, validate, and version normative rule files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from fia_ml.normative.schema import NormativeRulesDocument, parse_rules_document
from fia_ml.utils import secure_file_io as sio


@dataclass(frozen=True)
class LoadedRules:
    path: Path
    content_hash: str
    document: NormativeRulesDocument

    @property
    def version(self) -> str:
        return self.document.version

    @property
    def rule_count(self) -> int:
        return len(self.document.rules)

    def to_version_metadata(self) -> dict[str, str | int]:
        return {
            "version": self.document.version,
            "description": self.document.description,
            "rules_path": str(self.path),
            "content_hash": self.content_hash,
            "rule_count": self.rule_count,
        }


def rules_file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of the raw rules file bytes."""
    data = sio.read_bytes(path)
    return hashlib.sha256(data).hexdigest()


def load_rules(path: Path) -> LoadedRules:
    """Load and validate a normative rules YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    raw = sio.read_yaml(path)
    document = parse_rules_document(raw)
    return LoadedRules(
        path=path,
        content_hash=rules_file_hash(path),
        document=document,
    )
