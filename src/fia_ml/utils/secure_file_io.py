"""
#############################################################################
### Secure file I/O utility
###
### @file secure_file_io.py
### @author Sebastian Russo
### @date 2026
#############################################################################

General-purpose, hardened helpers for reading and writing files by path.

Why not just use ``with open(...)`` directly everywhere?
    * No path safety  -> a path built from user input (``../../etc/passwd``)
        or a malicious symlink can read/clobber files outside your project.
    * No crash safety  -> ``open(path, "w")`` truncates the file immediately,
        so a crash (or a JSON serialization error) mid-write leaves the file
        half-written / corrupt.
    * No size guard    -> ``json.load`` / ``read()`` on a huge or malicious
        file can exhaust memory.
    * No concurrency    -> two threads writing the same file race.

This module addresses all four:
    * Optional allowed-root confinement (see ``set_allowed_root``). When set,
        every resolved path must live inside that root; ``..`` traversal and
        symlinks escaping the root are rejected.
    * Atomic writes: data is written to a temp file in the same directory,
        flushed + fsynced, then ``os.replace``-d over the target (atomic on
        POSIX and Windows). Readers never see a partial file.
    * A configurable max-read-size guard (``MAX_READ_BYTES``).
    * A per-path ``threading.Lock`` so reads/writes of the same file are
        serialized. Read-modify-write helpers hold the lock for the whole
        transaction.

Layers
    Primitives : read_bytes / write_bytes / read_text / write_text
    Formats    : read_json / write_json, read_csv / write_csv,
                    read_yaml / write_yaml   (YAML needs PyYAML installed)
    Records    : read_all / save_all / find_by_id / find_by_field /
                    create_record / update_record / delete_record
                    (treat a JSON file as a list[dict] "collection")
"""

#Native imports
import csv
import io
import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional, Union

#Optional dependency - only required by the read_yaml / write_yaml helpers
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

PathLike = Union[str, os.PathLike]


# --- CONFIGURATION AREA ---
#Maximum number of bytes any read_* helper will pull into memory. Guards
#against accidentally (or maliciously) loading a multi-GB file. Override per
#call with the ``max_bytes`` argument, or change this module-level default.
MAX_READ_BYTES: int = 50 * 1024 * 1024  # 50 MiB

#When not None, every path is resolved and required to live inside this root.
#Set it once at startup (e.g. to your project's data dir) via set_allowed_root.
_allowed_root: Optional[Path] = None

#Sentinel so callers can pass default=None to read_json and still distinguish
#"file missing" from "file contained null".
_RAISE = object()

#Per-path locks - created on first access. Keyed by the resolved path string.
_locks: dict[str, threading.Lock] = {}
_locks_meta_lock = threading.Lock()  # guards the _locks dict itself


# --- EXCEPTIONS ---
class FileStoreError(Exception):
    """Base class for all errors raised by this module."""


class PathSecurityError(FileStoreError):
    """Raised when a path escapes the configured allowed root."""


class FileTooLargeError(FileStoreError):
    """Raised when a file exceeds the allowed read size."""


# --- CONFIG HELPERS ---
def set_allowed_root(root: Optional[PathLike]) -> None:
    """Confine all subsequent file access to *root* (or disable with ``None``).

    Parameters:
        root: Directory that every accessed path must live inside. Pass
            ``None`` to disable confinement (paths are then trusted).
    """
    global _allowed_root
    _allowed_root = Path(root).resolve() if root is not None else None


def get_allowed_root() -> Optional[Path]:
    """Return the currently configured allowed root, or ``None`` if disabled."""
    return _allowed_root


# --- INTERNAL HELPERS ---
def _resolve_path(path: PathLike) -> Path:
    """Resolve *path* to an absolute path and enforce root confinement.

    Symlinks are resolved, so a link pointing outside the allowed root is
    rejected just like a literal ``..`` traversal.
    """
    resolved = Path(path).resolve()
    if _allowed_root is not None and not resolved.is_relative_to(_allowed_root):
        raise PathSecurityError(
            f"Path '{resolved}' is outside the allowed root '{_allowed_root}'."
        )
    return resolved


def _get_lock(path: Path) -> threading.Lock:
    """Return (and lazily create) the Lock guarding *path*."""
    key = str(path)
    with _locks_meta_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _read_bytes_unlocked(path: Path, max_bytes: Optional[int]) -> bytes:
    """Read *path* enforcing *max_bytes*. Assumes the lock is already held."""
    if not path.is_file():
        raise FileNotFoundError(f"No such file: '{path}'")

    if max_bytes is not None:
        size = path.stat().st_size
        if size > max_bytes:
            raise FileTooLargeError(
                f"File '{path}' is {size} bytes, exceeds limit of {max_bytes}."
            )

    with path.open("rb") as fh:
        #Read one extra byte to detect files that grew between stat and read
        #(TOCTOU), e.g. another writer appending concurrently.
        limit = -1 if max_bytes is None else max_bytes + 1
        data = fh.read(limit)

    if max_bytes is not None and len(data) > max_bytes:
        raise FileTooLargeError(
            f"File '{path}' exceeds the read limit of {max_bytes} bytes."
        )
    return data


def _atomic_write_unlocked(path: Path, data: bytes) -> None:
    """Atomically write *data* to *path*. Assumes the lock is already held.

    Strategy: write to a temp file in the same directory, flush + fsync, then
    ``os.replace`` over the destination. On any failure the temp file is
    removed and the original (if any) is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    #Capture the existing file mode so we can preserve it across the replace
    #(mkstemp creates 0o600 which is often too strict for shared configs).
    existing_mode: Optional[int] = None
    if path.exists():
        existing_mode = stat.S_IMODE(path.stat().st_mode)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, path)  # atomic
    finally:
        #If os.replace succeeded the temp file is already gone; this only
        #cleans up when something failed before/at the replace.
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Primitives: bytes & text
# ---------------------------------------------------------------------------
def read_bytes(path: PathLike, *, max_bytes: Optional[int] = MAX_READ_BYTES) -> bytes:
    """Read and return the raw bytes of *path*.

    Parameters:
        path:      File to read.
        max_bytes: Reject files larger than this. ``None`` disables the guard.

    Raises:
        FileNotFoundError, PathSecurityError, FileTooLargeError.
    """
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        return _read_bytes_unlocked(resolved, max_bytes)


def write_bytes(path: PathLike, data: bytes) -> None:
    """Atomically write raw *data* (bytes) to *path*, creating parents."""
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        _atomic_write_unlocked(resolved, data)


def read_text(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    max_bytes: Optional[int] = MAX_READ_BYTES,
) -> str:
    """Read and return the text content of *path* (default UTF-8)."""
    return read_bytes(path, max_bytes=max_bytes).decode(encoding)


def write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically write *text* to *path* (default UTF-8)."""
    write_bytes(path, text.encode(encoding))


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def read_json(
    path: PathLike,
    *,
    default: Any = _RAISE,
    encoding: str = "utf-8",
    max_bytes: Optional[int] = MAX_READ_BYTES,
) -> Any:
    """Read and parse a JSON file.

    Parameters:
        path:      JSON file to read.
        default:   Value to return if the file does not exist. If omitted, a
                missing file raises ``FileNotFoundError`` (use e.g.
                ``default=[]`` or ``default={}`` to get an empty container).
        encoding:  Text encoding (default UTF-8).
        max_bytes: Read-size guard.

    Raises:
        json.JSONDecodeError on malformed content; FileNotFoundError if the
        file is missing and no *default* was supplied.
    """
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        if not resolved.is_file():
            if default is _RAISE:
                raise FileNotFoundError(f"No such file: '{resolved}'")
            return default
        raw = _read_bytes_unlocked(resolved, max_bytes)
    return json.loads(raw.decode(encoding))


def write_json(
    path: PathLike,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Serialize *data* to JSON and atomically write it to *path*."""
    text = json.dumps(
        data, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys
    )
    write_text(path, text, encoding=encoding)


# ---------------------------------------------------------------------------
# CSV  (returns / accepts list[dict])
# ---------------------------------------------------------------------------
def read_csv(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    max_bytes: Optional[int] = MAX_READ_BYTES,
    **reader_kwargs: Any,
) -> list[dict]:
    """Read a CSV file into a list of row dicts (via ``csv.DictReader``).

    Extra keyword args are forwarded to ``csv.DictReader`` (e.g.
    ``delimiter=";"``).
    """
    text = read_text(path, encoding=encoding, max_bytes=max_bytes)
    reader = csv.DictReader(io.StringIO(text), **reader_kwargs)
    return [dict(row) for row in reader]


def write_csv(
    path: PathLike,
    rows: list[dict],
    *,
    fieldnames: Optional[list[str]] = None,
    encoding: str = "utf-8",
    **writer_kwargs: Any,
) -> None:
    """Atomically write *rows* (list of dicts) to a CSV file.

    Parameters:
        fieldnames: Column order. If omitted, it is derived from the union of
                    keys across all rows, preserving first-seen order.
    """
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, **writer_kwargs)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue(), encoding=encoding)


# ---------------------------------------------------------------------------
# YAML  (requires PyYAML; uses safe_load / safe_dump, never full_load)
# ---------------------------------------------------------------------------
def _require_yaml() -> None:
    if not _HAS_YAML:
        raise FileStoreError(
            "PyYAML is not installed. Add 'pyyaml' to requirements.txt "
            "to use read_yaml / write_yaml."
        )


def read_yaml(
    path: PathLike,
    *,
    default: Any = _RAISE,
    encoding: str = "utf-8",
    max_bytes: Optional[int] = MAX_READ_BYTES,
) -> Any:
    """Read and parse a YAML file using ``yaml.safe_load`` (no code execution).

    See ``read_json`` for the *default* semantics.
    """
    _require_yaml()
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        if not resolved.is_file():
            if default is _RAISE:
                raise FileNotFoundError(f"No such file: '{resolved}'")
            return default
        raw = _read_bytes_unlocked(resolved, max_bytes)
    return yaml.safe_load(raw.decode(encoding))


def write_yaml(
    path: PathLike,
    data: Any,
    *,
    encoding: str = "utf-8",
    **dump_kwargs: Any,
) -> None:
    """Serialize *data* to YAML (``yaml.safe_dump``) and atomically write it."""
    _require_yaml()
    dump_kwargs.setdefault("default_flow_style", False)
    dump_kwargs.setdefault("sort_keys", False)
    dump_kwargs.setdefault("allow_unicode", True)
    text = yaml.safe_dump(data, **dump_kwargs)
    write_text(path, text, encoding=encoding)


# ---------------------------------------------------------------------------
# Record helpers - treat a JSON file as a list[dict] "collection"
# ---------------------------------------------------------------------------
# These hold the per-file lock for the whole read-modify-write transaction, so
# concurrent create/update/delete calls on the same file cannot interleave.
def _load_records(path: Path, max_bytes: Optional[int]) -> list:
    """Load a JSON list from *path* (empty list if missing). Lock must be held."""
    if not path.is_file():
        return []
    raw = _read_bytes_unlocked(path, max_bytes)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise FileStoreError(
            f"Record helpers expect a JSON array at '{path}', got {type(data).__name__}."
        )
    return data


def read_all(path: PathLike, *, max_bytes: Optional[int] = MAX_READ_BYTES) -> list:
    """Return every record in the JSON file at *path* (``[]`` if it doesn't exist)."""
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        return _load_records(resolved, max_bytes)


def save_all(path: PathLike, data: list) -> None:
    """Overwrite the JSON file at *path* with *data* (full replacement)."""
    write_json(path, data)


def find_by_id(
    path: PathLike, record_id: Any, *, id_field: str = "id"
) -> Optional[dict]:
    """Return the first record whose *id_field* equals *record_id*, or ``None``."""
    for record in read_all(path):
        if record.get(id_field) == record_id:
            return record
    return None


def find_by_field(path: PathLike, field: str, value: Any) -> list:
    """Return all records where ``record[field] == value``."""
    return [r for r in read_all(path) if r.get(field) == value]


def create_record(path: PathLike, record: dict) -> dict:
    """Append *record* to the JSON collection at *path* and persist it."""
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        data = _load_records(resolved, MAX_READ_BYTES)
        data.append(record)
        _atomic_write_unlocked(
            resolved, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        )
    return record


def update_record(
    path: PathLike, record_id: Any, updates: dict, *, id_field: str = "id"
) -> dict:
    """Merge *updates* into the record matching *record_id* and persist.

    Raises:
        ValueError: if no record with *record_id* exists.
    """
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        data = _load_records(resolved, MAX_READ_BYTES)

        updated_record = None
        for record in data:
            if record.get(id_field) == record_id:
                record.update(updates)
                updated_record = record
                break

        if updated_record is None:
            raise ValueError(
                f"Record with {id_field}='{record_id}' not found in '{resolved}'."
            )

        _atomic_write_unlocked(
            resolved, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        )
    return updated_record


def delete_record(
    path: PathLike, record_id: Any, *, id_field: str = "id"
) -> dict:
    """Remove the record matching *record_id* from *path* and persist.

    Raises:
        ValueError: if no record with *record_id* exists.
    """
    resolved = _resolve_path(path)
    with _get_lock(resolved):
        data = _load_records(resolved, MAX_READ_BYTES)

        deleted_record = None
        remaining = []
        for record in data:
            if deleted_record is None and record.get(id_field) == record_id:
                deleted_record = record
            else:
                remaining.append(record)

        if deleted_record is None:
            raise ValueError(
                f"Record with {id_field}='{record_id}' not found in '{resolved}'."
            )

        _atomic_write_unlocked(
            resolved, json.dumps(remaining, indent=2, ensure_ascii=False).encode("utf-8")
        )
    return deleted_record


#Example usage
"""
from src.utils import secure_file_io as fio

# (Optional) confine ALL access to a trusted directory - blocks ../ escapes:
fio.set_allowed_root("src/resources")

# Generic JSON / text / csv / yaml, by path:
cfg   = fio.read_json("src/resources/config.json", default={})
fio.write_json("src/resources/config.json", {"debug": True})

text  = fio.read_text("src/resources/notes.md")
rows  = fio.read_csv("src/resources/users.csv")
data  = fio.read_yaml("src/resources/settings.yaml", default={})

# JSON-file-as-collection CRUD (replaces the old json_store API):
users = "src/resources/mock_db_jsons/users.json"
fio.create_record(users, {"id": "u1", "name": "Ada"})
fio.update_record(users, "u1", {"name": "Ada Lovelace"})
user  = fio.find_by_id(users, "u1")
fio.delete_record(users, "u1")
"""
