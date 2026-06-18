"""Memory versioning — SQLite-backed edit history per memory.

Every memory is born as version 1 at the moment it is stored. Each subsequent
edit (label, source, importance, tags, validity) appends a new version
snapshot; the *current* state of the live :class:`EnrichedMemoryStore` is
always the latest version, with older versions preserved verbatim.

Two paths in:

* **Auto-snapshot on store** — when an :class:`EnrichedMemoryStore` is
  constructed with ``versions=VersionStore(...)``, every ``store()`` writes
  a version-1 record automatically. The metadata, label, and tag set are
  captured at write time.

* **Explicit edit** — :func:`update_memory` mutates the live entry
  (re-encoding the label HV if the label changed) and appends the new
  version snapshot.

Schema (single table)::

    versions(
        memory_id   INTEGER NOT NULL,
        version     INTEGER NOT NULL,
        label       TEXT,
        source      TEXT,
        importance  REAL,
        tags        TEXT,          -- JSON array
        valid_from  TEXT,          -- ISO 8601, UTC
        valid_until TEXT,          -- ISO 8601 or NULL
        edited_at   REAL NOT NULL, -- epoch seconds
        editor      TEXT,          -- free-form actor tag
        PRIMARY KEY (memory_id, version)
    )

The table is small (one row per edit per memory) so an O(n²) "find latest"
scan would still be cheap, but we keep an index on ``memory_id`` to make
list lookups O(log n).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

from kohaku.attention import encode_text
from kohaku.enriched import EnrichedMemoryStore, _aware, _normalise_tag

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    memory_id   INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    label       TEXT,
    source      TEXT,
    importance  REAL,
    tags        TEXT NOT NULL DEFAULT '[]',
    valid_from  TEXT,
    valid_until TEXT,
    edited_at   REAL NOT NULL,
    editor      TEXT,
    PRIMARY KEY (memory_id, version)
);
CREATE INDEX IF NOT EXISTS versions_memory_idx ON versions(memory_id);
"""


@dataclass(frozen=True)
class MemoryVersion:
    """One immutable snapshot of a memory's state at edit time."""

    memory_id: int
    version: int
    label: str
    source: str
    importance: float
    tags: tuple = ()
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    edited_at: float = 0.0
    editor: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "memory_id": int(self.memory_id),
            "version": int(self.version),
            "label": self.label,
            "source": self.source,
            "importance": float(self.importance),
            "tags": list(self.tags),
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "edited_at": float(self.edited_at),
            "editor": self.editor,
        }


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one :func:`update_memory` call."""

    memory_id: int
    version: int
    changed_fields: tuple
    hv_re_encoded: bool

    def to_dict(self) -> dict:
        return {
            "memory_id": int(self.memory_id),
            "version": int(self.version),
            "changed_fields": list(self.changed_fields),
            "hv_re_encoded": bool(self.hv_re_encoded),
        }


class VersionStore:
    """SQLite-backed memory-version history.

    ``db_path=":memory:"`` (default) is ephemeral. Pass a filesystem path
    for cross-process persistence.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self._path = str(db_path)
        # check_same_thread=False is safe because every call holds the lock.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.RLock()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "VersionStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── core API ──────────────────────────────────────────────────────────
    def record(
        self,
        memory_id: int,
        *,
        label: str,
        source: str,
        importance: float,
        tags: Optional[List[str]] = None,
        valid_from: Optional[Any] = None,
        valid_until: Optional[Any] = None,
        editor: Optional[str] = None,
    ) -> MemoryVersion:
        """Append a new version snapshot. Returns the recorded :class:`MemoryVersion`."""
        if memory_id < 0:
            raise ValueError("memory_id must be >= 0")
        clean_tags = sorted({_normalise_tag(t) for t in (tags or []) if _normalise_tag(t)})
        edited_at = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM versions WHERE memory_id = ?",
                (int(memory_id),),
            ).fetchone()
            version = (int(row[0]) if row else 0) + 1
            self._conn.execute(
                """INSERT INTO versions(memory_id, version, label, source, importance,
                                         tags, valid_from, valid_until, edited_at, editor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(memory_id),
                    version,
                    label,
                    source,
                    float(importance),
                    json.dumps(clean_tags),
                    _iso_or_none(valid_from),
                    _iso_or_none(valid_until),
                    edited_at,
                    editor,
                ),
            )
            self._conn.commit()
        return MemoryVersion(
            memory_id=int(memory_id),
            version=version,
            label=label,
            source=source,
            importance=float(importance),
            tags=tuple(clean_tags),
            valid_from=_iso_or_none(valid_from),
            valid_until=_iso_or_none(valid_until),
            edited_at=edited_at,
            editor=editor,
        )

    def list_versions(self, memory_id: int) -> List[MemoryVersion]:
        """All versions for one memory, ascending. Empty if unknown."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, version, label, source, importance, tags, "
                "valid_from, valid_until, edited_at, editor "
                "FROM versions WHERE memory_id = ? ORDER BY version ASC",
                (int(memory_id),),
            ).fetchall()
        return [_row_to_version(r) for r in rows]

    def get_version(self, memory_id: int, version: int) -> Optional[MemoryVersion]:
        if version <= 0:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT memory_id, version, label, source, importance, tags, "
                "valid_from, valid_until, edited_at, editor "
                "FROM versions WHERE memory_id = ? AND version = ?",
                (int(memory_id), int(version)),
            ).fetchone()
        return _row_to_version(row) if row else None

    def latest_version(self, memory_id: int) -> Optional[MemoryVersion]:
        with self._lock:
            row = self._conn.execute(
                "SELECT memory_id, version, label, source, importance, tags, "
                "valid_from, valid_until, edited_at, editor "
                "FROM versions WHERE memory_id = ? ORDER BY version DESC LIMIT 1",
                (int(memory_id),),
            ).fetchone()
        return _row_to_version(row) if row else None

    def count(self, memory_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM versions WHERE memory_id = ?",
                (int(memory_id),),
            ).fetchone()
        return int(row[0]) if row else 0

    def delete(self, memory_id: int) -> int:
        """Drop every version for ``memory_id``. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM versions WHERE memory_id = ?",
                (int(memory_id),),
            )
            self._conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM versions")
            self._conn.commit()

    # ── snapshot helper ───────────────────────────────────────────────────
    def snapshot_metadata(
        self,
        memory_id: int,
        store: EnrichedMemoryStore,
        *,
        editor: Optional[str] = None,
    ) -> Optional[MemoryVersion]:
        """Record the *current* state of a live entry as a new version.

        Returns ``None`` if the entry no longer exists in the store.
        """
        meta = store.get_metadata(memory_id)
        if meta is None:
            return None
        live_label = _live_label(store, memory_id)
        return self.record(
            memory_id=memory_id,
            label=live_label,
            source=meta.source,
            importance=float(meta.importance),
            tags=list(meta.tags),
            valid_from=meta.valid_from,
            valid_until=meta.valid_until,
            editor=editor,
        )


# ──────────────────────────── update path ──────────────────────────────────

# Sentinel for the "field was not supplied" case. ``None`` can't be used
# because callers may legitimately want to clear ``valid_until``.
_UNSET = object()


def update_memory(
    store: EnrichedMemoryStore,
    memory_id: int,
    versions: VersionStore,
    *,
    label: Any = _UNSET,
    source: Any = _UNSET,
    importance: Any = _UNSET,
    tags: Any = _UNSET,
    valid_until: Any = _UNSET,
    editor: Optional[str] = None,
) -> UpdateResult:
    """Apply an edit to a live memory, persisting the new state as a version.

    Any field left as ``_UNSET`` is preserved untouched. ``label`` triggers
    re-encoding of the underlying bipolar hypervector (key + value).
    """
    if memory_id < 0:
        raise ValueError("memory_id must be >= 0")
    entry = next(
        (e for e in store.episodic._entries if e.id == memory_id), None,
    )
    if entry is None:
        raise KeyError(f"unknown memory_id {memory_id}")
    meta = store.get_metadata(memory_id)
    if meta is None:
        raise KeyError(f"missing metadata for memory_id {memory_id}")

    changed: List[str] = []
    hv_re_encoded = False

    if label is not _UNSET and label != entry.label:
        new_label = str(label)
        new_hv = encode_text(new_label)
        entry.key = new_hv
        entry.value = new_hv
        entry.label = new_label
        changed.append("label")
        hv_re_encoded = True

    if source is not _UNSET and source != meta.source:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        meta.source = source
        changed.append("source")

    if importance is not _UNSET:
        try:
            imp = float(importance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"importance must be a number: {exc}") from exc
        if not 0.0 <= imp <= 1.0:
            raise ValueError("importance must be in [0, 1]")
        if imp != meta.importance:
            meta.importance = imp
            changed.append("importance")

    if tags is not _UNSET:
        if not isinstance(tags, (list, tuple, set)):
            raise ValueError("tags must be a list/tuple/set of strings")
        new_tags = {_normalise_tag(t) for t in tags if _normalise_tag(t)}
        if new_tags != set(meta.tags):
            meta.tags = new_tags
            changed.append("tags")

    if valid_until is not _UNSET:
        if valid_until is None:
            if meta.valid_until is not None:
                meta.valid_until = None
                changed.append("valid_until")
        else:
            dt = _coerce_datetime(valid_until)
            if dt is None:
                raise ValueError("valid_until must be a datetime, ISO string, or None")
            dt = _aware(dt)
            if dt < meta.valid_from:
                raise ValueError("valid_until must be >= valid_from")
            if meta.valid_until != dt:
                meta.valid_until = dt
                changed.append("valid_until")

    snapshot = versions.snapshot_metadata(memory_id, store, editor=editor)
    if snapshot is None:
        raise RuntimeError(
            "version snapshot returned None for live memory — store inconsistency"
        )
    return UpdateResult(
        memory_id=memory_id,
        version=snapshot.version,
        changed_fields=tuple(changed),
        hv_re_encoded=hv_re_encoded,
    )


# ──────────────────────────── helpers ──────────────────────────────────────

def _iso_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(value)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _row_to_version(row: Any) -> MemoryVersion:
    (mid, ver, label, source, importance, tags_json,
     vf, vu, edited_at, editor) = row
    try:
        tags = tuple(json.loads(tags_json or "[]"))
    except json.JSONDecodeError:
        logger.warning("dropping malformed tags JSON for memory %s v%s", mid, ver)
        tags = ()
    return MemoryVersion(
        memory_id=int(mid),
        version=int(ver),
        label=label or "",
        source=source or "",
        importance=float(importance or 0.0),
        tags=tags,
        valid_from=vf,
        valid_until=vu,
        edited_at=float(edited_at or 0.0),
        editor=editor,
    )


def _live_label(store: EnrichedMemoryStore, memory_id: int) -> str:
    for e in store.episodic.entries():
        if e.id == memory_id:
            return e.label
    return ""
