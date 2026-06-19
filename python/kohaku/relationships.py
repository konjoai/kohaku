"""Typed semantic relationships between memories — SQLite-backed.

Provenance (already shipped in Phase 12) tracks *lineage* — which episodes
were merged to form a prototype. This module is its sibling: it tracks
*semantic* relationships that a caller asserts after the fact.

A relationship is a (source_id → target_id, relation_type) triple plus
optional metadata. Two memories can have several relationships of
different types in either direction. The relation_type vocabulary is
extensible — :data:`KNOWN_RELATIONS` lists the documented set:

* ``supports``     — target reinforces source
* ``contradicts``  — target is inconsistent with source (a stronger
  signal than the soft contradiction-score from
  :mod:`kohaku.conflicts`; this is a user assertion).
* ``extends``      — target adds detail or scope to source
* ``derived_from`` — source was concluded from target (inference edge)
* ``references``   — target is cited by source

The schema is::

    relations(
        source_id   INTEGER NOT NULL,
        target_id   INTEGER NOT NULL,
        relation_type TEXT NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}',
        created_at  REAL NOT NULL,
        PRIMARY KEY (source_id, target_id, relation_type)
    )

The primary key prevents duplicates of the same (source, target, type)
triple. Calling :meth:`record` with an existing triple is an upsert —
``metadata`` and ``created_at`` get the new values.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


KNOWN_RELATIONS: frozenset[str] = frozenset(
    {
        "supports",
        "contradicts",
        "extends",
        "derived_from",
        "references",
    }
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS relations (
    source_id     INTEGER NOT NULL,
    target_id     INTEGER NOT NULL,
    relation_type TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, relation_type)
);
CREATE INDEX IF NOT EXISTS relations_target_idx ON relations(target_id);
CREATE INDEX IF NOT EXISTS relations_type_idx   ON relations(relation_type);
"""


@dataclass(frozen=True)
class Relationship:
    """One directed, typed edge between two memories."""

    source_id: int
    target_id: int
    relation_type: str
    metadata: dict[str, Any]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": int(self.source_id),
            "target_id": int(self.target_id),
            "relation_type": self.relation_type,
            "metadata": dict(self.metadata),
            "created_at": float(self.created_at),
        }


class RelationshipStore:
    """SQLite-backed registry of typed memory relationships.

    ``":memory:"`` is the default; pass a filesystem path for persistence.
    Thread-safe — every operation holds an internal lock.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.RLock()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "RelationshipStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()
            return int(row[0]) if row else 0

    # ── write ─────────────────────────────────────────────────────────────
    def record(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Relationship:
        """Upsert one relationship row."""
        if source_id < 0 or target_id < 0:
            raise ValueError("memory ids must be >= 0")
        if source_id == target_id:
            raise ValueError("source_id and target_id must differ")
        rel = (relation_type or "").strip()
        if not rel:
            raise ValueError("relation_type must be non-empty")
        meta = dict(metadata or {})
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO relations(source_id, target_id, relation_type,
                                          metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                       metadata = excluded.metadata,
                       created_at = excluded.created_at""",
                (int(source_id), int(target_id), rel, json.dumps(meta), now),
            )
            self._conn.commit()
        return Relationship(
            source_id=int(source_id),
            target_id=int(target_id),
            relation_type=rel,
            metadata=meta,
            created_at=now,
        )

    def delete(
        self,
        source_id: int,
        target_id: int,
        relation_type: Optional[str] = None,
    ) -> int:
        """Drop matching rows. Without ``relation_type``, drops every
        relation between the pair. Returns count deleted."""
        with self._lock:
            if relation_type is None:
                cur = self._conn.execute(
                    "DELETE FROM relations WHERE source_id = ? AND target_id = ?",
                    (int(source_id), int(target_id)),
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM relations WHERE source_id = ? AND target_id = ? "
                    "AND relation_type = ?",
                    (int(source_id), int(target_id), relation_type),
                )
            self._conn.commit()
            return cur.rowcount

    def delete_all_for(self, memory_id: int) -> int:
        """Drop every relation touching ``memory_id`` (either side).

        Called by :func:`kohaku.bulk_ops.batch_delete_by_ids` so the
        relationship table doesn't accumulate dangling edges to deleted
        memories.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                (int(memory_id), int(memory_id)),
            )
            self._conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM relations")
            self._conn.commit()

    # ── read ──────────────────────────────────────────────────────────────
    def list_outgoing(
        self,
        source_id: int,
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """All relationships originating at ``source_id``."""
        return self._list_directed(source_id, "source_id", relation_type)

    def list_incoming(
        self,
        target_id: int,
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """All relationships pointing at ``target_id``."""
        return self._list_directed(target_id, "target_id", relation_type)

    def list_related(
        self,
        memory_id: int,
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Union of outgoing + incoming edges for ``memory_id``."""
        outgoing = self.list_outgoing(memory_id, relation_type)
        incoming = self.list_incoming(memory_id, relation_type)
        seen: set[tuple[int, int, str]] = set()
        out: List[Relationship] = []
        for r in outgoing + incoming:
            key = (r.source_id, r.target_id, r.relation_type)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def list_by_type(self, relation_type: str) -> List[Relationship]:
        """Every relationship with the given type."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id, target_id, relation_type, metadata, created_at "
                "FROM relations WHERE relation_type = ? "
                "ORDER BY created_at DESC",
                (relation_type,),
            ).fetchall()
        return [_row_to_relationship(r) for r in rows]

    def counts_by_type(self) -> Dict[str, int]:
        """Histogram of relationship types."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type"
            ).fetchall()
        return {rt: int(cnt) for rt, cnt in rows}

    def _list_directed(
        self,
        memory_id: int,
        column: str,
        relation_type: Optional[str],
    ) -> List[Relationship]:
        if column not in ("source_id", "target_id"):
            raise ValueError("internal: column must be source_id or target_id")
        with self._lock:
            if relation_type is None:
                rows = self._conn.execute(
                    f"SELECT source_id, target_id, relation_type, metadata, created_at "
                    f"FROM relations WHERE {column} = ? "
                    f"ORDER BY created_at DESC",
                    (int(memory_id),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT source_id, target_id, relation_type, metadata, created_at "
                    f"FROM relations WHERE {column} = ? AND relation_type = ? "
                    f"ORDER BY created_at DESC",
                    (int(memory_id), relation_type),
                ).fetchall()
        return [_row_to_relationship(r) for r in rows]


def _row_to_relationship(row: tuple[Any, ...]) -> Relationship:
    source_id, target_id, relation_type, metadata_json, created_at = row
    try:
        meta = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        logger.warning(
            "dropping malformed relation metadata %s→%s/%s",
            source_id,
            target_id,
            relation_type,
        )
        meta = {}
    return Relationship(
        source_id=int(source_id),
        target_id=int(target_id),
        relation_type=str(relation_type),
        metadata=meta,
        created_at=float(created_at),
    )
