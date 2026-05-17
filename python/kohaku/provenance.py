"""Provenance graph — DAG of memory lineage backed by SQLite.

Every memory write records a :class:`ProvenanceRecord` capturing where the
memory came from: its parent memories (if any), the source class
(``user_input``, ``inference``, ``consolidation``, ``import``, plus the
existing kohaku source labels like ``web_search`` / ``tool_result``), the
creation time, and optional free-form metadata.

The graph is a DAG: parents → children, where consolidation produces a
node whose parents are the memories that were merged.

API surface
-----------
* :func:`ProvenanceGraph.record` — upsert a node + its incoming edges.
* :func:`ProvenanceGraph.get_ancestors` — BFS up to ``max_depth`` levels.
* :func:`ProvenanceGraph.get_descendants` — BFS down to ``max_depth``.
* :func:`ProvenanceGraph.get_full_graph` — both directions plus the union
  edge list, suitable for client-side rendering.

Storage: a single SQLite table::

    provenance(
        memory_id   TEXT PRIMARY KEY,
        parent_ids  TEXT,                 -- JSON list of strings
        source_type TEXT NOT NULL,
        created_at  REAL NOT NULL,        -- epoch seconds
        metadata    TEXT                  -- JSON object
    )

A secondary index on ``parent_ids`` is not used — descendant lookups scan
the table; for the kohaku scale (<10⁵ rows) this is fine. If profiling
ever shows it as a hot path, split the edges into their own ``edges``
table with ``(parent_id, child_id)`` rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Recognised provenance source classes. The store accepts any string so the
# existing `kohaku.enriched` source labels (`user_input`, `tool_result`,
# `web_search`, `agent_inference`) flow through unchanged; this set is just
# the documented vocabulary for callers that want a hint.
KNOWN_SOURCE_TYPES: frozenset[str] = frozenset({
    "user_input",
    "inference",
    "agent_inference",
    "consolidation",
    "import",
    "tool_result",
    "web_search",
})


@dataclass(frozen=True)
class ProvenanceNode:
    """One node in the provenance DAG.

    ``children_count`` and ``parent_count`` are counted at the time of
    the lookup — they reflect the live graph, not the row that was stored.
    """

    memory_id: str
    source_type: str
    depth: int
    children_count: int
    parent_count: int
    created_at: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "source_type": self.source_type,
            "depth": self.depth,
            "children_count": self.children_count,
            "parent_count": self.parent_count,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProvenanceGraphResult:
    """Bidirectional traversal result for one root memory.

    ``edges`` is the de-duplicated edge list across both ancestor and
    descendant sub-graphs, suitable for rendering as a DAG.
    """

    root_id: str
    ancestors: List[ProvenanceNode]
    descendants: List[ProvenanceNode]
    edges: List[Tuple[str, str]]   # (parent_id, child_id)
    nodes: List[ProvenanceNode]

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "ancestors": [n.to_dict() for n in self.ancestors],
            "descendants": [n.to_dict() for n in self.descendants],
            "edges": [list(e) for e in self.edges],
            "nodes": [n.to_dict() for n in self.nodes],
        }


# ──────────────────────────── Schema ────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provenance (
    memory_id   TEXT PRIMARY KEY,
    parent_ids  TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL,
    created_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS provenance_source_idx ON provenance(source_type);
"""


class ProvenanceGraph:
    """SQLite-backed DAG of memory lineage.

    Pass ``":memory:"`` (default) for an ephemeral in-process store, or a
    filesystem path for persistence. Thread-safe — all writes and reads
    hold an internal lock so the underlying ``sqlite3.Connection`` stays
    single-threaded.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self._path = str(db_path)
        # check_same_thread=False is safe here because every call goes
        # through `self._lock`. Without that flag, SQLite would refuse
        # cross-thread use even though we synchronise access ourselves.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.RLock()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ProvenanceGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── write ─────────────────────────────────────────────────────────────
    def record(
        self,
        memory_id: Union[str, int],
        parent_ids: Optional[Iterable[Union[str, int]]] = None,
        source_type: str = "user_input",
        metadata: Optional[dict] = None,
    ) -> ProvenanceNode:
        """Upsert one provenance row. Returns the node as it exists after the write.

        ``parent_ids`` are stored as strings; integer ids are coerced. An
        empty parent list marks the memory as a *root* (no ancestors).
        """
        mid = str(memory_id)
        if not mid:
            raise ValueError("memory_id must be non-empty")
        if not source_type:
            raise ValueError("source_type must be non-empty")
        parents = [str(p) for p in (parent_ids or [])]
        if any(p == mid for p in parents):
            raise ValueError("memory_id cannot be its own parent")
        meta = dict(metadata or {})
        now = time.time()

        with self._lock:
            # Preserve created_at across upserts so the node's age stays stable.
            row = self._conn.execute(
                "SELECT created_at FROM provenance WHERE memory_id = ?",
                (mid,),
            ).fetchone()
            created_at = row[0] if row is not None else now
            self._conn.execute(
                """INSERT INTO provenance(memory_id, parent_ids, source_type,
                                          created_at, metadata)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                       parent_ids = excluded.parent_ids,
                       source_type = excluded.source_type,
                       metadata = excluded.metadata""",
                (mid, json.dumps(parents), source_type,
                 created_at, json.dumps(meta)),
            )
            self._conn.commit()
            return self._node_for(mid, depth=0)

    def record_consolidation(
        self,
        merged_id: Union[str, int],
        source_ids: Iterable[Union[str, int]],
        metadata: Optional[dict] = None,
    ) -> ProvenanceNode:
        """Convenience wrapper for sleep-phase consolidation: emit a node whose
        parents are the merged source memories with ``source_type='consolidation'``."""
        meta = dict(metadata or {})
        meta.setdefault("event", "sleep_consolidation")
        return self.record(
            memory_id=merged_id,
            parent_ids=source_ids,
            source_type="consolidation",
            metadata=meta,
        )

    # ── read ──────────────────────────────────────────────────────────────
    def has(self, memory_id: Union[str, int]) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM provenance WHERE memory_id = ?",
                (str(memory_id),),
            ).fetchone()
            return row is not None

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM provenance").fetchone()
            return int(row[0]) if row else 0

    def _node_for(self, memory_id: str, *, depth: int) -> ProvenanceNode:
        """Read one node from disk and count its live in/out degree."""
        row = self._conn.execute(
            "SELECT memory_id, parent_ids, source_type, created_at, metadata "
            "FROM provenance WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            # Phantom node — appears in someone's parent_ids but was never recorded.
            return ProvenanceNode(
                memory_id=memory_id,
                source_type="unknown",
                depth=depth,
                children_count=self._count_children(memory_id),
                parent_count=0,
                created_at=0.0,
                metadata={},
            )
        mid, parents_json, source_type, created_at, meta_json = row
        parents = json.loads(parents_json or "[]")
        meta = json.loads(meta_json or "{}")
        return ProvenanceNode(
            memory_id=mid,
            source_type=source_type,
            depth=depth,
            parent_count=len(parents),
            children_count=self._count_children(mid),
            created_at=float(created_at),
            metadata=meta,
        )

    def _count_children(self, memory_id: str) -> int:
        # Scan parent_ids JSON. SQLite has json_each but this is just as fast
        # for our scale and works against the stdlib sqlite3 without modules.
        cur = self._conn.execute(
            "SELECT parent_ids FROM provenance WHERE parent_ids LIKE ?",
            (f"%{json.dumps(memory_id)}%",),
        )
        count = 0
        for (raw,) in cur:
            if memory_id in json.loads(raw or "[]"):
                count += 1
        return count

    def _direct_parents(self, memory_id: str) -> List[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT parent_ids FROM provenance WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return []
        return [str(p) for p in json.loads(row[0] or "[]")]

    def _direct_children(self, memory_id: str) -> List[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT memory_id, parent_ids FROM provenance "
                "WHERE parent_ids LIKE ?",
                (f"%{json.dumps(memory_id)}%",),
            )
            out: List[str] = []
            for child_id, raw in cur:
                if memory_id in json.loads(raw or "[]"):
                    out.append(str(child_id))
        return out

    # ── traversal ─────────────────────────────────────────────────────────
    def get_ancestors(
        self,
        memory_id: Union[str, int],
        max_depth: int = 10,
    ) -> List[ProvenanceNode]:
        """BFS up the graph. Depth 0 is *not* included; the root is the query."""
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        return self._bfs(memory_id, max_depth=max_depth, direction="up")

    def get_descendants(
        self,
        memory_id: Union[str, int],
        max_depth: int = 10,
    ) -> List[ProvenanceNode]:
        """BFS down the graph. Depth 0 is *not* included; the root is the query."""
        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        return self._bfs(memory_id, max_depth=max_depth, direction="down")

    def _bfs(
        self,
        memory_id: Union[str, int],
        *,
        max_depth: int,
        direction: str,
    ) -> List[ProvenanceNode]:
        start = str(memory_id)
        visited: Set[str] = {start}
        out: List[ProvenanceNode] = []
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        with self._lock:
            while queue:
                cur, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                neighbours = (
                    self._direct_parents(cur)
                    if direction == "up" else self._direct_children(cur)
                )
                for nb in neighbours:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    out.append(self._node_for(nb, depth=depth + 1))
                    queue.append((nb, depth + 1))
        return out

    def get_full_graph(
        self,
        memory_id: Union[str, int],
        max_depth: int = 5,
    ) -> ProvenanceGraphResult:
        """Both-direction traversal. Edges are de-duplicated; nodes are union."""
        root_id = str(memory_id)
        ancestors = self.get_ancestors(root_id, max_depth=max_depth)
        descendants = self.get_descendants(root_id, max_depth=max_depth)
        with self._lock:
            root_node = self._node_for(root_id, depth=0)

        node_index: dict[str, ProvenanceNode] = {root_node.memory_id: root_node}
        for n in ancestors + descendants:
            node_index.setdefault(n.memory_id, n)

        edges: Set[Tuple[str, str]] = set()
        with self._lock:
            for nid in node_index:
                for parent in self._direct_parents(nid):
                    if parent in node_index:
                        edges.add((parent, nid))
        return ProvenanceGraphResult(
            root_id=root_id,
            ancestors=ancestors,
            descendants=descendants,
            edges=sorted(edges),
            nodes=list(node_index.values()),
        )

    # ── maintenance ───────────────────────────────────────────────────────
    def delete(self, memory_id: Union[str, int]) -> bool:
        """Delete one row. Returns True if a row was removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM provenance WHERE memory_id = ?",
                (str(memory_id),),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM provenance")
            self._conn.commit()
