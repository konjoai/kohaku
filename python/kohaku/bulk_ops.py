"""Bulk memory operations — update / delete / export many memories in one call.

Each operation aggregates per-row outcomes into a structured report so the
API layer can return a single, well-typed summary. Individual failures
don't abort the batch — they accumulate into ``errors``.

Operations
----------
* :func:`batch_update` — apply :func:`kohaku.versions.update_memory` to a
  list of `{memory_id, field…}` dicts.
* :func:`batch_delete_by_ids` — drop the listed memories from the store
  (and from the version + provenance + relationship side-tables when
  attached).
* :func:`batch_delete_by_filter` — drop everything matching a filter
  spec (``stale_days``, ``source``, ``tags_any``, ``older_than_days``).
* :func:`batch_export` — extract a specific subset of memories via the
  existing :mod:`kohaku.portability` exporters.

All four are read-and-write-safe under a single :class:`threading.Lock`
held by the caller (e.g. ``RestState.lock`` in the FastAPI app).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kohaku.enriched import EnrichedMemoryStore
from kohaku.portability import ExportBundle, export_csv, export_markdown
from kohaku.portability import export_json as _portability_export_json
from kohaku.versions import VersionStore, update_memory

logger = logging.getLogger(__name__)


# ──────────────────────────── DTOs ─────────────────────────────────────────

@dataclass(frozen=True)
class BatchUpdateReport:
    processed: int
    updated: int
    failed: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "processed": int(self.processed),
            "updated": int(self.updated),
            "failed": int(self.failed),
            "errors": list(self.errors),
            "results": list(self.results),
        }


@dataclass(frozen=True)
class BatchDeleteReport:
    processed: int
    deleted: int
    failed: int
    deleted_ids: Tuple[int, ...]
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "processed": int(self.processed),
            "deleted": int(self.deleted),
            "failed": int(self.failed),
            "deleted_ids": [int(i) for i in self.deleted_ids],
            "errors": list(self.errors),
        }


# ──────────────────────────── batch update ─────────────────────────────────

_UPDATE_FIELDS = ("label", "source", "importance", "tags", "valid_until")


def batch_update(
    store: EnrichedMemoryStore,
    versions: VersionStore,
    updates: List[Dict[str, Any]],
    *,
    editor: Optional[str] = None,
) -> BatchUpdateReport:
    """Apply an edit to many memories. Each update must include ``memory_id``;
    every other key is forwarded to :func:`update_memory` as a keyword arg."""
    if not isinstance(updates, list):
        raise TypeError("updates must be a list of dicts")

    processed = len(updates)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    updated_count = 0

    for idx, row in enumerate(updates):
        if not isinstance(row, dict):
            errors.append({"index": idx, "error": "row must be a dict"})
            continue
        memory_id = row.get("memory_id") or row.get("id")
        if memory_id is None:
            errors.append({"index": idx, "error": "missing memory_id"})
            continue
        try:
            memory_id = int(memory_id)
        except (TypeError, ValueError):
            errors.append({"index": idx, "error": f"invalid memory_id {memory_id!r}"})
            continue
        kwargs = {k: row[k] for k in _UPDATE_FIELDS if k in row}
        row_editor = row.get("editor", editor)
        try:
            result = update_memory(
                store, memory_id, versions,
                editor=row_editor, **kwargs,
            )
        except KeyError as exc:
            errors.append({"index": idx, "memory_id": memory_id,
                            "error": f"not found: {exc}"})
            continue
        except (ValueError, TypeError) as exc:
            errors.append({"index": idx, "memory_id": memory_id,
                            "error": str(exc)})
            continue
        results.append(result.to_dict())
        updated_count += 1

    return BatchUpdateReport(
        processed=processed,
        updated=updated_count,
        failed=len(errors),
        errors=errors,
        results=results,
    )


# ──────────────────────────── batch delete ─────────────────────────────────

def _remove_entries(store: EnrichedMemoryStore, ids: Iterable[int]) -> int:
    """Drop entries from the underlying :class:`EpisodicMemory`."""
    target = {int(i) for i in ids}
    if not target:
        return 0
    kept = [e for e in store.episodic._entries if e.id not in target]
    removed = len(store.episodic._entries) - len(kept)
    store.episodic._entries = kept
    if removed:
        store.episodic._mark_mutated()  # invalidate the retrieval-index cache
    for eid in target:
        store._meta.pop(eid, None)
        if store.provenance is not None:
            try:
                store.provenance.delete(eid)
            except (AttributeError, OSError) as exc:
                logger.warning(
                    "provenance delete failed for %s (%s)",
                    eid, exc.__class__.__name__,
                )
        if store.versions is not None:
            try:
                store.versions.delete(eid)
            except (AttributeError, OSError) as exc:
                logger.warning(
                    "version delete failed for %s (%s)",
                    eid, exc.__class__.__name__,
                )
    return removed


def batch_delete_by_ids(
    store: EnrichedMemoryStore,
    ids: List[int],
    *,
    relationships: "Optional[object]" = None,
) -> BatchDeleteReport:
    """Delete the listed memory ids. Missing ids are reported in ``errors``."""
    if not isinstance(ids, list):
        raise TypeError("ids must be a list of integers")

    live_ids = {e.id for e in store.episodic.entries()}
    target_ids: List[int] = []
    errors: List[Dict[str, Any]] = []
    for idx, raw in enumerate(ids):
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            errors.append({"index": idx, "error": f"invalid id {raw!r}"})
            continue
        if mid not in live_ids:
            errors.append({"index": idx, "memory_id": mid, "error": "not found"})
            continue
        target_ids.append(mid)

    removed = _remove_entries(store, target_ids)
    if relationships is not None:
        for mid in target_ids:
            try:
                relationships.delete_all_for(mid)
            except (AttributeError, OSError) as exc:
                logger.warning(
                    "relationship cleanup failed for %s (%s)",
                    mid, exc.__class__.__name__,
                )

    return BatchDeleteReport(
        processed=len(ids),
        deleted=removed,
        failed=len(errors),
        deleted_ids=tuple(target_ids),
        errors=errors,
    )


def batch_delete_by_filter(
    store: EnrichedMemoryStore,
    *,
    stale_days: Optional[int] = None,
    older_than_days: Optional[float] = None,
    source: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    max_importance: Optional[float] = None,
    relationships: "Optional[object]" = None,
    now: Optional[datetime] = None,
) -> BatchDeleteReport:
    """Drop every live memory that matches all of the supplied filters.

    At least one filter must be set — calling this with everything ``None``
    raises (we don't accept an unconditional wipe by accident).
    """
    filters_set = [
        stale_days, older_than_days, source, tags_any, max_importance,
    ]
    if all(f is None for f in filters_set):
        raise ValueError(
            "at least one filter must be set "
            "(stale_days / older_than_days / source / tags_any / max_importance)"
        )
    if stale_days is not None and stale_days <= 0:
        raise ValueError("stale_days must be > 0")
    if older_than_days is not None and older_than_days <= 0:
        raise ValueError("older_than_days must be > 0")
    if max_importance is not None and not 0.0 <= max_importance <= 1.0:
        raise ValueError("max_importance must be in [0, 1]")

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    any_set = set()
    if tags_any:
        for t in tags_any:
            if isinstance(t, str) and t.strip():
                any_set.add(t.strip().lower())

    matched: List[int] = []
    for e in store.episodic.entries():
        meta = store.get_metadata(e.id)
        if meta is None:
            continue
        if source is not None and meta.source != source:
            continue
        if any_set and not (meta.tags & any_set):
            continue
        if max_importance is not None and meta.importance > max_importance:
            continue
        if stale_days is not None:
            age = (now - meta.valid_from).total_seconds() / 86_400.0
            if not (age >= stale_days and meta.reinforcement_count == 0):
                continue
        if older_than_days is not None:
            age = (now - meta.valid_from).total_seconds() / 86_400.0
            if age < older_than_days:
                continue
        matched.append(e.id)

    return batch_delete_by_ids(store, matched, relationships=relationships)


# ──────────────────────────── batch export ─────────────────────────────────

def batch_export(
    store: EnrichedMemoryStore,
    ids: List[int],
    *,
    fmt: str = "json",
) -> ExportBundle:
    """Export only the listed memory ids as one bundle.

    Internally builds a transient :class:`_SubsetView` that exposes the
    same surface :mod:`kohaku.portability` expects (``episodic``,
    ``get_metadata``, ``dims``, ``provenance``) restricted to ``ids``.
    """
    if fmt.lower() not in ("json", "markdown", "csv"):
        raise ValueError(
            "fmt must be one of 'json', 'markdown', 'csv', got {!r}".format(fmt)
        )
    wanted = {int(i) for i in ids}
    view = _SubsetView(store, wanted)
    if fmt.lower() == "json":
        return _portability_export_json(view)  # type: ignore[arg-type]
    if fmt.lower() == "markdown":
        return export_markdown(view)  # type: ignore[arg-type]
    return export_csv(view)  # type: ignore[arg-type]


class _SubsetView:
    """Narrow view over an :class:`EnrichedMemoryStore` restricted to a set
    of ids. Implements the minimum surface that the portability exporters
    consume so we can reuse them without copying their logic."""

    def __init__(self, store: EnrichedMemoryStore, wanted: set[int]) -> None:
        self._store = store
        self._wanted = wanted

    @property
    def dims(self) -> int:
        return self._store.dims

    @property
    def episodic(self) -> Any:
        return _SubsetEpisodic(self._store, self._wanted)

    @property
    def provenance(self):
        return self._store.provenance

    def get_metadata(self, entry_id: int):
        if entry_id not in self._wanted:
            return None
        return self._store.get_metadata(entry_id)


class _SubsetEpisodic:
    """Iterable-of-entries view restricted to the wanted id set."""

    def __init__(self, store: EnrichedMemoryStore, wanted: set[int]) -> None:
        self._store = store
        self._wanted = wanted

    def entries(self):
        return [e for e in self._store.episodic.entries() if e.id in self._wanted]
