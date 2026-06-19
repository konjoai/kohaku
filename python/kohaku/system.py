"""Unified system snapshot — save/load a whole enriched setup as one bundle (B3).

A production kohaku deployment is more than its hypervectors: there's the
episodic store, the per-memory metadata (validity, salience, tags), and up to
three SQLite-backed side stores — provenance lineage, version history, and
typed relationships. Previously each persisted separately (``.hkb`` here, three
loose ``.db`` files there) with nothing tying them together.

:func:`save_system` writes all of it into a single directory with a
``manifest.json``; :func:`load_system` reads it back into a wired-up
:class:`~kohaku.enriched.EnrichedMemoryStore` plus the side stores. The
round-trip is exact — hypervectors come from the packed ``.hkb`` and the
SQLite stores are copied byte-for-byte via the sqlite backup API (so even
``:memory:`` stores persist).

    >>> from kohaku.system import save_system, load_system
    >>> save_system(store, "snap/", provenance=pg, versions=vs)   # doctest: +SKIP
    >>> bundle = load_system("snap/")                             # doctest: +SKIP
    >>> bundle.store, bundle.provenance, bundle.versions          # doctest: +SKIP
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from kohaku.enriched import EnrichedMemoryStore
from kohaku.enriched_meta import MemoryMetadata, _aware
from kohaku.persistence import load_binary, save_binary
from kohaku.provenance import ProvenanceGraph
from kohaku.relationships import RelationshipStore
from kohaku.versions import VersionStore

SCHEMA_VERSION = 1
_MEMORY_FILE = "memory.hkb"
_METADATA_FILE = "metadata.json"
_MANIFEST_FILE = "manifest.json"
_PROVENANCE_DB = "provenance.db"
_VERSIONS_DB = "versions.db"
_RELATIONSHIPS_DB = "relationships.db"


@dataclass
class SystemBundle:
    """The reconstructed components returned by :func:`load_system`."""

    store: EnrichedMemoryStore
    provenance: Optional[ProvenanceGraph] = None
    versions: Optional[VersionStore] = None
    relationships: Optional[RelationshipStore] = None


def _meta_to_dict(meta: MemoryMetadata) -> Dict[str, Any]:
    return {
        "entry_id": meta.entry_id,
        "valid_from": meta.valid_from.isoformat(),
        "valid_until": meta.valid_until.isoformat() if meta.valid_until else None,
        "source": meta.source,
        "importance": meta.importance,
        "reinforcement_count": meta.reinforcement_count,
        "created_at": meta.created_at.isoformat(),
        "tags": sorted(meta.tags),
        "forgetting_rate": meta.forgetting_rate,
    }


def _meta_from_dict(d: Dict[str, Any]) -> MemoryMetadata:
    return MemoryMetadata(
        entry_id=int(d["entry_id"]),
        valid_from=datetime.fromisoformat(d["valid_from"]),
        valid_until=datetime.fromisoformat(d["valid_until"])
        if d.get("valid_until")
        else None,
        source=d.get("source", "user_input"),
        importance=float(d.get("importance", 0.5)),
        reinforcement_count=int(d.get("reinforcement_count", 0)),
        created_at=datetime.fromisoformat(d["created_at"])
        if d.get("created_at")
        else _aware(datetime.now(timezone.utc)),
        tags=set(d.get("tags", [])),
        forgetting_rate=d.get("forgetting_rate"),
    )


def _backup_sqlite(store: object, dest: Path) -> None:
    """Copy a SQLite-backed store's database to ``dest`` (works for :memory:)."""
    if dest.exists():
        dest.unlink()
    target = sqlite3.connect(str(dest))
    try:
        store._conn.backup(target)  # type: ignore[attr-defined]
    finally:
        target.close()


def save_system(
    store: EnrichedMemoryStore,
    directory: "str | os.PathLike[str]",
    *,
    provenance: Optional[ProvenanceGraph] = None,
    versions: Optional[VersionStore] = None,
    relationships: Optional[RelationshipStore] = None,
) -> None:
    """Write the full system (episodic + metadata + side stores) to ``directory``.

    The side stores default to those already attached to ``store``
    (``store.provenance`` / ``store.versions``) when not passed explicitly.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    save_binary(store.episodic, directory / _MEMORY_FILE)

    metadata = [
        _meta_to_dict(cast(MemoryMetadata, store.get_metadata(e.id)))
        for e in store.episodic.entries()
        if store.get_metadata(e.id) is not None
    ]
    _atomic_json(directory / _METADATA_FILE, {"records": metadata})

    provenance = (
        provenance if provenance is not None else getattr(store, "provenance", None)
    )
    versions = versions if versions is not None else getattr(store, "versions", None)

    components: List[str] = [_MEMORY_FILE, _METADATA_FILE]
    if isinstance(provenance, ProvenanceGraph):
        _backup_sqlite(provenance, directory / _PROVENANCE_DB)
        components.append(_PROVENANCE_DB)
    if isinstance(versions, VersionStore):
        _backup_sqlite(versions, directory / _VERSIONS_DB)
        components.append(_VERSIONS_DB)
    if isinstance(relationships, RelationshipStore):
        _backup_sqlite(relationships, directory / _RELATIONSHIPS_DB)
        components.append(_RELATIONSHIPS_DB)

    manifest = {
        "schema": SCHEMA_VERSION,
        "dims": store.dims,
        "capacity": store.episodic._capacity,
        "half_life_days": store.half_life_days,
        "reinforcement_k": store.reinforcement_k,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "num_memories": len(store),
    }
    _atomic_json(directory / _MANIFEST_FILE, manifest)


def load_system(directory: "str | os.PathLike[str]") -> SystemBundle:
    """Reconstruct a :class:`SystemBundle` written by :func:`save_system`."""
    directory = Path(directory)
    manifest_path = directory / _MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {_MANIFEST_FILE} in {directory}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    memory = load_binary(directory / _MEMORY_FILE)
    with open(directory / _METADATA_FILE, encoding="utf-8") as fh:
        meta_payload = json.load(fh)
    metadata: Dict[int, MemoryMetadata] = {}
    for rec in meta_payload.get("records", []):
        meta = _meta_from_dict(rec)
        metadata[meta.entry_id] = meta

    store = EnrichedMemoryStore.from_state(
        memory,
        metadata,
        capacity=int(manifest.get("capacity", 1000)),
        dims=int(manifest["dims"]),
        half_life_days=float(manifest.get("half_life_days", 30.0)),
        reinforcement_k=float(manifest.get("reinforcement_k", 0.1)),
    )

    components = set(manifest.get("components", []))
    provenance = versions = relationships = None
    if _PROVENANCE_DB in components:
        provenance = ProvenanceGraph(directory / _PROVENANCE_DB)
        store.provenance = provenance
    if _VERSIONS_DB in components:
        versions = VersionStore(directory / _VERSIONS_DB)
        store.versions = versions
    if _RELATIONSHIPS_DB in components:
        relationships = RelationshipStore(directory / _RELATIONSHIPS_DB)

    return SystemBundle(
        store=store,
        provenance=provenance,
        versions=versions,
        relationships=relationships,
    )


def _atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
