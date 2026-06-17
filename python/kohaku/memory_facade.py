"""``Memory`` — the one-line front door to kohaku.

Everything else in the package is composable machinery: hypervectors,
episodic stores, salience, decay, consolidation, provenance. Most callers
just want *store a string, get strings back*. This facade wires the enriched
store + deterministic text encoding into that minimal surface:

    >>> from kohaku import Memory
    >>> mem = Memory()
    >>> mem.store("User prefers Italian wine")
    0
    >>> hits = mem.query("What does the user like?")
    >>> hits[0].text
    'User prefers Italian wine'

Text is encoded to a bipolar hypervector via :func:`kohaku.encode_text`
(deterministic LCG token-bundling), so similarity reflects token overlap.
Because the encoding is deterministic, :meth:`save` / :meth:`load` round-trip
the full store from labels + metadata alone — no hypervectors are written to
disk.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

from kohaku._pure import DIMS
from kohaku.attention import encode_text
from kohaku.enriched import EnrichedMemoryStore
from kohaku.enriched_meta import DEFAULT_HALF_LIFE_DAYS, DEFAULT_IMPORTANCE, SortMode

__all__ = ["Memory", "MemoryHit"]

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MemoryHit:
    """A single retrieval result from :meth:`Memory.query`.

    ``score`` mirrors whichever ranking was requested (similarity by default,
    salience when ``sort='salience'``) so callers can sort/threshold on one
    field without caring which mode produced it.
    """
    id: int
    text: str
    score: float
    similarity: float
    salience: float
    source: str
    importance: float
    tags: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": round(float(self.score), 6),
            "similarity": round(float(self.similarity), 6),
            "salience": round(float(self.salience), 6),
            "source": self.source,
            "importance": float(self.importance),
            "tags": list(self.tags),
        }


class Memory:
    """String-in / string-out episodic memory.

    Thin wrapper over :class:`kohaku.EnrichedMemoryStore` that handles text
    encoding and exposes only what a typical LLM-memory caller needs. For the
    full surface (provenance graphs, version history, typed relationships,
    consolidation daemons) reach for ``EnrichedMemoryStore`` directly.

    Parameters
    ----------
    capacity:
        Maximum number of memories before FIFO eviction.
    dims:
        Hypervector dimensionality (must match across save/load).
    half_life_days:
        Recency half-life used by salience scoring.
    """

    def __init__(
        self,
        capacity: int = 1000,
        *,
        dims: int = DIMS,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        self._dims = dims
        self._capacity = capacity
        self._store = EnrichedMemoryStore(
            capacity=capacity, dims=dims, half_life_days=half_life_days
        )

    # ── write ───────────────────────────────────────────────────────────────
    def store(
        self,
        text: str,
        *,
        source: str = "user_input",
        importance: float = DEFAULT_IMPORTANCE,
        tags: Optional[Sequence[str]] = None,
        valid_until: Optional[datetime] = None,
        forgetting_rate: Optional[float] = None,
    ) -> int:
        """Encode ``text`` and store it. Returns the new memory id.

        The text becomes both the retrieval key and the stored value (the
        memory *is* its own content), and doubles as the human-readable label.
        """
        if not text or not text.strip():
            raise ValueError("cannot store empty text")
        hv = encode_text(text, dims=self._dims)
        return self._store.store(
            hv,
            hv,
            text,
            source=source,
            importance=importance,
            tags=list(tags) if tags is not None else None,
            valid_until=valid_until,
            forgetting_rate=forgetting_rate,
        )

    # ── read ────────────────────────────────────────────────────────────────
    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        sort: SortMode = "similarity",
        source: Optional[str] = None,
        include_expired: bool = False,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        reinforce: bool = True,
    ) -> List[MemoryHit]:
        """Retrieve the ``top_k`` memories most relevant to ``text``.

        ``sort`` selects the ranking ("similarity", "salience", or "recency").
        ``reinforce=True`` bumps the reinforcement count of every returned hit,
        driving the salience feedback loop; pass ``False`` for read-only probes.
        """
        if not text or not text.strip():
            raise ValueError("cannot query with empty text")
        hv = encode_text(text, dims=self._dims)
        results = self._store.query(
            hv,
            top_k=top_k,
            sort=sort,
            source_filter=source,
            include_expired=include_expired,
            tags_any=list(tags_any) if tags_any is not None else None,
            tags_all=list(tags_all) if tags_all is not None else None,
            reinforce_hits=reinforce,
        )
        score_key = "salience" if sort == "salience" else "similarity"
        return [
            MemoryHit(
                id=r.entry_id,
                text=r.label,
                score=float(getattr(r, score_key)),
                similarity=float(r.similarity),
                salience=float(r.salience),
                source=r.source,
                importance=float(r.importance),
                tags=tuple(r.tags),
            )
            for r in results
        ]

    # ── maintenance ───────────────────────────────────────────────────────────
    def reinforce(self, memory_id: int, delta: int = 1) -> None:
        """Strengthen a memory (raises its salience). No-op if unknown."""
        self._store.reinforce(memory_id, delta=delta)

    def expire(self, *, now: Optional[datetime] = None) -> List[int]:
        """Drop memories past their ``valid_until``. Returns the dropped ids."""
        return self._store.expire_old(now=now)

    def clear(self) -> None:
        self._store.clear()

    @property
    def store_(self) -> EnrichedMemoryStore:
        """Escape hatch to the underlying enriched store for advanced use."""
        return self._store

    def __len__(self) -> int:
        return len(self._store)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist the store to a JSON file.

        Only labels + metadata are written; hypervectors are re-derived on
        load via the deterministic encoder, so the round-trip is exact.
        """
        records = []
        for meta_dict in self._store.list_memories(include_expired=True, sort="recency"):
            meta = self._store.get_metadata(meta_dict["entry_id"])
            if meta is None:  # pragma: no cover - defensive
                continue
            records.append({
                "label": meta_dict["label"],
                "source": meta.source,
                "importance": meta.importance,
                "reinforcement_count": meta.reinforcement_count,
                "created_at": meta.created_at.isoformat(),
                "valid_from": meta.valid_from.isoformat(),
                "valid_until": meta.valid_until.isoformat() if meta.valid_until else None,
                "tags": sorted(meta.tags),
                "forgetting_rate": meta.forgetting_rate,
            })
        payload = {
            "schema": _SCHEMA_VERSION,
            "dims": self._dims,
            "capacity": self._capacity,
            "half_life_days": self._store.half_life_days,
            "records": records,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Memory":
        """Reconstruct a :class:`Memory` previously written by :meth:`save`."""
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        mem = cls(
            capacity=int(payload.get("capacity", 1000)),
            dims=int(payload.get("dims", DIMS)),
            half_life_days=float(payload.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)),
        )
        for rec in payload.get("records", []):
            eid = mem.store(
                rec["label"],
                source=rec.get("source", "user_input"),
                importance=float(rec.get("importance", DEFAULT_IMPORTANCE)),
                tags=rec.get("tags") or None,
                valid_until=_parse_dt(rec.get("valid_until")),
                forgetting_rate=rec.get("forgetting_rate"),
            )
            # Restore salience-affecting fields the store would otherwise reset.
            meta = mem._store.get_metadata(eid)
            if meta is not None:
                meta.reinforcement_count = int(rec.get("reinforcement_count", 0))
                created = _parse_dt(rec.get("created_at"))
                if created is not None:
                    meta.created_at = created
                valid_from = _parse_dt(rec.get("valid_from"))
                if valid_from is not None:
                    meta.valid_from = valid_from
        return mem


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)
