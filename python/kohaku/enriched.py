"""Enriched memory store — temporal validity + salience + provenance.

Wraps :class:`kohaku.EpisodicMemory` with a parallel metadata table keyed by
``entry_id``. The core HDC engine stays pure; this layer adds:

* **Temporal validity intervals** — every memory has ``valid_from`` and
  ``valid_until``. Queries filter expired items by default.

* **Salience scoring** — each memory carries ``importance`` (0–1) and a
  ``reinforcement_count`` that increments on every retrieval hit. The
  composite salience score is

      salience = importance · recency_decay · (1 + reinforcement_count · k) · trust(source)

  where ``recency_decay = 0.5 ** (age_days / half_life)`` (Ebbinghaus) and
  ``trust(source)`` is read from :data:`SOURCE_TRUST_WEIGHTS`.

* **Provenance** — every memory carries a ``source`` string (e.g.
  ``"user_input"``, ``"web_search"``, ``"tool_result"``, ``"agent_inference"``).
  Queries can filter by source. Memories from ``agent_inference`` receive a
  reduced default trust weight (0.5) — a baseline defense against
  agent-generated memory poisoning.

This module does not modify ``MemoryEntry`` or the ``.hkb`` binary format —
the metadata lives only in the wrapper. That keeps the core engine and the
LCG / Rust contract unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from kohaku._pure import DIMS, EpisodicMemory, HyperVector

# Default trust weights per source. Agent-generated memories are trusted
# less by default — they don't fail closed, just rank lower under salience.
SOURCE_TRUST_WEIGHTS: Dict[str, float] = {
    "user_input": 1.0,
    "tool_result": 0.9,
    "web_search": 0.8,
    "agent_inference": 0.5,
}

# Default reinforcement scaling — every retrieval bumps salience by `k`.
DEFAULT_REINFORCEMENT_K: float = 0.1
DEFAULT_HALF_LIFE_DAYS: float = 30.0
DEFAULT_IMPORTANCE: float = 0.5

SortMode = Literal["similarity", "salience", "recency"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """Promote naive datetimes to UTC-aware so comparisons are well-defined."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _normalise_tag(tag: str) -> str:
    """Lower-cased, whitespace-stripped tag (≤ 64 chars). Empty if invalid."""
    if not isinstance(tag, str):
        return ""
    cleaned = tag.strip().lower()
    return cleaned[:64]


@dataclass
class MemoryMetadata:
    """Per-memory metadata maintained outside :class:`EpisodicMemory`.

    All datetimes are stored as UTC-aware. Naive datetimes are promoted on
    write so downstream comparisons never raise ``TypeError``.
    """
    entry_id: int
    valid_from: datetime
    valid_until: Optional[datetime] = None
    source: str = "user_input"
    importance: float = DEFAULT_IMPORTANCE
    reinforcement_count: int = 0
    created_at: datetime = field(default_factory=_utcnow)
    tags: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if self.reinforcement_count < 0:
            raise ValueError("reinforcement_count must be >= 0")
        self.valid_from = _aware(self.valid_from)
        if self.valid_until is not None:
            self.valid_until = _aware(self.valid_until)
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until must be >= valid_from")
        self.created_at = _aware(self.created_at)
        if self.source == "":
            raise ValueError("source must be non-empty")
        # Coerce tags to a set of normalised non-empty strings. Empty tags are
        # silently dropped — they're never useful for filtering.
        self.tags = {_normalise_tag(t) for t in self.tags if _normalise_tag(t)}

    def is_valid_at(self, now: datetime) -> bool:
        """True if this memory is currently active at ``now``."""
        now = _aware(now)
        if now < self.valid_from:
            return False
        if self.valid_until is not None and now > self.valid_until:
            return False
        return True

    def age_days(self, now: datetime) -> float:
        """Days elapsed since ``created_at`` (used by the recency decay)."""
        now = _aware(now)
        delta = now - self.created_at
        return max(0.0, delta.total_seconds() / 86_400.0)

    def trust(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights if weights is not None else SOURCE_TRUST_WEIGHTS
        return w.get(self.source, 0.5)

    def salience(
        self,
        *,
        now: datetime,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        reinforcement_k: float = DEFAULT_REINFORCEMENT_K,
        trust_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Composite salience score.

        salience = importance · 0.5^(age / half_life) · (1 + count · k) · trust(source)
        """
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        age = self.age_days(now)
        recency = 0.5 ** (age / half_life_days)
        reinforcement = 1.0 + self.reinforcement_count * reinforcement_k
        return self.importance * recency * reinforcement * self.trust(trust_weights)


@dataclass(frozen=True)
class EnrichedRetrievalResult:
    """One row of an enriched query result."""
    entry_id: int
    label: str
    similarity: float
    salience: float
    source: str
    importance: float
    reinforcement_count: int
    valid_from: datetime
    valid_until: Optional[datetime]
    trust: float
    value: HyperVector
    tags: tuple = ()  # immutable view for the frozen dataclass

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "label": self.label,
            "similarity": round(float(self.similarity), 6),
            "salience": round(float(self.salience), 6),
            "source": self.source,
            "importance": float(self.importance),
            "reinforcement_count": int(self.reinforcement_count),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "trust": float(self.trust),
            "tags": list(self.tags),
        }


class EnrichedMemoryStore:
    """Episodic memory + per-entry temporal / salience / provenance metadata.

    Parameters
    ----------
    capacity:
        Underlying ``EpisodicMemory`` capacity.
    dims:
        Hypervector dimensionality.
    half_life_days:
        Default half-life used by salience's recency-decay term.
    reinforcement_k:
        Per-retrieval salience bump (`1 + count · k`).
    trust_weights:
        Override for :data:`SOURCE_TRUST_WEIGHTS`.
    """

    def __init__(
        self,
        capacity: int = 1000,
        dims: int = DIMS,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        reinforcement_k: float = DEFAULT_REINFORCEMENT_K,
        trust_weights: Optional[Dict[str, float]] = None,
        provenance: "Optional[object]" = None,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        if reinforcement_k < 0:
            raise ValueError("reinforcement_k must be >= 0")
        self._mem = EpisodicMemory(capacity=capacity)
        self._meta: Dict[int, MemoryMetadata] = {}
        self._dims = dims
        self.half_life_days = half_life_days
        self.reinforcement_k = reinforcement_k
        self.trust_weights = (
            dict(trust_weights) if trust_weights is not None else dict(SOURCE_TRUST_WEIGHTS)
        )
        # Optional provenance graph — when attached, every `store()` records a
        # lineage row. Typed as `object` to avoid a circular import at module
        # load; runtime duck-types `.record(memory_id, parent_ids, source_type)`.
        self.provenance = provenance

    # ── basic accessors ────────────────────────────────────────────────────
    @property
    def dims(self) -> int:
        return self._dims

    @property
    def episodic(self) -> EpisodicMemory:
        return self._mem

    def __len__(self) -> int:
        return len(self._mem)

    def get_metadata(self, entry_id: int) -> Optional[MemoryMetadata]:
        return self._meta.get(entry_id)

    def __contains__(self, entry_id: int) -> bool:
        return entry_id in self._meta

    # ── store ──────────────────────────────────────────────────────────────
    def store(
        self,
        key: HyperVector,
        value: HyperVector,
        label: str = "",
        *,
        source: str = "user_input",
        importance: float = DEFAULT_IMPORTANCE,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        parent_ids: Optional[List[int]] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Store an entry plus its metadata. Returns the new entry id.

        When :attr:`provenance` is attached the new entry is auto-recorded
        as a lineage node. ``parent_ids`` is forwarded to the graph and
        defaults to an empty list (= a root memory). The ``source`` field
        is reused as the graph's ``source_type``. ``tags`` are normalised
        to lowercase and deduped at the metadata boundary.
        """
        eid = self._mem.store(key, value, label)
        # If capacity caused FIFO eviction, drop the matching metadata row.
        live_ids = {e.id for e in self._mem.entries()}
        for old_id in list(self._meta.keys()):
            if old_id not in live_ids:
                del self._meta[old_id]
        self._meta[eid] = MemoryMetadata(
            entry_id=eid,
            valid_from=valid_from or _utcnow(),
            valid_until=valid_until,
            source=source,
            importance=importance,
            tags=set(tags or []),
        )
        if self.provenance is not None:
            self.provenance.record(
                memory_id=eid,
                parent_ids=list(parent_ids or []),
                source_type=source,
                metadata={"label": label},
            )
        return eid

    # ── tagging ───────────────────────────────────────────────────────────
    def add_tags(self, entry_id: int, tags: List[str]) -> Optional[set]:
        """Union the given tags into the entry's tag set. Returns the new
        set, or None if the entry is unknown."""
        meta = self._meta.get(entry_id)
        if meta is None:
            return None
        for t in tags:
            normalised = _normalise_tag(t)
            if normalised:
                meta.tags.add(normalised)
        return set(meta.tags)

    def remove_tags(self, entry_id: int, tags: List[str]) -> Optional[set]:
        """Difference the given tags from the entry's tag set. Returns the
        new set, or None if the entry is unknown."""
        meta = self._meta.get(entry_id)
        if meta is None:
            return None
        for t in tags:
            meta.tags.discard(_normalise_tag(t))
        return set(meta.tags)

    def get_tags(self, entry_id: int) -> Optional[set]:
        meta = self._meta.get(entry_id)
        return None if meta is None else set(meta.tags)

    def all_tags(self) -> Dict[str, int]:
        """Aggregate tag → count over all live memories."""
        counts: Dict[str, int] = {}
        for meta in self._meta.values():
            for t in meta.tags:
                counts[t] = counts.get(t, 0) + 1
        return counts

    # ── retrieval ──────────────────────────────────────────────────────────
    def query(
        self,
        query_key: HyperVector,
        top_k: int = 5,
        *,
        sort: SortMode = "similarity",
        source_filter: Optional[str] = None,
        include_expired: bool = False,
        min_similarity: Optional[float] = None,
        now: Optional[datetime] = None,
        reinforce_hits: bool = True,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
    ) -> List[EnrichedRetrievalResult]:
        """Return up to ``top_k`` matches, filtered by validity and (optionally)
        source, ranked by ``sort``.

        ``reinforce_hits=True`` increments ``reinforcement_count`` for every
        result returned — that's the engine of the salience feedback loop.
        """
        if top_k <= 0:
            return []
        now = _aware(now or _utcnow())
        any_set = {_normalise_tag(t) for t in (tags_any or []) if _normalise_tag(t)}
        all_set = {_normalise_tag(t) for t in (tags_all or []) if _normalise_tag(t)}

        scored: List[EnrichedRetrievalResult] = []
        for e in self._mem.entries():
            meta = self._meta.get(e.id)
            if meta is None:
                continue
            if not include_expired and not meta.is_valid_at(now):
                continue
            if source_filter is not None and meta.source != source_filter:
                continue
            if any_set and not (meta.tags & any_set):
                continue
            if all_set and not all_set.issubset(meta.tags):
                continue
            sim = float(e.key.cosine_similarity(query_key))
            if min_similarity is not None and sim < min_similarity:
                continue
            sal = meta.salience(
                now=now,
                half_life_days=self.half_life_days,
                reinforcement_k=self.reinforcement_k,
                trust_weights=self.trust_weights,
            )
            scored.append(
                EnrichedRetrievalResult(
                    entry_id=e.id,
                    label=e.label,
                    similarity=sim,
                    salience=sal,
                    source=meta.source,
                    importance=meta.importance,
                    reinforcement_count=meta.reinforcement_count,
                    valid_from=meta.valid_from,
                    valid_until=meta.valid_until,
                    trust=meta.trust(self.trust_weights),
                    value=e.value,
                    tags=tuple(sorted(meta.tags)),
                )
            )

        if sort == "salience":
            scored.sort(key=lambda r: r.salience, reverse=True)
        elif sort == "recency":
            scored.sort(key=lambda r: r.valid_from, reverse=True)
        else:  # similarity (default)
            scored.sort(key=lambda r: r.similarity, reverse=True)

        head = scored[:top_k]

        if reinforce_hits:
            for r in head:
                self.reinforce(r.entry_id)

        return head

    # ── listing without a query vector ─────────────────────────────────────
    def list_memories(
        self,
        *,
        sort: SortMode = "recency",
        source_filter: Optional[str] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
        now: Optional[datetime] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
    ) -> List[dict]:
        """Inventory all memories with their full metadata.

        Ranking is computed without a query vector — ``sort='similarity'`` is
        accepted but falls back to ``'salience'`` since there's nothing to be
        similar to.
        """
        now = _aware(now or _utcnow())
        any_set = {_normalise_tag(t) for t in (tags_any or []) if _normalise_tag(t)}
        all_set = {_normalise_tag(t) for t in (tags_all or []) if _normalise_tag(t)}
        items: List[dict] = []
        for e in self._mem.entries():
            meta = self._meta.get(e.id)
            if meta is None:
                continue
            if not include_expired and not meta.is_valid_at(now):
                continue
            if source_filter is not None and meta.source != source_filter:
                continue
            if any_set and not (meta.tags & any_set):
                continue
            if all_set and not all_set.issubset(meta.tags):
                continue
            sal = meta.salience(
                now=now,
                half_life_days=self.half_life_days,
                reinforcement_k=self.reinforcement_k,
                trust_weights=self.trust_weights,
            )
            items.append({
                "entry_id": e.id,
                "label": e.label,
                "timestamp": e.timestamp,
                "salience": round(float(sal), 6),
                "source": meta.source,
                "importance": float(meta.importance),
                "reinforcement_count": int(meta.reinforcement_count),
                "trust": meta.trust(self.trust_weights),
                "valid_from": meta.valid_from.isoformat(),
                "valid_until": meta.valid_until.isoformat() if meta.valid_until else None,
                "created_at": meta.created_at.isoformat(),
                "tags": sorted(meta.tags),
            })
        if sort == "recency":
            items.sort(key=lambda r: r["valid_from"], reverse=True)
        else:
            items.sort(key=lambda r: r["salience"], reverse=True)
        if limit is not None:
            items = items[: max(0, int(limit))]
        return items

    # ── reinforcement ──────────────────────────────────────────────────────
    def reinforce(self, entry_id: int, delta: int = 1) -> Optional[MemoryMetadata]:
        """Increment ``reinforcement_count`` for an entry. No-op if not found."""
        meta = self._meta.get(entry_id)
        if meta is None:
            return None
        meta.reinforcement_count += max(0, int(delta))
        return meta

    # ── expiry ─────────────────────────────────────────────────────────────
    def expire_old(self, *, now: Optional[datetime] = None) -> List[int]:
        """Drop entries whose ``valid_until`` has passed. Returns dropped ids."""
        now = _aware(now or _utcnow())
        # Find entries to drop.
        drop_ids = [
            eid for eid, meta in self._meta.items()
            if meta.valid_until is not None and now > meta.valid_until
        ]
        if not drop_ids:
            return []
        drop_set = set(drop_ids)
        # Rebuild EpisodicMemory's _entries list in place. This is a wrapper
        # responsibility — EpisodicMemory itself has no public removal API and
        # we don't want to expose one (would break the FIFO/timestamp contract).
        kept = [e for e in self._mem._entries if e.id not in drop_set]
        self._mem._entries = kept
        for eid in drop_ids:
            self._meta.pop(eid, None)
        return drop_ids

    def clear(self) -> None:
        self._mem.clear()
        self._meta.clear()
