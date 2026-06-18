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
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

from kohaku._pure import DIMS, HyperVector
from kohaku.analogy import AnalogicalMemory, AnalogyResult
from kohaku.extraction import Triple
from kohaku.ann import LSHIndex
from kohaku.attention import encode_text
from kohaku.compositional import complete_cue, compose
from kohaku.enriched import EnrichedMemoryStore
from kohaku.enriched_meta import DEFAULT_HALF_LIFE_DAYS, DEFAULT_IMPORTANCE, SortMode

logger = logging.getLogger(__name__)

Encoder = Callable[[str], HyperVector]

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
    encoder:
        Optional ``str -> HyperVector`` callable. Defaults to the lexical
        :func:`kohaku.encode_text` (token-overlap similarity). Pass a
        :class:`kohaku.semantic.EmbeddingEncoder` for meaning-based recall.
        A store written with a custom encoder must be reloaded with the same
        one (see :meth:`load`).
    ann:
        When True, maintain a bipolar-LSH index (:class:`kohaku.ann.LSHIndex`)
        and use it to narrow similarity queries to a candidate set before exact
        ranking — sub-linear retrieval past ~10⁴ memories. Exact cosine still
        ranks the candidates, so results are unchanged except for the rare LSH
        miss; salience/recency sorts and empty candidate sets fall back to a
        full scan.
    """

    def __init__(
        self,
        capacity: int = 1000,
        *,
        dims: int = DIMS,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        encoder: Optional[Encoder] = None,
        ann: bool = False,
    ) -> None:
        self._dims = dims
        self._capacity = capacity
        self._encoder = encoder
        self._store = EnrichedMemoryStore(
            capacity=capacity, dims=dims, half_life_days=half_life_days
        )
        self._index = LSHIndex(dims) if ann else None
        self._analogical: Optional[AnalogicalMemory] = None

    @property
    def ann_enabled(self) -> bool:
        return self._index is not None

    # ── relational reasoning (Track D) ───────────────────────────────────────
    @property
    def analogical(self) -> AnalogicalMemory:
        """The structured :class:`AnalogicalMemory` for relational reasoning.

        Lazily created. Records are kept alongside the episodic store and
        persisted by :meth:`save`. This is the algebra-over-memory surface that
        plain cosine retrieval can't provide.
        """
        if self._analogical is None:
            self._analogical = AnalogicalMemory(dims=self._dims)
        return self._analogical

    def add_record(self, name: str, fields: dict) -> None:
        """Store a structured record (``{attribute: value}``) for reasoning.

        Distinct from :meth:`store` (free-text episodic memory): records power
        :meth:`attribute` (recall a field) and :meth:`analogy` (relational
        transfer), e.g. ``analogy("USA", "Mexico", "dollar") -> "peso"``.
        """
        self.analogical.add_record(name, fields)

    def attribute(self, name: str, attribute: str) -> AnalogyResult:
        """Recall the value of ``attribute`` in record ``name`` (unbind + cleanup)."""
        return self.analogical.get(name, attribute)

    def analogy(self, source: str, target: str, value: str) -> AnalogyResult:
        """Analogical transfer: "the ``value`` of ``source`` is to ``target`` as…"."""
        return self.analogical.analogy(source, target, value)

    def learn(
        self,
        text: str,
        *,
        source: str = "user_input",
        importance: float = DEFAULT_IMPORTANCE,
        tags: Optional[Sequence[str]] = None,
    ) -> Tuple[int, List[Triple]]:
        """Ingest ``text`` as both episodic memory *and* structured knowledge.

        Stores the prose verbatim (like :meth:`store`) and, in the same call,
        extracts ``(subject, attribute, value)`` triples into the analogical
        store — so a single "learn this" feeds both free-text recall and
        relational reasoning. Returns ``(memory_id, triples_learned)``; the
        triple list is empty when nothing parsed (no fabricated structure).
        """
        eid = self.store(text, source=source, importance=importance, tags=tags)
        return eid, self.analogical.learn(text)

    def _rebuild_index(self) -> None:
        if self._index is None:
            return
        self._index.clear()
        for e in self._store.episodic.entries():
            self._index.add(e.id, e.key)

    def _encode(self, text: str) -> HyperVector:
        if self._encoder is not None:
            return self._encoder(text)
        return encode_text(text, dims=self._dims)

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
        hv = self._encode(text)
        n_before = len(self._store)
        eid = self._store.store(
            hv,
            hv,
            text,
            source=source,
            importance=importance,
            tags=list(tags) if tags is not None else None,
            valid_until=valid_until,
            forgetting_rate=forgetting_rate,
        )
        if self._index is not None:
            # len not growing means a FIFO eviction fired — rebuild from the
            # live entries so the evicted id leaves the index too.
            if len(self._store) <= n_before:
                self._rebuild_index()
            else:
                self._index.add(eid, hv)
        return eid

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
        hv = self._encode(text)
        return self._query_hv(
            hv,
            top_k,
            sort=sort,
            source=source,
            include_expired=include_expired,
            tags_any=tags_any,
            tags_all=tags_all,
            reinforce=reinforce,
        )

    def recall_composite(
        self,
        cues: Sequence[str],
        top_k: int = 5,
        *,
        cleanup: bool = False,
        sort: SortMode = "similarity",
        source: Optional[str] = None,
        include_expired: bool = False,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        reinforce: bool = True,
    ) -> List[MemoryHit]:
        """Retrieve memories matching *all* of several cues (a soft conjunction).

        Each cue is encoded and bundled into one composite query, so the memory
        closest to the combination ranks highest — multi-constraint recall in a
        single pass. With ``cleanup=True`` the composite is first pulled toward
        the nearest stored memory by a Hopfield associator (pattern completion),
        which makes recall robust to noisy or partial cues at ``O(N·D)`` cost.
        """
        texts = [c for c in cues if c and c.strip()]
        if not texts:
            raise ValueError("recall_composite requires at least one non-empty cue")
        composite = compose([self._encode(t) for t in texts])
        if cleanup:
            keys = [e.key for e in self._store.episodic.entries()]
            composite = complete_cue(composite, keys)
        return self._query_hv(
            composite,
            top_k,
            sort=sort,
            source=source,
            include_expired=include_expired,
            tags_any=tags_any,
            tags_all=tags_all,
            reinforce=reinforce,
        )

    def _query_hv(
        self,
        hv: HyperVector,
        top_k: int,
        *,
        sort: SortMode,
        source: Optional[str],
        include_expired: bool,
        tags_any: Optional[Sequence[str]],
        tags_all: Optional[Sequence[str]],
        reinforce: bool,
    ) -> List[MemoryHit]:
        """Shared retrieval core: ANN-narrow, score, and wrap as ``MemoryHit``s."""
        # ANN narrows the scan for similarity queries; empty candidate sets and
        # non-similarity sorts fall back to a full exact scan (candidate_ids=None).
        candidate_ids = None
        if self._index is not None and sort == "similarity":
            cand = self._index.candidates(hv)
            if cand:
                candidate_ids = cand
        results = self._store.query(
            hv,
            top_k=top_k,
            sort=sort,
            source_filter=source,
            include_expired=include_expired,
            tags_any=list(tags_any) if tags_any is not None else None,
            tags_all=list(tags_all) if tags_all is not None else None,
            reinforce_hits=reinforce,
            candidate_ids=candidate_ids,
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
        dropped = self._store.expire_old(now=now)
        if dropped and self._index is not None:
            for eid in dropped:
                self._index.remove(eid)
        return dropped

    def clear(self) -> None:
        self._store.clear()
        if self._index is not None:
            self._index.clear()

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
            "encoder": "custom" if self._encoder is not None else "lexical",
            "records": records,
            "analogical": self._analogical.to_dict() if self._analogical else None,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(
        cls, path: str, *, encoder: Optional[Encoder] = None, ann: bool = False
    ) -> "Memory":
        """Reconstruct a :class:`Memory` previously written by :meth:`save`.

        Hypervectors are re-derived by re-encoding the stored labels, so a
        store saved with a custom ``encoder`` must be reloaded with the same
        one — otherwise similarity scores won't match the original. A mismatch
        (custom-saved store, no encoder supplied) is logged as a warning.
        ``ann`` rebuilds the LSH index as entries are re-stored.
        """
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("encoder") == "custom" and encoder is None:
            logger.warning(
                "loading a store saved with a custom encoder but none was "
                "supplied; re-encoding with the default lexical encoder — "
                "similarity scores will differ from the original."
            )
        mem = cls(
            capacity=int(payload.get("capacity", 1000)),
            dims=int(payload.get("dims", DIMS)),
            half_life_days=float(payload.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)),
            encoder=encoder,
            ann=ann,
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
        analogical = payload.get("analogical")
        if analogical:
            mem._analogical = AnalogicalMemory.from_dict(analogical)
        return mem


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)
