"""kyro ↔ kohaku bridge — use HDC episodic memory as a RAG retrieval backend.

kyro (the sibling RAG pipeline) defaults to dense + sparse retrieval over
Qdrant. This module exposes an `HDCRetriever` with the minimal surface that
swaps in for, or augments, a vector store: ``ingest()`` to populate, and
``retrieve()`` to fetch top-k matches with optional Ebbinghaus decay.

The retriever owns its own `EpisodicMemory` and keeps a parallel mapping of
``entry_id → original document text`` (HVs are lossy by construction — you
cannot decode a chunk back from its hypervector). Retrieval results carry both
the raw cosine similarity and, when `half_life` is set, the decayed score.

No hard dependency on kyro: this file imports only from `kohaku` so the
bridge ships in this repo and kyro pulls it via `pip install kohaku`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

from kohaku._pure import DIMS, EpisodicMemory
from kohaku._query import query as _episodic_query
from kohaku.attention import encode_text
from kohaku.decay import DecayConfig, query_with_decay


Document = Union[str, dict]


@dataclass(frozen=True)
class RetrievedChunk:
    """One result from `HDCRetriever.retrieve()`.

    Attributes
    ----------
    entry_id:
        EpisodicMemory entry ID. Stable for the lifetime of the retriever.
    doc_id:
        The document's external ID (caller-supplied, or auto-assigned).
    text:
        The original document text — kept verbatim alongside the HV because
        hypervectors are not invertible.
    similarity:
        Raw cosine similarity to the query hypervector, in [-1, 1].
    decayed_similarity:
        Ebbinghaus-weighted score when the retriever was queried with
        a `half_life`; `None` otherwise.
    age:
        Ticks elapsed since the chunk was ingested (newest = 0).
    """

    entry_id: int
    doc_id: str
    text: str
    similarity: float
    decayed_similarity: Optional[float]
    age: int

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "similarity": self.similarity,
            "decayed_similarity": self.decayed_similarity,
            "age": self.age,
        }


class HDCRetriever:
    """Hyperdimensional retrieval backend for kyro-style RAG pipelines.

    Drop-in semantics for a vector store: `ingest(docs)` then `retrieve(query)`.
    All similarity is cosine over bipolar ±1 hypervectors of `dims` dimensions.
    """

    def __init__(self, capacity: int = 10_000, dims: int = DIMS) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if dims <= 0:
            raise ValueError("dims must be > 0")
        self._dims = int(dims)
        self._memory = EpisodicMemory(capacity=capacity)
        # Parallel map: entry_id → (doc_id, text). HVs alone cannot reproduce text.
        self._docs: dict[int, tuple[str, str]] = {}

    # ── basic accessors ────────────────────────────────────────────────────
    @property
    def dims(self) -> int:
        return self._dims

    @property
    def memory(self) -> EpisodicMemory:
        return self._memory

    def __len__(self) -> int:
        return len(self._memory)

    # ── ingest ────────────────────────────────────────────────────────────
    def ingest(self, documents: Iterable[Document]) -> List[int]:
        """Encode each document into a hypervector and store it.

        Parameters
        ----------
        documents:
            Iterable of either raw strings or dicts shaped
            ``{"text": str, "id"?: str}``. Dicts may carry extra fields
            (ignored). Empty text raises `ValueError` — never silently store
            a degenerate hypervector.

        Returns
        -------
        list[int]
            EpisodicMemory entry IDs assigned to each ingested document, in
            input order. Use these IDs to correlate with retrieval results.
        """
        ids: List[int] = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                text = doc
                doc_id = f"doc-{self._memory._next_id}"
            elif isinstance(doc, dict):
                text = doc.get("text", "")
                doc_id = str(doc.get("id", f"doc-{self._memory._next_id}"))
            else:
                raise TypeError(
                    f"document #{i} must be str or dict, got {type(doc).__name__}"
                )
            text = text.strip()
            if not text:
                raise ValueError(f"document #{i} has empty text")
            hv = encode_text(text, dims=self._dims)
            entry_id = self._memory.store(hv, hv, doc_id)
            self._docs[entry_id] = (doc_id, text)
            ids.append(entry_id)
        return ids

    # ── retrieve ──────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        half_life: Optional[float] = None,
        floor: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Top-k associative retrieval, optionally decay-weighted.

        Parameters
        ----------
        query:
            Free-text probe. Encoded with `encode_text`.
        top_k:
            Number of results to return (capped to memory size).
        half_life:
            When set, applies Ebbinghaus exponential decay
            (``weight = 0.5 ** (age / half_life)``) and ranks by the decayed
            score. The raw cosine is still returned alongside.
        floor:
            Minimum decay weight (saturates the curve for very old entries).

        Returns
        -------
        list[RetrievedChunk]
            Sorted by score descending. Score = decayed when `half_life` is
            set, otherwise raw cosine.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if self._memory.is_empty:
            return []

        probe = encode_text(query, dims=self._dims)
        raw_hits = _episodic_query(self._memory, probe, top_k=top_k)
        raw_by_id = {r.entry_id: r.similarity for r in raw_hits}

        if half_life is not None:
            cfg = DecayConfig(half_life=half_life, floor=floor)
            ranked = query_with_decay(self._memory, probe, top_k=top_k, config=cfg)
        else:
            ranked = raw_hits

        # `now` matches kohaku.decay.query_with_decay so age numbers line up
        # exactly with what the decay computation saw.
        now = self._memory._timestamp - 1
        ts_by_id = {e.id: e.timestamp for e in self._memory.entries()}

        out: List[RetrievedChunk] = []
        for r in ranked:
            doc_id, text = self._docs.get(r.entry_id, (r.label, ""))
            age = max(0, now - ts_by_id.get(r.entry_id, now))
            out.append(
                RetrievedChunk(
                    entry_id=r.entry_id,
                    doc_id=doc_id,
                    text=text,
                    similarity=raw_by_id.get(r.entry_id, r.similarity),
                    decayed_similarity=(r.similarity if half_life is not None else None),
                    age=age,
                )
            )
        return out

    # ── housekeeping ──────────────────────────────────────────────────────
    def clear(self) -> None:
        """Drop every chunk. Mainly for tests and tenant resets."""
        self._memory.clear()
        self._docs.clear()
