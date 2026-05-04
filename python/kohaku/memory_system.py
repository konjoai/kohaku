"""Episodic vs semantic memory distinction — combined two-store system.

Two complementary stores, modeled on the human memory split (Tulving 1972):

* **Episodic** (:class:`kohaku.EpisodicMemory`) — raw experiences with
  timestamps. Subject to forgetting curves; capacity-limited; FIFO eviction.
  This is "I had a coffee this morning at 8:14am."

* **Semantic** (:class:`kohaku.learning.ItemMemory`) — distilled prototypes
  built by online HDC learning. Stable; no decay; one prototype per label.
  This is "coffee is dark, hot, and bitter."

The :class:`MemorySystem` wires the two together. New experiences land in
episodic memory. Periodically (or on demand) :meth:`consolidate_to_semantic`
runs the consolidation pass over episodic memory and pushes each cluster
centroid into semantic memory as a prototype — forming new semantic concepts
or reinforcing existing ones. Recall queries both stores; the caller can
pick whichever source is needed.

This design mirrors **memory consolidation during sleep** in mammals: the
hippocampus (episodic) replays patterns, the neocortex (semantic) integrates
them into stable representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku._query import RetrievalResult, query as episodic_query
from kohaku.consolidation import consolidate
from kohaku.decay import DecayConfig, query_with_decay
from kohaku.learning import ItemMemory


@dataclass(frozen=True)
class CombinedRecall:
    """One row of a combined episodic + semantic recall."""
    source: str          # "episodic" or "semantic"
    label: str
    similarity: float
    entry_id: int
    value: HyperVector


class MemorySystem:
    """Combined episodic + semantic memory store.

    Parameters
    ----------
    episodic_capacity:
        Capacity of the underlying :class:`EpisodicMemory`. Default 1000.
    dims:
        Hypervector dimensionality (default 10 000).
    decay_config:
        Optional default :class:`DecayConfig` used when querying the episodic
        store with decay enabled. ``None`` means no decay by default.
    """

    def __init__(
        self,
        episodic_capacity: int = 1000,
        dims: int = DIMS,
        decay_config: Optional[DecayConfig] = None,
    ) -> None:
        self.episodic: EpisodicMemory = EpisodicMemory(capacity=episodic_capacity)
        self.semantic: ItemMemory = ItemMemory(dims=dims)
        self._dims = int(dims)
        self._decay_config = decay_config

    # ── basic ──────────────────────────────────────────────────────────────
    @property
    def dims(self) -> int:
        return self._dims

    @property
    def num_episodes(self) -> int:
        return len(self.episodic)

    @property
    def num_concepts(self) -> int:
        return len(self.semantic)

    # ── storage ────────────────────────────────────────────────────────────
    def store_episode(
        self,
        key: HyperVector,
        value: HyperVector,
        label: str,
    ) -> int:
        """Store a raw experience in episodic memory. Returns its entry id."""
        return self.episodic.store(key, value, label)

    def reinforce_concept(
        self,
        label: str,
        vector: HyperVector,
        weight: float = 1.0,
    ) -> None:
        """Push a positive example into semantic memory directly (skipping
        episodic). Equivalent to teaching a fact by name."""
        self.semantic.add(label, vector, weight=weight)

    def teach(
        self,
        label: str,
        vector: HyperVector,
        correct: bool,
        weight: float = 1.0,
    ) -> None:
        """Supervised feedback: reinforce (correct) or suppress (wrong)."""
        self.semantic.train_from_feedback(label, vector, correct, weight=weight)

    # ── consolidation ──────────────────────────────────────────────────────
    def consolidate_to_semantic(
        self,
        similarity_threshold: float = 0.3,
    ) -> int:
        """Promote episodic clusters into semantic prototypes.

        Runs :func:`kohaku.consolidate` over episodic memory, then pushes each
        cluster's centroid into semantic memory under the cluster's seed label
        with weight = cluster size (so larger clusters move the prototype
        more). Returns the number of clusters promoted.

        This is the core "sleep consolidation" operation. Repeated calls
        continue refining existing semantic prototypes.
        """
        clusters = consolidate(self.episodic, similarity_threshold=similarity_threshold)
        for c in clusters:
            self.semantic.add(c.label, c.centroid_key, weight=float(c.size))
        return len(clusters)

    # ── recall ─────────────────────────────────────────────────────────────
    def recall(
        self,
        query_key: HyperVector,
        top_k: int = 3,
        use_decay: bool = False,
        decay_config: Optional[DecayConfig] = None,
    ) -> List[CombinedRecall]:
        """Query both stores and return a merged ranked list.

        For each store the top-``top_k`` matches are scored:
            * episodic: cosine similarity (or decayed similarity if
              ``use_decay=True``).
            * semantic: cosine similarity to each prototype.

        Results are tagged with ``source`` ("episodic" / "semantic") and
        merged by descending similarity, capped at ``top_k`` overall.
        """
        results: List[CombinedRecall] = []

        # Episodic side
        if not self.episodic.is_empty:
            cfg = decay_config or self._decay_config
            ep_results: List[RetrievalResult]
            if use_decay and cfg is not None:
                ep_results = query_with_decay(
                    self.episodic, query_key, top_k=top_k, config=cfg
                )
            else:
                ep_results = episodic_query(self.episodic, query_key, top_k=top_k)
            for r in ep_results:
                results.append(
                    CombinedRecall(
                        source="episodic",
                        label=r.label,
                        similarity=r.similarity,
                        entry_id=r.entry_id,
                        value=r.value,
                    )
                )

        # Semantic side
        if len(self.semantic) > 0:
            sem_results = self.semantic.predict(query_key, top_k=top_k)
            for r in sem_results:
                results.append(
                    CombinedRecall(
                        source="semantic",
                        label=r.label,
                        similarity=r.similarity,
                        entry_id=r.entry_id,
                        value=r.value,
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return self.num_episodes + self.num_concepts

    def __repr__(self) -> str:
        return (
            f"MemorySystem(episodes={self.num_episodes}/{self.episodic._capacity}, "
            f"concepts={self.num_concepts}, dims={self._dims})"
        )
