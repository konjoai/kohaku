from __future__ import annotations
from dataclasses import dataclass

from kohaku._index import index_for
from kohaku._pure import HyperVector, EpisodicMemory


@dataclass(frozen=True)
class RetrievalResult:
    entry_id: int
    label: str
    similarity: float
    value: HyperVector


def query(memory: EpisodicMemory, query_key: HyperVector, top_k: int) -> list[RetrievalResult]:
    """Return top_k entries by cosine similarity, descending.

    Similarities are computed in one batched pass through a resident retrieval
    index (Rust bit-packed popcount when the extension is present, NumPy matmul
    otherwise). The index is cached per memory and reused until the memory
    changes, so repeated probes skip re-packing the keys.
    """
    if top_k <= 0 or memory.is_empty:
        return []
    entries = memory.entries()
    ranked = index_for(memory, entries).topk(query_key.data, top_k)
    return [
        RetrievalResult(
            entry_id=entries[i].id,
            label=entries[i].label,
            similarity=sim,
            value=entries[i].value,
        )
        for i, sim in ranked
    ]


def query_threshold(
    memory: EpisodicMemory, query_key: HyperVector, threshold: float
) -> list[RetrievalResult]:
    """Return all entries with similarity >= threshold, sorted descending."""
    if memory.is_empty:
        return []
    results = []
    for e in memory.entries():
        sim = e.key.cosine_similarity(query_key)
        if sim >= threshold:
            results.append(
                RetrievalResult(
                    entry_id=e.id,
                    label=e.label,
                    similarity=sim,
                    value=e.value,
                )
            )
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results
