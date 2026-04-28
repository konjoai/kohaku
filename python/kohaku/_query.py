from __future__ import annotations
from dataclasses import dataclass
from kohaku._pure import HyperVector, EpisodicMemory


@dataclass(frozen=True)
class RetrievalResult:
    entry_id: int
    label: str
    similarity: float
    value: HyperVector


def query(memory: EpisodicMemory, query_key: HyperVector, top_k: int) -> list[RetrievalResult]:
    """Return top_k entries by cosine similarity, descending."""
    if top_k <= 0 or memory.is_empty:
        return []
    results = [
        RetrievalResult(
            entry_id=e.id,
            label=e.label,
            similarity=e.key.cosine_similarity(query_key),
            value=e.value,
        )
        for e in memory.entries()
    ]
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[:top_k]


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
