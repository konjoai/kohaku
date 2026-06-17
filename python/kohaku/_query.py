from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from kohaku._accel import cosine_topk
from kohaku._pure import HyperVector, EpisodicMemory


@dataclass(frozen=True)
class RetrievalResult:
    entry_id: int
    label: str
    similarity: float
    value: HyperVector


def query(memory: EpisodicMemory, query_key: HyperVector, top_k: int) -> list[RetrievalResult]:
    """Return top_k entries by cosine similarity, descending.

    Similarities are computed in one batched pass (Rust bit-packed popcount when
    the extension is present, NumPy matmul otherwise) instead of a per-entry
    Python loop.
    """
    if top_k <= 0 or memory.is_empty:
        return []
    entries = memory.entries()
    keys = np.stack([e.key.data for e in entries])
    ranked = cosine_topk(query_key.data, keys, top_k)
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
