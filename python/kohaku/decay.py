"""Forgetting curves — exponential temporal decay on similarity scores.

Models the Ebbinghaus forgetting curve:

    weight(age) = exp(-ln(2) * age / half_life)
                = 0.5 ** (age / half_life)

`age` is measured in *timestamp ticks*: each store() advances the memory's internal clock
by 1, so age = (current_clock - entry.timestamp). Decayed similarity:

    decayed_sim = raw_sim * weight(age)

This preserves the sign of similarity (a strongly negative match stays strongly negative
under low decay) and pushes old memories asymptotically toward zero.

Optional `floor` clamps the decay weight to a minimum, modeling residual long-term recall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from kohaku._index import index_for
from kohaku._pure import EpisodicMemory, HyperVector
from kohaku._query import RetrievalResult


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for exponential temporal decay.

    Attributes
    ----------
    half_life:
        Number of timestamp ticks at which the decay weight reaches 0.5.
        Must be > 0.
    floor:
        Minimum decay weight. Old memories never decay below this. Default 0.0.
    """

    half_life: float = 100.0
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise ValueError("half_life must be > 0")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")


def decay_weight(age: int, config: DecayConfig) -> float:
    """Return the decay weight for an entry of given *age* (in ticks)."""
    if age < 0:
        raise ValueError("age must be >= 0")
    w = math.pow(0.5, age / config.half_life)
    return max(w, config.floor)


def query_with_decay(
    memory: EpisodicMemory,
    query_key: HyperVector,
    top_k: int,
    config: DecayConfig | None = None,
) -> List[RetrievalResult]:
    """Top-k retrieval with exponential temporal decay applied to similarity.

    Memory's internal clock at query time defines "now"; older entries (smaller timestamp)
    receive more decay. Returns results sorted by *decayed* similarity descending.
    """
    cfg = config or DecayConfig()
    if top_k <= 0 or memory.is_empty:
        return []

    # `_timestamp` advances on each store and is the "next" tick — the most recent
    # entry has timestamp == _timestamp - 1, so age = (_timestamp - 1) - entry.timestamp.
    now = memory._timestamp - 1
    entries = memory.entries()
    sims = index_for(memory, entries).all_scores(query_key.data)
    results: List[RetrievalResult] = []
    for raw, e in zip(sims, entries):
        age = now - e.timestamp
        if age < 0:
            age = 0
        w = decay_weight(age, cfg)
        results.append(
            RetrievalResult(
                entry_id=e.id,
                label=e.label,
                similarity=float(raw) * w,
                value=e.value,
            )
        )
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[:top_k]
