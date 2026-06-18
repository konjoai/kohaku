"""Multi-hop associative chaining over an EpisodicMemory.

:func:`chain_query` iteratively retrieves the highest-similarity unvisited
entry for the current query key, then follows that entry's stored key HV to
the next hop — building a relational chain across the memory graph.

Example::

    result = chain_query(memory, start_key=question_hv, hops=3)
    for hop in result.hops:
        print(hop.hop, hop.label, hop.similarity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from kohaku._pure import EpisodicMemory, HyperVector
from kohaku._query import query


@dataclass(frozen=True)
class HopResult:
    hop: int
    entry_id: int
    label: str
    similarity: float


@dataclass
class ChainResult:
    hops: List[HopResult]
    terminated_early: bool

    def labels(self) -> List[str]:
        return [h.label for h in self.hops]

    def similarities(self) -> List[float]:
        return [h.similarity for h in self.hops]


def chain_query(
    memory: EpisodicMemory,
    start_key: HyperVector,
    hops: int = 3,
    min_similarity: float = 0.0,
) -> ChainResult:
    """Walk the memory graph by iteratively following the nearest neighbour.

    Each hop retrieves the highest-similarity unvisited entry, then uses that
    entry's stored key HV as the next query.

    Args:
        memory: The EpisodicMemory to traverse.
        start_key: Initial query HV.
        hops: Maximum number of hops (each hop produces one HopResult).
        min_similarity: Stop early if the best unvisited match has similarity
            below this value.

    Returns:
        :class:`ChainResult`.  ``terminated_early`` is ``True`` when the
        chain ended before ``hops`` steps (empty memory, low similarity, or
        no unvisited candidates remain).
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if memory.is_empty:
        return ChainResult(hops=[], terminated_early=True)

    visited_ids: Set[int] = set()
    chain: List[HopResult] = []
    current_key = start_key

    for hop_idx in range(hops):
        candidates = query(memory, current_key, top_k=len(visited_ids) + 1)
        best = next((r for r in candidates if r.entry_id not in visited_ids), None)
        if best is None or best.similarity < min_similarity:
            return ChainResult(hops=chain, terminated_early=True)

        chain.append(
            HopResult(
                hop=hop_idx,
                entry_id=best.entry_id,
                label=best.label,
                similarity=best.similarity,
            )
        )
        visited_ids.add(best.entry_id)

        matched = next((e for e in memory.entries() if e.id == best.entry_id), None)
        if matched is None:
            return ChainResult(hops=chain, terminated_early=True)
        current_key = matched.key

    return ChainResult(hops=chain, terminated_early=False)
