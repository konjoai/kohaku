"""Memory compaction: remove near-duplicate entries before eviction."""
from __future__ import annotations
import logging
from typing import List, Set
from ._index import index_over
from ._pure import EpisodicMemory

logger = logging.getLogger(__name__)


def find_duplicates(
    memory: EpisodicMemory,
    similarity_threshold: float = 0.95,
) -> List[Set[int]]:
    """Return groups of near-duplicate entry IDs (by key similarity).

    Each set contains IDs that are mutually similar above the threshold.
    The lowest-ID entry in each set is the canonical one to keep.
    """
    entries = memory.entries()  # returns list[MemoryEntry]
    n = len(entries)
    groups: List[Set[int]] = []
    visited: Set[int] = set()
    if n < 2:
        return groups

    # Batched cosine: one resident index, one scored pass per pivot row, instead
    # of an O(n²) Python cosine double loop.
    idx = index_over(entries)
    for i in range(n):
        if entries[i].id in visited:
            continue
        sims = idx.all_scores(entries[i].key.data)
        group: Set[int] = {entries[i].id}
        for j in range(i + 1, n):
            if entries[j].id in visited:
                continue
            if sims[j] >= similarity_threshold:
                group.add(entries[j].id)
                visited.add(entries[j].id)
        if len(group) > 1:
            visited.add(entries[i].id)
            groups.append(group)

    return groups


def deduplicate(
    memory: EpisodicMemory,
    similarity_threshold: float = 0.95,
) -> int:
    """Remove near-duplicate entries from memory in-place.

    Keeps the oldest entry (lowest ID) in each duplicate group.
    Returns the number of entries removed.
    """
    groups = find_duplicates(memory, similarity_threshold)
    # Collect IDs to remove (all but the smallest ID in each group)
    ids_to_remove: Set[int] = set()
    for group in groups:
        sorted_ids = sorted(group)
        for entry_id in sorted_ids[1:]:
            ids_to_remove.add(entry_id)

    if not ids_to_remove:
        return 0

    # Rebuild _entries list in-place, keeping only non-removed entries
    memory._entries[:] = [e for e in memory._entries if e.id not in ids_to_remove]
    memory._mark_mutated()  # invalidate the retrieval-index cache
    removed = len(ids_to_remove)
    logger.info("deduplicate: removed %d near-duplicate entries", removed)
    return removed


def compact(memory: EpisodicMemory, target_utilization: float = 0.7) -> int:
    """Compact memory: deduplicate, then evict oldest entries to reach target utilization.

    Returns the total number of entries removed.
    """
    if not (0.0 < target_utilization <= 1.0):
        raise ValueError("target_utilization must be in (0, 1]")

    removed = deduplicate(memory)
    target = int(memory._capacity * target_utilization)
    # Evict oldest entries (front of the list = oldest, since FIFO)
    evicted = 0
    while len(memory._entries) > target:
        memory._entries.pop(0)
        evicted += 1
    if evicted:
        memory._mark_mutated()  # invalidate the retrieval-index cache
    return removed + evicted
