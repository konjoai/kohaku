"""Memory compaction: remove near-duplicate entries before eviction."""
from __future__ import annotations
import logging
from typing import List, Set
from ._pure import EpisodicMemory, MemoryEntry

logger = logging.getLogger(__name__)


def cosine_similarity(a: "MemoryEntry", b: "MemoryEntry") -> float:
    """Cosine similarity between the keys of two MemoryEntry objects."""
    return a.key.cosine_similarity(b.key)


def _entry_cosine(a: MemoryEntry, b: MemoryEntry) -> float:
    """Cosine similarity between two entry keys."""
    return a.key.cosine_similarity(b.key)


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

    for i in range(n):
        if entries[i].id in visited:
            continue
        group: Set[int] = {entries[i].id}
        for j in range(i + 1, n):
            if entries[j].id in visited:
                continue
            sim = _entry_cosine(entries[i], entries[j])
            if sim >= similarity_threshold:
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
    while len(memory._entries) > target:
        memory._entries.pop(0)
        removed += 1
    return removed
