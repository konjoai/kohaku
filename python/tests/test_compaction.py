"""Tests for kohaku.compaction — find_duplicates, deduplicate, compact."""

from __future__ import annotations

import numpy as np
import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.compaction import compact, deduplicate, find_duplicates


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    data = base.data.copy()
    data[rng.choice(DIMS, size=int(flip_frac * DIMS), replace=False)] *= -1
    return HyperVector(data)


# ---------------------------------------------------------------------------
# cosine similarity helper (tested indirectly through find_duplicates)
# ---------------------------------------------------------------------------


def test_identical_keys_have_similarity_one() -> None:
    """An entry compared with itself has cosine similarity 1.0."""
    mem = EpisodicMemory(capacity=5)
    k = _hv(seed=1)
    mem.store(k, _hv(seed=2), label="a")
    mem.store(k, _hv(seed=3), label="b")  # identical key
    groups = find_duplicates(mem, similarity_threshold=0.99)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_orthogonal_keys_not_duplicates() -> None:
    """Random 10k-D vectors are nearly orthogonal — should not be flagged."""
    mem = EpisodicMemory(capacity=10)
    for i in range(6):
        mem.store(_hv(seed=i), _hv(seed=i + 100), label=f"e{i}")
    groups = find_duplicates(mem, similarity_threshold=0.95)
    assert groups == []


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------


def test_find_duplicates_returns_groups_by_id() -> None:
    """All near-identical entries land in one group; IDs match stored entries."""
    mem = EpisodicMemory(capacity=10)
    base = _hv(seed=42)
    ids = []
    for i in range(4):
        ids.append(mem.store(_noisy(base, 0.01, seed=i), _hv(seed=200 + i), label="x"))
    # Add one unrelated entry
    mem.store(_hv(seed=999), _hv(seed=998), label="other")

    groups = find_duplicates(mem, similarity_threshold=0.95)
    assert len(groups) == 1
    # All four near-duplicate IDs should appear in the group
    assert set(ids).issubset(groups[0])


def test_find_duplicates_empty_memory() -> None:
    mem = EpisodicMemory(capacity=5)
    assert find_duplicates(mem) == []


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_removes_near_duplicates() -> None:
    mem = EpisodicMemory(capacity=10)
    base = _hv(seed=7)
    for i in range(4):
        mem.store(_noisy(base, 0.01, seed=i), _hv(seed=100 + i), label="dup")
    mem.store(_hv(seed=999), _hv(seed=998), label="unique")

    removed = deduplicate(mem, similarity_threshold=0.95)
    assert removed >= 2  # at least 2 of the 4 near-duplicates pruned
    assert len(mem) < 5  # memory is smaller


def test_deduplicate_keeps_oldest_entry() -> None:
    """deduplicate must keep the entry with the smallest ID."""
    mem = EpisodicMemory(capacity=10)
    base = _hv(seed=5)
    first_id = mem.store(base, _hv(seed=10), label="first")
    for i in range(3):
        mem.store(_noisy(base, 0.01, seed=i), _hv(seed=20 + i), label=f"dup{i}")

    deduplicate(mem, similarity_threshold=0.95)
    surviving_ids = [e.id for e in mem.entries()]
    assert first_id in surviving_ids, "oldest entry must survive deduplication"


def test_deduplicate_noop_when_no_duplicates() -> None:
    mem = EpisodicMemory(capacity=10)
    for i in range(5):
        mem.store(_hv(seed=i * 100), _hv(seed=i * 100 + 1), label=f"e{i}")
    removed = deduplicate(mem, similarity_threshold=0.95)
    assert removed == 0
    assert len(mem) == 5


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def test_compact_bad_utilization_raises() -> None:
    mem = EpisodicMemory(capacity=10)
    with pytest.raises(ValueError):
        compact(mem, target_utilization=0.0)
    with pytest.raises(ValueError):
        compact(mem, target_utilization=1.5)


def test_compact_reduces_to_target() -> None:
    """compact() must leave len(mem) <= capacity * target_utilization."""
    capacity = 20
    mem = EpisodicMemory(capacity=capacity)
    for i in range(capacity):  # fill to 100%
        mem.store(_hv(seed=i), _hv(seed=i + 1000), label=f"e{i}")

    compact(mem, target_utilization=0.5)
    assert len(mem) <= int(capacity * 0.5)


def test_compact_deduplicates_before_eviction() -> None:
    """compact() should deduplicate first, then evict — dedup count is included."""
    capacity = 10
    mem = EpisodicMemory(capacity=capacity)
    base = _hv(seed=3)
    # 6 near-identical entries
    for i in range(6):
        mem.store(_noisy(base, 0.01, seed=i), _hv(seed=100 + i), label="dup")
    # 4 unique entries
    for i in range(4):
        mem.store(_hv(seed=200 + i * 100), _hv(seed=300 + i * 100), label=f"u{i}")

    removed = compact(mem, target_utilization=0.7)
    assert removed > 0
    assert len(mem) <= int(capacity * 0.7)
