"""Tests for memory compaction and deduplication."""
import pytest
from kohaku.compaction import cosine_similarity, find_duplicates, deduplicate, compact
from kohaku._pure import EpisodicMemory, HyperVector


DIM = 64


def make_memory(capacity=20):
    return EpisodicMemory(capacity=capacity)


def rand_hv(seed=0):
    return HyperVector.random(dims=DIM, seed=seed)


def test_cosine_similarity_identical():
    """cosine_similarity on two entries with the same key returns ~1.0."""
    mem = make_memory()
    v = rand_hv(0)
    mem.store(v, rand_hv(1), label="a")
    mem.store(v, rand_hv(2), label="b")
    entries = mem.entries()
    assert abs(cosine_similarity(entries[0], entries[0]) - 1.0) < 1e-6


def test_cosine_similarity_range():
    mem = make_memory()
    mem.store(rand_hv(0), rand_hv(10), label="a")
    mem.store(rand_hv(1), rand_hv(11), label="b")
    entries = mem.entries()
    sim = cosine_similarity(entries[0], entries[1])
    assert -1.0 <= sim <= 1.0


def test_find_duplicates_no_dups():
    mem = make_memory()
    for i in range(5):
        mem.store(rand_hv(i), rand_hv(i + 100), label=f"e{i}")
    groups = find_duplicates(mem, similarity_threshold=0.99)
    assert len(groups) == 0


def test_find_duplicates_identical():
    mem = make_memory()
    v = rand_hv(0)
    mem.store(v, rand_hv(10), label="a")
    mem.store(v, rand_hv(11), label="b")
    groups = find_duplicates(mem, similarity_threshold=0.99)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_deduplicate_removes_duplicate():
    mem = make_memory()
    v = rand_hv(0)
    mem.store(v, rand_hv(10), label="a")
    mem.store(v, rand_hv(11), label="b")
    initial = len(mem)
    removed = deduplicate(mem, similarity_threshold=0.99)
    assert removed == 1
    assert len(mem) == initial - 1


def test_deduplicate_keeps_oldest():
    mem = make_memory()
    v = rand_hv(0)
    mem.store(v, rand_hv(10), label="first")
    mem.store(v, rand_hv(11), label="second")
    deduplicate(mem, similarity_threshold=0.99)
    remaining = mem.entries()
    assert len(remaining) == 1
    assert remaining[0].label == "first"


def test_deduplicate_no_duplicates_noop():
    mem = make_memory()
    for i in range(5):
        mem.store(rand_hv(i), rand_hv(i + 100), label=f"e{i}")
    before = len(mem)
    removed = deduplicate(mem, similarity_threshold=0.99)
    assert removed == 0
    assert len(mem) == before


def test_compact_bad_utilization():
    mem = make_memory()
    with pytest.raises(ValueError):
        compact(mem, target_utilization=0.0)
    with pytest.raises(ValueError):
        compact(mem, target_utilization=1.5)


def test_compact_reduces_to_target():
    mem = make_memory(capacity=10)
    for i in range(10):
        mem.store(rand_hv(i), rand_hv(i + 100), label=f"e{i}")
    compact(mem, target_utilization=0.5)
    assert len(mem) <= 5


def test_compact_deduplicates_first():
    mem = make_memory(capacity=10)
    v = rand_hv(0)
    mem.store(v, rand_hv(10), label="dup-a")
    mem.store(v, rand_hv(11), label="dup-b")
    for i in range(3, 8):
        mem.store(rand_hv(i), rand_hv(i + 100), label=f"e{i}")
    removed = compact(mem, target_utilization=0.8)
    assert removed > 0
