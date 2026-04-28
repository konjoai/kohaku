"""Tests for the pure-Python HDC implementation."""
from __future__ import annotations

import pytest
import numpy as np
from kohaku._pure import HyperVector, EpisodicMemory, DIMS
from kohaku._query import query, query_threshold


# ─── HyperVector tests ────────────────────────────────────────────────────────

def test_random_vector_shape():
    """Generated vector must have exactly DIMS components."""
    hv = HyperVector.random(DIMS, seed=42)
    assert len(hv) == DIMS


def test_random_vector_bipolar():
    """Every component must be exactly +1 or -1."""
    hv = HyperVector.random(DIMS, seed=7)
    unique = set(hv.data.tolist())
    assert unique == {1, -1}, f"Expected only {{1, -1}}, got {unique}"


def test_random_deterministic():
    """Same seed must produce byte-identical vectors."""
    a = HyperVector.random(DIMS, seed=99)
    b = HyperVector.random(DIMS, seed=99)
    assert np.array_equal(a.data, b.data), "Identical seeds must yield identical vectors"


def test_different_seeds_orthogonal():
    """Two random vectors from different seeds should be near-orthogonal."""
    a = HyperVector.random(DIMS, seed=1)
    b = HyperVector.random(DIMS, seed=2)
    sim = abs(a.cosine_similarity(b))
    assert sim < 0.05, f"Different-seed vectors should be near-orthogonal, got |sim|={sim:.4f}"


def test_identical_similarity_is_one():
    """A vector compared to itself must have cosine similarity 1.0."""
    hv = HyperVector.random(DIMS, seed=123)
    sim = hv.cosine_similarity(hv)
    assert abs(sim - 1.0) < 1e-5, f"Self-similarity must be 1.0, got {sim}"


def test_bundle_similar_to_inputs():
    """Bundled vector must have positive cosine similarity (> 0.4) to each constituent."""
    a = HyperVector.random(DIMS, seed=10)
    b = HyperVector.random(DIMS, seed=20)
    c = HyperVector.random(DIMS, seed=30)
    bundled = HyperVector.bundle_all([a, b, c])
    sim_a = bundled.cosine_similarity(a)
    sim_b = bundled.cosine_similarity(b)
    sim_c = bundled.cosine_similarity(c)
    assert sim_a > 0.4, f"Bundle should be similar to a, got {sim_a:.4f}"
    assert sim_b > 0.4, f"Bundle should be similar to b, got {sim_b:.4f}"
    assert sim_c > 0.4, f"Bundle should be similar to c, got {sim_c:.4f}"


def test_bind_recovers_with_self():
    """(a bind b) bind a must recover b with similarity > 0.9."""
    a = HyperVector.random(DIMS, seed=11)
    b = HyperVector.random(DIMS, seed=22)
    bound = a.bind(b)
    recovered = bound.bind(a)
    sim = recovered.cosine_similarity(b)
    assert sim > 0.9, f"(a⊗b)⊗a should recover b (sim > 0.9), got {sim:.4f}"


def test_permute_differs_from_original():
    """A permuted vector should be near-orthogonal to the original."""
    hv = HyperVector.random(DIMS, seed=5)
    permuted = hv.permute(1)
    sim = abs(hv.cosine_similarity(permuted))
    assert sim < 0.1, f"Permuted vector should differ from original, got |sim|={sim:.4f}"


def test_permute_invertible():
    """permute(d) followed by permute(DIMS - d) must recover the original."""
    hv = HyperVector.random(DIMS, seed=5)
    shift = 137
    permuted = hv.permute(shift)
    recovered = permuted.permute(DIMS - shift)
    sim = hv.cosine_similarity(recovered)
    assert abs(sim - 1.0) < 1e-5, f"Double-permute must recover original, got sim={sim:.6f}"


def test_memory_store_and_len():
    """Storing entries must increase len by 1 each time."""
    mem = EpisodicMemory(capacity=10)
    assert len(mem) == 0
    k = HyperVector.random(DIMS, seed=1)
    v = HyperVector.random(DIMS, seed=2)
    mem.store(k, v, "first")
    assert len(mem) == 1
    k2 = HyperVector.random(DIMS, seed=3)
    v2 = HyperVector.random(DIMS, seed=4)
    mem.store(k2, v2, "second")
    assert len(mem) == 2


def test_memory_fifo_eviction():
    """When at capacity, the oldest entry must be evicted (FIFO)."""
    mem = EpisodicMemory(capacity=3)
    for i in range(3):
        k = HyperVector.random(DIMS, seed=i * 10)
        v = HyperVector.random(DIMS, seed=i * 10 + 1)
        mem.store(k, v, f"entry-{i}")
    assert len(mem) == 3
    # Store a 4th — should evict entry-0
    k4 = HyperVector.random(DIMS, seed=99)
    v4 = HyperVector.random(DIMS, seed=100)
    mem.store(k4, v4, "entry-3")
    assert len(mem) == 3
    labels = [e.label for e in mem.entries()]
    assert "entry-0" not in labels, "entry-0 should have been evicted"
    assert "entry-1" in labels
    assert "entry-2" in labels
    assert "entry-3" in labels


def test_query_top_k_count():
    """query must return exactly top_k results (or fewer if memory is smaller)."""
    mem = EpisodicMemory(capacity=20)
    for i in range(10):
        k = HyperVector.random(DIMS, seed=i * 7 + 1)
        v = HyperVector.random(DIMS, seed=i * 7 + 2)
        mem.store(k, v, f"item-{i}")
    qk = HyperVector.random(DIMS, seed=5)
    results = query(mem, qk, top_k=5)
    assert len(results) == 5


def test_query_sorted_descending():
    """Results from query must be sorted in descending similarity order."""
    mem = EpisodicMemory(capacity=20)
    for i in range(10):
        k = HyperVector.random(DIMS, seed=i * 7 + 1)
        v = HyperVector.random(DIMS, seed=i * 7 + 2)
        mem.store(k, v, f"item-{i}")
    qk = HyperVector.random(DIMS, seed=5)
    results = query(mem, qk, top_k=10)
    for j in range(len(results) - 1):
        assert results[j].similarity >= results[j + 1].similarity, (
            f"Results not sorted: {results[j].similarity} < {results[j + 1].similarity}"
        )


def test_query_threshold_filters():
    """query_threshold must return only entries with similarity >= threshold."""
    mem = EpisodicMemory(capacity=20)
    # Store a key, then query with that exact key — sim == 1.0 guaranteed for that entry
    k = HyperVector.random(DIMS, seed=42)
    v = HyperVector.random(DIMS, seed=43)
    mem.store(k, v, "target")
    for i in range(5):
        ki = HyperVector.random(DIMS, seed=i * 100 + 200)
        vi = HyperVector.random(DIMS, seed=i * 100 + 201)
        mem.store(ki, vi, f"noise-{i}")
    results = query_threshold(mem, k, threshold=0.9)
    assert len(results) >= 1, "Should find at least the exact-match entry"
    for r in results:
        assert r.similarity >= 0.9, f"Result below threshold: {r.similarity}"
    labels = [r.label for r in results]
    assert "target" in labels, "Exact-match entry must be in results"


def test_hamming_distance_identical():
    """Hamming distance between a vector and itself must be 0.0."""
    hv = HyperVector.random(DIMS, seed=77)
    assert hv.hamming_distance(hv) == 0.0
