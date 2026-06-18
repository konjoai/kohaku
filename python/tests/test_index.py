"""Tests for the resident retrieval index (C1 slice 2).

The index must rank identically to the NumPy baseline on either backend, and
its per-memory cache must rebuild exactly when the memory's contents change.
"""

from __future__ import annotations

import numpy as np
import pytest

from kohaku._accel import HAS_RUST, _numpy_cosine_topk
from kohaku._index import RetrievalIndex, _INDEX_CACHE, index_for, index_over
from kohaku._pure import EpisodicMemory, HyperVector


def _bipolar(n, dims, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((n, dims)) > 0.5, np.int8(1), np.int8(-1))


def test_index_matches_numpy_ranking():
    keys = _bipolar(40, 256, 7)
    idx = RetrievalIndex(keys)
    for qi in (0, 19, 39):
        q = keys[qi]
        got = idx.topk(q, 10)
        want = _numpy_cosine_topk(q, keys, 10)
        assert [i for i, _ in got] == [i for i, _ in want]
        for (_, gs), (_, ws) in zip(got, want):
            assert gs == pytest.approx(ws, abs=1e-5)


def test_index_self_match_is_one():
    keys = _bipolar(6, 128, 2)
    idx = RetrievalIndex(keys)
    out = idx.topk(keys[3], 6)
    assert out[0][0] == 3
    assert out[0][1] == pytest.approx(1.0, abs=1e-6)


def test_index_empty_and_zero_k():
    assert RetrievalIndex(np.empty((0, 0), dtype=np.int8)).topk(np.ones(4), 5) == []
    idx = RetrievalIndex(_bipolar(3, 32, 1))
    assert idx.topk(np.ones(32, dtype=np.int8), 0) == []
    assert len(RetrievalIndex(np.empty((0, 0), dtype=np.int8))) == 0


def test_all_scores_row_order_and_agreement():
    keys = _bipolar(12, 64, 5)
    idx = RetrievalIndex(keys)
    q = keys[4]
    sims = idx.all_scores(q)
    assert sims.shape == (12,)
    assert sims[4] == pytest.approx(1.0, abs=1e-6)
    # Every score must match the top-k pass for the same row.
    for i, s in idx.topk(q, 12):
        assert sims[i] == pytest.approx(s, abs=1e-6)


def test_all_pairs_matches_per_row_all_scores():
    """The batched all-pairs matrix must equal stacking all_scores row by row —
    the parity guarantee the conflict/duplicate scans depend on."""
    keys = _bipolar(15, 128, 13)
    idx = RetrievalIndex(keys)
    mat = idx.all_pairs()
    assert mat.shape == (15, 15)
    for i in range(15):
        row = idx.all_scores(keys[i])
        np.testing.assert_allclose(mat[i], row, atol=1e-5)


def test_all_pairs_symmetric_with_unit_diagonal():
    keys = _bipolar(10, 96, 21)
    mat = RetrievalIndex(keys).all_pairs()
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(mat), np.ones(10), atol=1e-6)


def test_all_pairs_empty():
    assert RetrievalIndex(np.empty((0, 0), dtype=np.int8)).all_pairs().shape == (0, 0)


def test_all_pairs_matches_numpy_dot_baseline():
    """Independent check: for bipolar rows, cosine = MMᵀ / dims."""
    keys = _bipolar(8, 64, 33)
    mat = RetrievalIndex(keys).all_pairs()
    f = keys.astype(np.float32)
    expected = (f @ f.T) / 64.0
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(mat, expected, atol=1e-5)


def _store_random(memory, seed):
    hv = HyperVector(_bipolar(1, 64, seed)[0])
    return memory.store(hv, hv, f"e{seed}")


def test_cache_reuses_until_memory_changes():
    mem = EpisodicMemory(capacity=100)
    for s in range(5):
        _store_random(mem, s)
    entries = mem.entries()
    first = index_for(mem, entries)
    # Same contents → same cached object.
    assert index_for(mem, mem.entries()) is first
    # A store changes the fingerprint → fresh index.
    _store_random(mem, 99)
    second = index_for(mem, mem.entries())
    assert second is not first
    assert len(second) == 6


def test_cache_invalidates_on_clear():
    mem = EpisodicMemory(capacity=100)
    for s in range(3):
        _store_random(mem, s)
    before = index_for(mem, mem.entries())
    mem.clear()
    for s in range(3):
        _store_random(mem, s + 10)
    after = index_for(mem, mem.entries())
    assert after is not before


def test_cache_evicted_with_memory():
    mem = EpisodicMemory(capacity=10)
    _store_random(mem, 1)
    index_for(mem, mem.entries())
    assert mem in _INDEX_CACHE
    del mem
    import gc

    gc.collect()
    # WeakKeyDictionary must not keep the memory (or its index) alive.
    assert all(k is not None for k in _INDEX_CACHE.keys())


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")
def test_packed_index_len_via_class():
    keys = _bipolar(7, 96, 3)
    from kohaku import _kohaku_rs

    packed = _kohaku_rs.PackedIndex(np.ascontiguousarray(keys, dtype=np.int8))
    assert len(packed) == 7


# ── slice 3: batch-scan helper + direct-mutation cache safety ────────────────


def test_index_over_matches_numpy_ranking():
    mem = EpisodicMemory(capacity=100)
    keys = _bipolar(25, 128, 11)
    for i in range(25):
        hv = HyperVector(keys[i])
        mem.store(hv, hv, f"e{i}")
    idx = index_over(mem.entries())
    for qi in (0, 12, 24):
        got = idx.topk(keys[qi], 8)
        want = _numpy_cosine_topk(keys[qi], keys, 8)
        assert [i for i, _ in got] == [i for i, _ in want]


def test_index_over_empty():
    assert len(index_over([])) == 0
    assert index_over([]).topk(np.ones(4, dtype=np.int8), 3) == []


def test_mark_mutated_invalidates_cache():
    mem = EpisodicMemory(capacity=100)
    for s in range(6):
        hv = HyperVector(_bipolar(1, 64, s)[0])
        mem.store(hv, hv, f"e{s}")
    first = index_for(mem, mem.entries())
    # Simulate a direct _entries delete (as compaction/conflicts/bulk do).
    mem._entries[:] = mem._entries[:3]
    mem._mark_mutated()
    second = index_for(mem, mem.entries())
    assert second is not first
    assert len(second) == 3


def test_query_reflects_deletion_after_compaction():
    """Regression: a delete via compaction must not leave query() on a stale
    cached index (would point at removed rows / raise IndexError)."""
    from kohaku import query as core_query
    from kohaku.compaction import deduplicate

    mem = EpisodicMemory(capacity=100)
    dup = HyperVector(_bipolar(1, 128, 1)[0])
    mem.store(dup, dup, "dup-a")
    mem.store(dup, dup, "dup-b")  # identical key → cosine 1.0
    for s in range(2, 6):
        hv = HyperVector(_bipolar(1, 128, s)[0])
        mem.store(hv, hv, f"e{s}")

    probe = HyperVector(_bipolar(1, 128, 2)[0])
    core_query(mem, probe, 5)  # populate the per-memory index cache
    removed = deduplicate(mem, similarity_threshold=0.99)
    assert removed == 1  # one of the identical pair dropped

    res = core_query(mem, probe, 10)  # must use a rebuilt index
    live = {e.id for e in mem.entries()}
    assert {r.entry_id for r in res} <= live
    assert len(res) == len(live)
