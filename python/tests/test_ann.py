"""Tests for the bipolar-LSH ANN index (B2) and its Memory integration."""

from __future__ import annotations

import numpy as np
import pytest

from kohaku import LSHIndex, Memory
from kohaku._pure import EpisodicMemory, HyperVector


def _flip(hv: HyperVector, n_bits: int, seed: int = 0) -> HyperVector:
    data = hv.data.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], size=n_bits, replace=False)
    data[idx] *= -1
    return HyperVector(data)


def test_add_len_contains_remove():
    idx = LSHIndex(dims=2048)
    idx.add(1, HyperVector.random(2048, seed=1))
    idx.add(2, HyperVector.random(2048, seed=2))
    assert len(idx) == 2
    assert 1 in idx and 2 in idx
    assert idx.remove(1) is True
    assert 1 not in idx
    assert idx.remove(999) is False
    assert len(idx) == 1


def test_invalid_params():
    with pytest.raises(ValueError):
        LSHIndex(dims=128, hash_bits=0)
    with pytest.raises(ValueError):
        LSHIndex(dims=128, hash_bits=64)
    with pytest.raises(ValueError):
        LSHIndex(dims=128, num_tables=0)


def test_dim_mismatch_raises():
    idx = LSHIndex(dims=512)
    with pytest.raises(ValueError):
        idx.add(1, HyperVector.random(256, seed=1))


def test_self_match_ranks_first():
    idx = LSHIndex(dims=4096)
    base = HyperVector.random(4096, seed=7)
    for s in range(2, 12):
        idx.add(s, HyperVector.random(4096, seed=s))
    idx.add(1, base)
    out = idx.query(base, top_k=3)
    assert out[0][0] == 1
    assert out[0][1] == pytest.approx(1.0, abs=1e-6)


def test_near_duplicate_recovered_as_candidate():
    dims = 8192
    idx = LSHIndex(dims=dims, num_tables=12, hash_bits=14)
    base = HyperVector.random(dims, seed=42)
    near = _flip(base, n_bits=dims // 20, seed=1)  # 5% flips → cosine ≈ 0.9
    idx.add(100, near)
    for s in range(200, 240):
        idx.add(s, HyperVector.random(dims, seed=s))
    cand = idx.candidates(base)
    assert 100 in cand  # the near-duplicate collides in some table
    top = idx.query(base, top_k=1)
    assert top[0][0] == 100


def test_query_empty_index_returns_empty():
    idx = LSHIndex(dims=512)
    assert idx.query(HyperVector.random(512, seed=1)) == []
    assert idx.query(HyperVector.random(512, seed=1), top_k=0) == []


def test_from_memory():
    mem = EpisodicMemory(capacity=10)
    for s in range(1, 6):
        hv = HyperVector.random(2048, seed=s)
        mem.store(hv, hv, f"m{s}")
    idx = LSHIndex.from_memory(mem)
    assert len(idx) == 5


def test_add_replaces_existing():
    idx = LSHIndex(dims=1024)
    idx.add(1, HyperVector.random(1024, seed=1))
    idx.add(1, HyperVector.random(1024, seed=2))  # same id, new vector
    assert len(idx) == 1


# ── Memory facade integration ───────────────────────────────────────────────

_PHRASES = [
    "the quick brown fox jumps over the lazy dog",
    "a fast auburn fox leaps above a sleepy hound",
    "python is a popular programming language",
    "rust is a systems programming language",
    "paris is the capital city of france",
    "berlin is the capital city of germany",
    "the stock market rose three percent today",
    "investors cheered as equities climbed higher",
    "photosynthesis converts sunlight into energy",
    "chlorophyll captures light for plant growth",
]


def test_ann_facade_matches_exact_for_similarity():
    exact = Memory(dims=4096)
    approx = Memory(dims=4096, ann=True)
    assert approx.ann_enabled and not exact.ann_enabled
    for p in _PHRASES:
        exact.store(p)
        approx.store(p)
    for q in _PHRASES:
        e = [h.text for h in exact.query(q, top_k=3, reinforce=False)]
        a = [h.text for h in approx.query(q, top_k=3, reinforce=False)]
        # Top-1 must agree; the ANN path scores its candidates exactly.
        assert a[0] == e[0]


def test_ann_facade_eviction_keeps_index_consistent():
    mem = Memory(capacity=3, dims=2048, ann=True)
    for p in _PHRASES[:5]:
        mem.store(p)
    assert len(mem) == 3
    # Index must not carry evicted ids — its size tracks the live store.
    assert len(mem._index) == 3
    hits = mem.query(_PHRASES[4], top_k=3, reinforce=False)
    assert hits and hits[0].text == _PHRASES[4]


def test_ann_facade_expire_removes_from_index():
    from datetime import datetime, timedelta, timezone

    mem = Memory(dims=2048, ann=True)
    now = datetime.now(timezone.utc)
    mem.store("ephemeral fact", valid_until=now + timedelta(hours=1))
    mem.store("permanent fact")
    mem.expire(now=now + timedelta(days=1))
    assert len(mem._index) == 1
    hits = mem.query("permanent fact", reinforce=False)
    assert hits[0].text == "permanent fact"


def test_ann_facade_clear():
    mem = Memory(dims=2048, ann=True)
    mem.store("a phrase here")
    mem.clear()
    assert len(mem) == 0 and len(mem._index) == 0
