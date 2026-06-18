"""Tests for kohaku.chaining — multi-hop associative chaining."""

from __future__ import annotations

import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.chaining import chain_query


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


# ── argument validation ───────────────────────────────────────────────────────


def test_hops_zero_raises():
    mem = EpisodicMemory()
    with pytest.raises(ValueError, match="hops"):
        chain_query(mem, _hv(1), hops=0)


def test_hops_negative_raises():
    mem = EpisodicMemory()
    with pytest.raises(ValueError, match="hops"):
        chain_query(mem, _hv(1), hops=-1)


# ── empty memory ──────────────────────────────────────────────────────────────


def test_empty_memory_returns_terminated_early():
    mem = EpisodicMemory()
    result = chain_query(mem, _hv(1), hops=3)
    assert result.terminated_early is True
    assert result.hops == []


# ── single entry ──────────────────────────────────────────────────────────────


def test_single_entry_hop1():
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "sole")
    result = chain_query(mem, hv, hops=1)
    assert len(result.hops) == 1
    assert result.hops[0].label == "sole"
    assert result.terminated_early is False


def test_single_entry_hop2_terminates():
    """After visiting the only entry, hop 2 has no candidates."""
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "sole")
    result = chain_query(mem, hv, hops=2)
    assert len(result.hops) == 1
    assert result.terminated_early is True


# ── chain traversal ───────────────────────────────────────────────────────────


def test_chain_visits_distinct_entries():
    """Three entries: chain should visit distinct IDs."""
    mem = EpisodicMemory()
    hv_a = _hv(1)
    hv_b = _hv(2)
    hv_c = _hv(3)
    mem.store(hv_a, hv_a, "A")
    mem.store(hv_b, hv_b, "B")
    mem.store(hv_c, hv_c, "C")

    result = chain_query(mem, hv_a, hops=3)
    visited = [h.entry_id for h in result.hops]
    assert len(visited) == len(set(visited)), "Chain revisited an entry"


def test_hop_indices_are_sequential():
    mem = EpisodicMemory()
    for i in range(3):
        hv = _hv(i + 1)
        mem.store(hv, hv, f"e{i}")
    result = chain_query(mem, _hv(1), hops=3)
    for idx, hop in enumerate(result.hops):
        assert hop.hop == idx


def test_hops_limit_respected():
    mem = EpisodicMemory()
    for i in range(10):
        hv = _hv(i + 1)
        mem.store(hv, hv, f"e{i}")
    result = chain_query(mem, _hv(1), hops=4)
    assert len(result.hops) <= 4


# ── min_similarity ────────────────────────────────────────────────────────────


def test_min_similarity_stops_chain():
    """With min_similarity=1.0 only exact matches pass; after the first hop the key
    changes, so subsequent entries (random, near-orthogonal) won't reach 1.0."""
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "exact")
    # Add dissimilar entries
    for i in range(5):
        v = _hv(100 + i)
        mem.store(v, v, f"noise_{i}")

    result = chain_query(mem, hv, hops=3, min_similarity=1.0)
    # First hop matches exactly; subsequent hops are below 1.0 → terminates.
    assert result.hops[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert len(result.hops) <= 2


def test_min_similarity_neg_one_does_not_stop():
    """min_similarity=-1.0 accepts even negative-cosine matches, never terminates early."""
    mem = EpisodicMemory()
    for i in range(3):
        v = _hv(i + 1)
        mem.store(v, v, f"e{i}")
    result = chain_query(mem, _hv(1), hops=3, min_similarity=-1.0)
    assert len(result.hops) == 3
    assert result.terminated_early is False


# ── ChainResult helpers ───────────────────────────────────────────────────────


def test_labels_and_similarities():
    mem = EpisodicMemory()
    hv = _hv(1)
    mem.store(hv, hv, "alpha")
    result = chain_query(mem, hv, hops=1)
    assert result.labels() == ["alpha"]
    assert len(result.similarities()) == 1
    assert result.similarities()[0] > 0.0


def test_terminated_early_false_on_full_chain():
    mem = EpisodicMemory()
    for i in range(5):
        v = _hv(i + 1)
        mem.store(v, v, f"e{i}")
    result = chain_query(mem, _hv(1), hops=3)
    assert result.terminated_early is False
    assert len(result.hops) == 3
