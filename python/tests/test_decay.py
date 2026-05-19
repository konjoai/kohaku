"""Tests for kohaku.decay — exponential temporal decay on similarity."""
from __future__ import annotations


import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.decay import DecayConfig, decay_weight, query_with_decay


def test_decay_weight_at_zero_age() -> None:
    cfg = DecayConfig(half_life=10.0)
    assert decay_weight(0, cfg) == pytest.approx(1.0)


def test_decay_weight_at_half_life() -> None:
    cfg = DecayConfig(half_life=10.0)
    assert decay_weight(10, cfg) == pytest.approx(0.5)


def test_decay_weight_two_half_lives() -> None:
    cfg = DecayConfig(half_life=5.0)
    assert decay_weight(10, cfg) == pytest.approx(0.25)


def test_decay_weight_floor_clamps() -> None:
    cfg = DecayConfig(half_life=1.0, floor=0.1)
    # At age=100, raw weight is ~7.9e-31 → clamped to 0.1
    assert decay_weight(100, cfg) == pytest.approx(0.1)


def test_decay_config_validates_half_life() -> None:
    with pytest.raises(ValueError):
        DecayConfig(half_life=0)
    with pytest.raises(ValueError):
        DecayConfig(half_life=-1)


def test_decay_config_validates_floor() -> None:
    with pytest.raises(ValueError):
        DecayConfig(half_life=1, floor=-0.1)
    with pytest.raises(ValueError):
        DecayConfig(half_life=1, floor=1.5)


def test_decay_negative_age_raises() -> None:
    with pytest.raises(ValueError):
        decay_weight(-1, DecayConfig(half_life=10.0))


def test_query_with_decay_recent_beats_old_for_same_match() -> None:
    """Two entries with identical keys but different ages — recent should win."""
    mem = EpisodicMemory(capacity=100)
    target = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=2)

    # Store target first (oldest)
    old_id = mem.store(target, val, label="old")
    # Pad with 5 unrelated entries to age the first one slightly
    for i in range(5):
        mem.store(HyperVector.random(DIMS, seed=1000 + i), val, label="filler")
    # Store target again (most recent)
    new_id = mem.store(target, val, label="new")

    cfg = DecayConfig(half_life=2.0)
    # top_k=2 with target as query: both target-keyed entries have raw sim 1.0,
    # but the older one is decayed by ~0.5^(6/2) = 0.125 vs the new one at 1.0.
    results = query_with_decay(mem, target, top_k=2, config=cfg)
    assert results[0].entry_id == new_id
    assert results[1].entry_id == old_id
    assert results[0].similarity > results[1].similarity
    assert results[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert results[1].similarity == pytest.approx(0.5 ** (6 / 2.0), abs=1e-6)


def test_query_with_decay_no_decay_when_half_life_huge() -> None:
    """Massive half-life → behaves like normal query."""
    from kohaku._query import query

    mem = EpisodicMemory(capacity=10)
    target = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=2)
    mem.store(target, val, "a")
    for i in range(5):
        mem.store(HyperVector.random(DIMS, seed=100 + i), val, f"f{i}")

    cfg = DecayConfig(half_life=1e12)
    decayed = query_with_decay(mem, target, top_k=3, config=cfg)
    plain = query(mem, target, top_k=3)
    assert [r.entry_id for r in decayed] == [r.entry_id for r in plain]
    for d, p in zip(decayed, plain):
        assert d.similarity == pytest.approx(p.similarity, rel=1e-6)


def test_query_with_decay_empty_memory() -> None:
    mem = EpisodicMemory(capacity=5)
    target = HyperVector.random(DIMS, seed=1)
    assert query_with_decay(mem, target, top_k=3) == []


def test_query_with_decay_top_k_zero() -> None:
    mem = EpisodicMemory(capacity=5)
    mem.store(HyperVector.random(DIMS, seed=1), HyperVector.random(DIMS, seed=2), "a")
    target = HyperVector.random(DIMS, seed=1)
    assert query_with_decay(mem, target, top_k=0) == []


def test_query_with_decay_uses_default_config() -> None:
    mem = EpisodicMemory(capacity=5)
    target = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=2)
    mem.store(target, val, "a")
    results = query_with_decay(mem, target, top_k=1)  # no config
    assert len(results) == 1
    assert results[0].similarity == pytest.approx(1.0, abs=1e-6)
