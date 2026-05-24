"""Tests for per-memory forgetting_rate override (Phase 15)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku._pure import HyperVector
from kohaku.enriched import (
    DEFAULT_HALF_LIFE_DAYS,
    EnrichedMemoryStore,
    MemoryMetadata,
)


def _utc(**kw) -> datetime:
    return datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc).replace(**kw)


def _meta(forgetting_rate=None, importance=0.5) -> MemoryMetadata:
    return MemoryMetadata(
        entry_id=1,
        valid_from=_utc(),
        importance=importance,
        forgetting_rate=forgetting_rate,
    )


# ---------------------------------------------------------------------------
# MemoryMetadata validation
# ---------------------------------------------------------------------------

def test_forgetting_rate_none_is_default() -> None:
    m = _meta()
    assert m.forgetting_rate is None


def test_forgetting_rate_positive_accepted() -> None:
    m = _meta(forgetting_rate=2.0)
    assert m.forgetting_rate == 2.0


def test_forgetting_rate_zero_rejected() -> None:
    with pytest.raises(ValueError, match="forgetting_rate"):
        _meta(forgetting_rate=0.0)


def test_forgetting_rate_negative_rejected() -> None:
    with pytest.raises(ValueError, match="forgetting_rate"):
        _meta(forgetting_rate=-1.0)


# ---------------------------------------------------------------------------
# Salience — effective half-life modulation
# ---------------------------------------------------------------------------

def test_high_forgetting_rate_decays_faster() -> None:
    """Rate > 1 → shorter half-life → lower salience for an aged memory."""
    now = _utc()
    aged_by = timedelta(days=DEFAULT_HALF_LIFE_DAYS)  # one half-life of aging
    past = now - aged_by

    fast = MemoryMetadata(entry_id=1, valid_from=past, created_at=past,
                          importance=1.0, forgetting_rate=2.0)
    slow = MemoryMetadata(entry_id=2, valid_from=past, created_at=past,
                          importance=1.0, forgetting_rate=None)

    sal_fast = fast.salience(now=now, half_life_days=DEFAULT_HALF_LIFE_DAYS)
    sal_slow = slow.salience(now=now, half_life_days=DEFAULT_HALF_LIFE_DAYS)

    assert sal_fast < sal_slow, "faster forgetting rate must yield lower salience"


def test_low_forgetting_rate_decays_slower() -> None:
    """Rate < 1 → longer half-life → higher salience for an aged memory."""
    now = _utc()
    aged_by = timedelta(days=DEFAULT_HALF_LIFE_DAYS)
    past = now - aged_by

    slow = MemoryMetadata(entry_id=1, valid_from=past, created_at=past,
                          importance=1.0, forgetting_rate=0.5)
    baseline = MemoryMetadata(entry_id=2, valid_from=past, created_at=past,
                              importance=1.0, forgetting_rate=None)

    sal_slow = slow.salience(now=now, half_life_days=DEFAULT_HALF_LIFE_DAYS)
    sal_base = baseline.salience(now=now, half_life_days=DEFAULT_HALF_LIFE_DAYS)

    assert sal_slow > sal_base, "slower forgetting rate must yield higher salience"


def test_forgetting_rate_one_matches_default() -> None:
    """forgetting_rate=1.0 must produce the same salience as no override."""
    now = _utc()
    aged_by = timedelta(days=DEFAULT_HALF_LIFE_DAYS * 2)
    past = now - aged_by

    rate1 = MemoryMetadata(entry_id=1, valid_from=past, created_at=past,
                           importance=0.7, forgetting_rate=1.0)
    default = MemoryMetadata(entry_id=2, valid_from=past, created_at=past,
                             importance=0.7, forgetting_rate=None)

    assert abs(
        rate1.salience(now=now) - default.salience(now=now)
    ) < 1e-9


def test_fresh_memory_salience_unaffected_by_rate() -> None:
    """At age=0 forgetting rate has no effect — recency=1.0 regardless."""
    now = _utc()
    fast = MemoryMetadata(entry_id=1, valid_from=now, created_at=now,
                          importance=0.8, forgetting_rate=10.0)
    baseline = MemoryMetadata(entry_id=2, valid_from=now, created_at=now,
                              importance=0.8, forgetting_rate=None)

    assert abs(fast.salience(now=now) - baseline.salience(now=now)) < 1e-6


# ---------------------------------------------------------------------------
# EnrichedMemoryStore integration
# ---------------------------------------------------------------------------

def test_store_accepts_forgetting_rate() -> None:
    store = EnrichedMemoryStore(capacity=10)
    hv = HyperVector.random(dims=store.dims, seed=1)
    eid = store.store(hv, hv, "fast memory", source="user_input",
                      forgetting_rate=3.0)
    meta = store.get_metadata(eid)
    assert meta is not None
    assert meta.forgetting_rate == 3.0


def test_store_default_forgetting_rate_is_none() -> None:
    store = EnrichedMemoryStore(capacity=10)
    hv = HyperVector.random(dims=store.dims, seed=2)
    eid = store.store(hv, hv, "normal memory")
    meta = store.get_metadata(eid)
    assert meta is not None
    assert meta.forgetting_rate is None


def test_query_ranking_respects_forgetting_rate() -> None:
    """The slow-forgetting memory must rank above the fast-forgetting one
    when the store has aged memories and we sort by salience."""
    store = EnrichedMemoryStore(capacity=10)
    hv = HyperVector.random(dims=store.dims, seed=42)

    # Store two memories with the same content but different forgetting rates.
    # Both share the same HV so cosine=1.0; salience is the tiebreaker.
    slow_id = store.store(hv, hv, "slow", source="user_input",
                          importance=0.9, forgetting_rate=0.1)
    fast_id = store.store(hv, hv, "fast", source="user_input",
                          importance=0.9, forgetting_rate=10.0)

    # Age both by manipulating created_at to 60 days ago.
    aged = datetime.now(timezone.utc) - timedelta(days=60)
    store._meta[slow_id].created_at = aged
    store._meta[fast_id].created_at = aged

    results = store.query(hv, top_k=2, sort="salience", reinforce_hits=False)
    assert len(results) == 2
    # Slow-forgetting memory (longer half-life) must score higher.
    assert results[0].entry_id == slow_id
