"""Tests for kohaku.enriched — temporal validity, salience, provenance, trust."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku._pure import HyperVector
from kohaku.enriched import (
    DEFAULT_HALF_LIFE_DAYS,
    SOURCE_TRUST_WEIGHTS,
    EnrichedMemoryStore,
    EnrichedRetrievalResult,
    MemoryMetadata,
)


def _utc(year=2026, month=5, day=12, hour=12, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ────────────────────────────── MemoryMetadata ────────────────────────────────


def test_metadata_promotes_naive_datetimes_to_utc():
    m = MemoryMetadata(
        entry_id=1,
        valid_from=datetime(2026, 5, 12, 12, 0),  # naive
        valid_until=None,
        source="user_input",
        importance=0.5,
    )
    assert m.valid_from.tzinfo is not None


def test_metadata_validates_importance_range():
    with pytest.raises(ValueError, match="importance"):
        MemoryMetadata(entry_id=1, valid_from=_utc(), importance=1.5)
    with pytest.raises(ValueError, match="importance"):
        MemoryMetadata(entry_id=1, valid_from=_utc(), importance=-0.1)


def test_metadata_validates_valid_until_after_from():
    with pytest.raises(ValueError, match="valid_until"):
        MemoryMetadata(
            entry_id=1,
            valid_from=_utc(day=12),
            valid_until=_utc(day=10),
        )


def test_metadata_validates_source_nonempty():
    with pytest.raises(ValueError, match="source"):
        MemoryMetadata(entry_id=1, valid_from=_utc(), source="")


def test_metadata_validates_reinforcement_nonneg():
    with pytest.raises(ValueError, match="reinforcement"):
        MemoryMetadata(entry_id=1, valid_from=_utc(), reinforcement_count=-3)


def test_is_valid_at_respects_from_and_until():
    m = MemoryMetadata(
        entry_id=1,
        valid_from=_utc(day=10),
        valid_until=_utc(day=15),
    )
    assert not m.is_valid_at(_utc(day=9))  # before window
    assert m.is_valid_at(_utc(day=11))  # inside window
    assert not m.is_valid_at(_utc(day=16))  # after window
    # No valid_until = always valid past start
    m2 = MemoryMetadata(entry_id=2, valid_from=_utc(day=1))
    assert m2.is_valid_at(_utc(year=2030))


def test_trust_weights_default_table():
    assert SOURCE_TRUST_WEIGHTS["user_input"] == 1.0
    assert SOURCE_TRUST_WEIGHTS["agent_inference"] == 0.5
    assert SOURCE_TRUST_WEIGHTS["tool_result"] == 0.9


def test_salience_recency_halves_at_half_life():
    m = MemoryMetadata(
        entry_id=1,
        valid_from=_utc(day=1),
        source="user_input",
        importance=1.0,
        created_at=_utc(day=1),
    )
    # exactly half-life days later → recency = 0.5
    later = _utc(day=1) + timedelta(days=DEFAULT_HALF_LIFE_DAYS)
    s = m.salience(now=later)
    assert s == pytest.approx(0.5, abs=1e-4)


def test_salience_reinforcement_bumps_score():
    m = MemoryMetadata(entry_id=1, valid_from=_utc(), importance=1.0, created_at=_utc())
    base = m.salience(now=_utc())
    m.reinforcement_count = 10
    bumped = m.salience(now=_utc())
    assert bumped > base
    assert bumped == pytest.approx(base * (1 + 10 * 0.1))


def test_salience_trust_modulates_score():
    user = MemoryMetadata(
        entry_id=1,
        valid_from=_utc(),
        source="user_input",
        importance=1.0,
        created_at=_utc(),
    )
    agent = MemoryMetadata(
        entry_id=2,
        valid_from=_utc(),
        source="agent_inference",
        importance=1.0,
        created_at=_utc(),
    )
    assert agent.salience(now=_utc()) < user.salience(now=_utc())
    assert agent.salience(now=_utc()) == pytest.approx(
        user.salience(now=_utc()) * 0.5,
        abs=1e-6,
    )


# ─────────────────────────── EnrichedMemoryStore ──────────────────────────────


def _store_with_phrase(store: EnrichedMemoryStore, phrase: str, **kwargs) -> int:
    hv = HyperVector.random(seed=hash(phrase) & 0xFFFF)
    return store.store(hv, hv, label=phrase, **kwargs)


def test_query_filters_expired_by_default():
    store = EnrichedMemoryStore(capacity=20)
    now = datetime.now(timezone.utc)
    expired_id = _store_with_phrase(
        store,
        "stale",
        valid_from=now - timedelta(days=5),
        valid_until=now - timedelta(days=1),
    )
    live_id = _store_with_phrase(store, "fresh")
    probe = HyperVector.random(seed=42)
    results = store.query(probe, top_k=10)
    ids = [r.entry_id for r in results]
    assert expired_id not in ids
    assert live_id in ids


def test_query_include_expired_returns_them():
    store = EnrichedMemoryStore(capacity=10)
    now = datetime.now(timezone.utc)
    expired_id = _store_with_phrase(
        store,
        "stale",
        valid_from=now - timedelta(days=5),
        valid_until=now - timedelta(days=1),
    )
    probe = HyperVector.random(seed=11)
    out = store.query(probe, top_k=10, include_expired=True)
    assert any(r.entry_id == expired_id for r in out)


def test_query_source_filter():
    store = EnrichedMemoryStore(capacity=10)
    eid_user = _store_with_phrase(store, "u", source="user_input")
    eid_agent = _store_with_phrase(store, "a", source="agent_inference")
    probe = HyperVector.random(seed=5)
    out = store.query(probe, top_k=10, source_filter="user_input")
    ids = [r.entry_id for r in out]
    assert eid_user in ids
    assert eid_agent not in ids


def test_query_sort_by_salience_reranks():
    """A low-similarity but highly-reinforced+important+user-sourced memory
    can outrank a high-similarity ephemeral one under sort='salience'."""
    store = EnrichedMemoryStore(capacity=10)
    # important user memory that DOES NOT match the probe
    target_text = "kohaku amber"
    other = HyperVector.random(seed=900)
    eid_imp = store.store(
        other, other, "important", source="user_input", importance=1.0
    )
    # reinforce it many times
    for _ in range(20):
        store.reinforce(eid_imp)
    # match-y memory but agent-sourced + low importance
    matchy = HyperVector.random(seed=hash(target_text) & 0xFFFF)
    eid_match = store.store(
        matchy, matchy, "match", source="agent_inference", importance=0.1
    )
    # query with the matchy vector
    by_sim = store.query(matchy, top_k=2, sort="similarity", reinforce_hits=False)
    by_sal = store.query(matchy, top_k=2, sort="salience", reinforce_hits=False)
    assert by_sim[0].entry_id == eid_match  # top sim is the matchy one
    assert by_sal[0].entry_id == eid_imp  # top salience is the important one


def test_reinforce_increments_count():
    store = EnrichedMemoryStore(capacity=5)
    eid = _store_with_phrase(store, "x")
    assert store.get_metadata(eid).reinforcement_count == 0
    store.reinforce(eid)
    store.reinforce(eid, delta=3)
    assert store.get_metadata(eid).reinforcement_count == 4


def test_reinforce_unknown_id_is_noop():
    store = EnrichedMemoryStore(capacity=5)
    assert store.reinforce(999) is None


def test_query_default_reinforces_hits():
    store = EnrichedMemoryStore(capacity=5)
    eid = _store_with_phrase(store, "y")
    probe = HyperVector.random(seed=hash("y") & 0xFFFF)
    store.query(probe, top_k=1)
    assert store.get_metadata(eid).reinforcement_count == 1


def test_expire_old_drops_expired_entries():
    store = EnrichedMemoryStore(capacity=10)
    now = datetime.now(timezone.utc)
    eid_expired = _store_with_phrase(
        store,
        "x",
        valid_from=now - timedelta(days=5),
        valid_until=now - timedelta(days=1),
    )
    eid_live = _store_with_phrase(store, "y")
    dropped = store.expire_old()
    assert dropped == [eid_expired]
    assert len(store) == 1
    assert store.get_metadata(eid_expired) is None
    assert store.get_metadata(eid_live) is not None


def test_list_memories_sorts_by_salience():
    store = EnrichedMemoryStore(capacity=10)
    eid_low = _store_with_phrase(store, "low", source="agent_inference", importance=0.1)
    eid_high = _store_with_phrase(store, "high", source="user_input", importance=0.9)
    out = store.list_memories(sort="salience")
    assert out[0]["entry_id"] == eid_high
    assert out[1]["entry_id"] == eid_low


def test_list_memories_source_filter_and_limit():
    store = EnrichedMemoryStore(capacity=10)
    for i in range(3):
        _store_with_phrase(store, f"u{i}", source="user_input")
    for i in range(2):
        _store_with_phrase(store, f"a{i}", source="agent_inference")
    out = store.list_memories(source_filter="user_input", limit=2)
    assert len(out) == 2
    assert all(r["source"] == "user_input" for r in out)


def test_store_invalid_importance_raises():
    store = EnrichedMemoryStore(capacity=3)
    hv = HyperVector.random(seed=1)
    with pytest.raises(ValueError, match="importance"):
        store.store(hv, hv, "bad", importance=2.0)


def test_capacity_eviction_drops_metadata():
    """FIFO eviction in the underlying memory should also drop stale metadata."""
    store = EnrichedMemoryStore(capacity=2)
    e1 = _store_with_phrase(store, "a")
    e2 = _store_with_phrase(store, "b")
    e3 = _store_with_phrase(store, "c")  # evicts e1
    assert store.get_metadata(e1) is None
    assert store.get_metadata(e2) is not None
    assert store.get_metadata(e3) is not None


def test_query_min_similarity():
    store = EnrichedMemoryStore(capacity=5)
    _store_with_phrase(store, "x")
    probe = HyperVector.random(seed=7777)  # near-orthogonal to "x"
    out = store.query(probe, top_k=5, min_similarity=0.5, reinforce_hits=False)
    assert out == []


def test_constructor_validates_args():
    with pytest.raises(ValueError, match="half_life_days"):
        EnrichedMemoryStore(half_life_days=0)
    with pytest.raises(ValueError, match="reinforcement_k"):
        EnrichedMemoryStore(reinforcement_k=-1)


def test_enriched_result_to_dict():
    store = EnrichedMemoryStore(capacity=3)
    eid = _store_with_phrase(store, "x", importance=0.7, source="tool_result")
    probe = HyperVector.random(seed=hash("x") & 0xFFFF)
    r = store.query(probe, top_k=1)[0]
    assert isinstance(r, EnrichedRetrievalResult)
    d = r.to_dict()
    assert d["entry_id"] == eid
    assert d["source"] == "tool_result"
    assert d["importance"] == 0.7
    assert "salience" in d and "trust" in d
    assert d["valid_until"] is None


# ─────────────── ANN-narrowed packed sub-index re-ranking (C1 follow-up) ──────


def test_candidate_ids_packed_subindex_agrees_with_full_scan():
    """An ANN-narrowed query (candidate_ids) must return exactly the candidate
    subset, ranked identically to a full exact scan filtered to those ids."""
    store = EnrichedMemoryStore()
    ids = [_store_with_phrase(store, f"phrase number {i}") for i in range(20)]
    probe = HyperVector.random(seed=hash("phrase number 3") & 0xFFFF)

    full = store.query(probe, top_k=20, reinforce_hits=False)
    full_rank = [r.entry_id for r in full]

    cand = {ids[3], ids[7], ids[11], ids[15]}
    narrowed = store.query(probe, top_k=20, candidate_ids=cand, reinforce_hits=False)
    narrowed_rank = [r.entry_id for r in narrowed]

    # Only candidates come back…
    assert set(narrowed_rank) == cand
    # …and in the same relative order as the full ranking restricted to them.
    assert narrowed_rank == [eid for eid in full_rank if eid in cand]


def test_empty_candidate_set_returns_nothing():
    store = EnrichedMemoryStore()
    for i in range(5):
        _store_with_phrase(store, f"item {i}")
    probe = HyperVector.random(seed=1)
    assert store.query(probe, top_k=5, candidate_ids=set(), reinforce_hits=False) == []
