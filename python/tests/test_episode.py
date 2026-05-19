"""Tests for kohaku.episode — single-shot episodic role binding."""
from __future__ import annotations

import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.episode import EpisodeStore, _ROLE_SEEDS


# ── helpers ──────────────────────────────────────────────────────────────────

def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


# ── construction ─────────────────────────────────────────────────────────────

def test_dims_zero_raises():
    with pytest.raises(ValueError, match="dims"):
        EpisodeStore(dims=0)


def test_capacity_zero_raises():
    with pytest.raises(ValueError, match="capacity"):
        EpisodeStore(capacity=0)


# ── store_episode ─────────────────────────────────────────────────────────────

def test_store_no_roles_raises():
    store = EpisodeStore()
    with pytest.raises(ValueError, match="At least one role"):
        store.store_episode("empty")


def test_store_returns_entry_id():
    store = EpisodeStore()
    eid = store.store_episode("e1", who=_hv(1))
    assert isinstance(eid, int) and eid >= 1


def test_len_tracks_episodes():
    store = EpisodeStore()
    assert len(store) == 0
    store.store_episode("e1", who=_hv(1))
    store.store_episode("e2", what=_hv(2))
    assert len(store) == 2


# ── query_episode ─────────────────────────────────────────────────────────────

def test_query_no_roles_raises():
    store = EpisodeStore()
    with pytest.raises(ValueError, match="At least one role"):
        store.query_episode()


def test_full_cue_retrieves_stored_episode():
    store = EpisodeStore()
    who = _hv(10)
    what = _hv(20)
    eid = store.store_episode("meeting", who=who, what=what)
    results = store.query_episode(who=who, what=what)
    assert results[0].entry_id == eid
    assert results[0].similarity > 0.5


def test_partial_cue_single_role_retrieves():
    store = EpisodeStore()
    who = _hv(10)
    what = _hv(20)
    when = _hv(30)
    eid = store.store_episode("event", who=who, what=what, when=when)
    results = store.query_episode(who=who, top_k=1)
    assert results[0].entry_id == eid


def test_partial_cue_selects_correct_episode():
    """Two episodes stored; partial cue picks the right one."""
    store = EpisodeStore()
    who_a = _hv(100)
    who_b = _hv(200)
    what_shared = _hv(300)

    eid_a = store.store_episode("alpha", who=who_a, what=what_shared)
    eid_b = store.store_episode("beta", who=who_b, what=what_shared)

    # Query by who_a → should return eid_a first
    results = store.query_episode(who=who_a, top_k=2)
    top_ids = [r.entry_id for r in results]
    assert top_ids[0] == eid_a
    assert eid_b in top_ids


def test_roles_returned_in_result():
    store = EpisodeStore()
    who = _hv(10)
    what = _hv(20)
    store.store_episode("r", who=who, what=what)
    results = store.query_episode(who=who, top_k=1)
    assert results[0].roles.who is who
    assert results[0].roles.what is what
    assert results[0].roles.when is None


def test_single_role_episode_roundtrip():
    store = EpisodeStore()
    what = _hv(42)
    eid = store.store_episode("solo", what=what)
    results = store.query_episode(what=what, top_k=1)
    assert results[0].entry_id == eid
    assert results[0].similarity > 0.9


# ── unbind_role ───────────────────────────────────────────────────────────────

def test_unbind_role_returns_original_hv():
    store = EpisodeStore()
    who = _hv(10)
    eid = store.store_episode("e", who=who)
    recovered = store.unbind_role(eid, "who")
    assert recovered is who


def test_unbind_role_not_provided_returns_none():
    store = EpisodeStore()
    eid = store.store_episode("e", who=_hv(1))
    assert store.unbind_role(eid, "what") is None


def test_unbind_role_unknown_entry_returns_none():
    store = EpisodeStore()
    assert store.unbind_role(9999, "who") is None


def test_unbind_role_invalid_name_raises():
    store = EpisodeStore()
    eid = store.store_episode("e", who=_hv(1))
    with pytest.raises(ValueError, match="Unknown role"):
        store.unbind_role(eid, "why")


# ── role HV determinism ───────────────────────────────────────────────────────

def test_role_hvs_are_deterministic():
    """Two independent EpisodeStore instances must share identical role HVs."""
    s1 = EpisodeStore()
    s2 = EpisodeStore()
    for role in _ROLE_SEEDS:
        assert (s1._role_hvs[role].data == s2._role_hvs[role].data).all()


def test_role_hvs_are_distinct():
    store = EpisodeStore()
    roles = list(_ROLE_SEEDS.keys())
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            sim = store._role_hvs[roles[i]].cosine_similarity(store._role_hvs[roles[j]])
            assert abs(sim) < 0.1, f"Role HVs {roles[i]} and {roles[j]} too similar: {sim}"
