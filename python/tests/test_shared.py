"""Tests for kohaku.shared — SharedMemoryPool cross-agent read-all union."""

from __future__ import annotations

import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.shared import SharedMemoryPool, SharedRetrievalResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


def _pool() -> SharedMemoryPool:
    return SharedMemoryPool(dimension=DIMS, default_capacity=50)


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    pool = _pool()
    assert pool.dimension == DIMS
    assert pool.agents_count() == 0
    assert pool.agent_ids == []


def test_init_bad_dimension_raises() -> None:
    with pytest.raises(ValueError):
        SharedMemoryPool(dimension=0)
    with pytest.raises(ValueError):
        SharedMemoryPool(dimension=-1)


def test_init_bad_capacity_raises() -> None:
    with pytest.raises(ValueError):
        SharedMemoryPool(dimension=DIMS, default_capacity=0)
    with pytest.raises(ValueError):
        SharedMemoryPool(dimension=DIMS, default_capacity=-5)


def test_empty_agent_id_raises_on_write() -> None:
    pool = _pool()
    with pytest.raises(ValueError):
        pool.write("", _hv(1), _hv(2))


# ---------------------------------------------------------------------------
# auto-provisioning
# ---------------------------------------------------------------------------


def test_unknown_agent_auto_provisioned_on_write() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="hello")
    assert "alice" in pool.agent_ids
    assert pool.agents_count() == 1


def test_unknown_agent_size_does_not_provision() -> None:
    pool = _pool()
    assert pool.size("ghost") == 0
    assert pool.agents_count() == 0


# ---------------------------------------------------------------------------
# read-all union (the defining behaviour)
# ---------------------------------------------------------------------------


def test_query_unions_across_agents() -> None:
    """A query sees memories written by *every* agent, not just one."""
    pool = _pool()
    key_a, val_a = _hv(1), _hv(2)
    key_b, val_b = _hv(3), _hv(4)
    pool.write("alice", key_a, val_a, label="alice-fact")
    pool.write("bob", key_b, val_b, label="bob-fact")

    # Probe with bob's key — alice can still surface bob's memory via the union
    results = pool.query(key_b, top_k=1)
    assert len(results) == 1
    assert results[0].label == "bob-fact"
    assert results[0].agent_id == "bob"


def test_query_tags_originating_agent() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="a")
    results = pool.query(_hv(1), top_k=1)
    assert results[0].agent_id == "alice"
    assert isinstance(results[0], SharedRetrievalResult)


def test_query_global_topk_across_namespaces() -> None:
    """Global top-k merges per-agent hits and re-ranks by similarity."""
    pool = _pool()
    # Three agents, one memory each; probe equals agent 5's key.
    probe = _hv(5)
    pool.write("a1", _hv(1), _hv(101), label="a1")
    pool.write("a2", _hv(5), _hv(105), label="a2")  # exact match -> sim 1.0
    pool.write("a3", _hv(9), _hv(109), label="a3")

    results = pool.query(probe, top_k=3)
    assert len(results) == 3
    # Best hit is the exact match from a2.
    assert results[0].agent_id == "a2"
    assert results[0].similarity == pytest.approx(1.0)
    # Sorted descending.
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_query_empty_pool_returns_empty() -> None:
    pool = _pool()
    assert pool.query(_hv(1), top_k=5) == []


def test_query_nonpositive_topk_returns_empty() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2))
    assert pool.query(_hv(1), top_k=0) == []
    assert pool.query(_hv(1), top_k=-3) == []


# ---------------------------------------------------------------------------
# read scoping
# ---------------------------------------------------------------------------


def test_query_scoped_to_subset_of_agents() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="alice-fact")
    pool.write("bob", _hv(1), _hv(3), label="bob-fact")

    # Restrict the read view to alice — bob's identical-key hit is excluded.
    results = pool.query(_hv(1), top_k=5, agents=["alice"])
    labels = {r.label for r in results}
    assert labels == {"alice-fact"}


def test_query_scope_skips_unknown_agents() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="alice-fact")
    results = pool.query(_hv(1), top_k=5, agents=["alice", "ghost"])
    assert len(results) == 1
    assert results[0].agent_id == "alice"


def test_query_empty_scope_returns_empty() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2))
    assert pool.query(_hv(1), top_k=5, agents=[]) == []


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------


def test_size_and_total_size() -> None:
    pool = _pool()
    for i in range(3):
        pool.write("alice", _hv(i), _hv(i + 100), label=f"a{i}")
    for i in range(5):
        pool.write("bob", _hv(i + 50), _hv(i + 150), label=f"b{i}")

    assert pool.size("alice") == 3
    assert pool.size("bob") == 5
    assert pool.total_size() == 8
    assert pool.agents_count() == 2


# ---------------------------------------------------------------------------
# drop_agent
# ---------------------------------------------------------------------------


def test_drop_agent_removes_namespace() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="x")
    assert pool.drop_agent("alice") is True
    assert pool.size("alice") == 0
    assert "alice" not in pool.agent_ids


def test_drop_nonexistent_agent_returns_false() -> None:
    pool = _pool()
    assert pool.drop_agent("nobody") is False


def test_dropped_agent_excluded_from_union() -> None:
    pool = _pool()
    pool.write("alice", _hv(1), _hv(2), label="alice-fact")
    pool.write("bob", _hv(3), _hv(4), label="bob-fact")
    pool.drop_agent("bob")
    results = pool.query(_hv(3), top_k=5)
    assert all(r.agent_id == "alice" for r in results)


# ---------------------------------------------------------------------------
# many agents
# ---------------------------------------------------------------------------


def test_many_agents_pooled_and_retrievable() -> None:
    pool = SharedMemoryPool(dimension=DIMS, default_capacity=20)
    for i in range(10):
        pool.write(
            f"agent_{i}", _hv(seed=i * 7 + 1), _hv(seed=i * 7 + 2), label=f"agent_{i}"
        )

    assert pool.agents_count() == 10
    assert pool.total_size() == 10

    # Probe matching agent_4's key surfaces agent_4 as the top hit.
    results = pool.query(_hv(seed=4 * 7 + 1), top_k=1)
    assert results[0].agent_id == "agent_4"
    assert results[0].label == "agent_4"


# ---------------------------------------------------------------------------
# poisoning defense (WriteValidator hardening)
# ---------------------------------------------------------------------------


def test_validation_disabled_by_default() -> None:
    pool = _pool()
    assert pool.validation_enabled is False
    result = pool.write("a", _hv(1), _hv(2), label="x")
    assert result.accepted and result.reason == "accepted"


def test_novelty_rejects_near_duplicate_in_same_namespace() -> None:
    pool = SharedMemoryPool(
        dimension=DIMS, default_capacity=50, duplicate_threshold=0.99
    )
    assert pool.validation_enabled is True
    first = pool.write("a", _hv(1), _hv(2), label="orig")
    assert first.accepted
    # Same key again → cosine 1.0 ≥ 0.99 → rejected, not stored.
    dup = pool.write("a", _hv(1), _hv(3), label="clone")
    assert not dup.accepted
    assert dup.reason == "near_duplicate"
    assert dup.nearest_similarity == pytest.approx(1.0, abs=1e-6)
    assert pool.size("a") == 1


def test_novelty_is_per_agent_cross_agent_overlap_allowed() -> None:
    # The same true fact learned independently by two agents must NOT be blocked.
    pool = SharedMemoryPool(
        dimension=DIMS, default_capacity=50, duplicate_threshold=0.99
    )
    assert pool.write("a", _hv(1), _hv(2), label="fact").accepted
    assert pool.write("b", _hv(1), _hv(2), label="fact").accepted  # other namespace
    assert pool.size("a") == 1 and pool.size("b") == 1


def test_rate_limit_rejects_burst() -> None:
    from kohaku.validation import RateLimit

    pool = SharedMemoryPool(
        dimension=DIMS,
        default_capacity=50,
        rate_limit=RateLimit(max_stores=2, window_seconds=60.0),
    )
    assert pool.write("a", _hv(1), _hv(101), label="1").accepted
    assert pool.write("a", _hv(2), _hv(102), label="2").accepted
    third = pool.write("a", _hv(3), _hv(103), label="3")
    assert not third.accepted
    assert third.reason == "rate_limit_exceeded"
    assert pool.size("a") == 2
    # The limit is per-agent: a different agent is unaffected.
    assert pool.write("b", _hv(4), _hv(104), label="b").accepted


def test_constructor_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="duplicate_threshold"):
        SharedMemoryPool(dimension=DIMS, duplicate_threshold=1.5)


def test_drop_agent_clears_validator_state() -> None:
    from kohaku.validation import RateLimit

    pool = SharedMemoryPool(
        dimension=DIMS,
        default_capacity=50,
        rate_limit=RateLimit(max_stores=1, window_seconds=60.0),
    )
    assert pool.write("a", _hv(1), _hv(2), label="1").accepted
    assert not pool.write("a", _hv(3), _hv(4), label="2").accepted  # over limit
    # Dropping resets the namespace and its rate-limit window.
    assert pool.drop_agent("a")
    assert pool.write("a", _hv(5), _hv(6), label="fresh").accepted
    assert pool.size("a") == 1


def test_validated_pool_memories_round_trip(tmp_path) -> None:
    pool = SharedMemoryPool(
        dimension=DIMS, default_capacity=50, duplicate_threshold=0.99
    )
    pool.write("a", _hv(1), _hv(2), label="kept")
    pool.save(tmp_path / "p")
    # Reconstruct WITH the same policy; memories persist, validation re-applies.
    loaded = SharedMemoryPool.load(tmp_path / "p")
    assert loaded.size("a") == 1
    assert loaded.validation_enabled is False  # policy is runtime, not persisted
