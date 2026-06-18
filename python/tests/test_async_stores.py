"""Tests for the async multi-agent store wrappers.

AsyncTenantMemoryStore and AsyncSharedMemoryPool offload blocking work to a
thread pool; the pool's query is a concurrent fan-out that must rank identically
to the sync path.
"""

from __future__ import annotations

import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku._async import AsyncSharedMemoryPool, AsyncTenantMemoryStore
from kohaku.shared import SharedMemoryPool
from kohaku.validation import RateLimit


def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


# ── AsyncTenantMemoryStore ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_tenant_store_retrieve_isolation():
    store = AsyncTenantMemoryStore(dimension=DIMS, capacity=50)
    k = _hv(1)
    await store.store("alice", k, _hv(2), "a-fact")
    await store.store("bob", _hv(3), _hv(4), "b-fact")

    hit = (await store.retrieve("alice", k, top_k=1))[0]
    assert hit.label == "a-fact"
    # Isolation: alice's key is absent from bob's namespace.
    assert (
        await store.retrieve("bob", k, top_k=1) == []
        or (await store.retrieve("bob", k, top_k=1))[0].label != "a-fact"
    )
    assert store.tenants_count() == 2
    assert store.size("alice") == 1


@pytest.mark.asyncio
async def test_async_tenant_store_save_load(tmp_path):
    store = AsyncTenantMemoryStore(dimension=DIMS, capacity=50)
    await store.store("t", _hv(5), _hv(6), "kept")
    await store.save(tmp_path / "t")

    loaded = await AsyncTenantMemoryStore.load(tmp_path / "t")
    assert loaded.size("t") == 1
    assert loaded.tenant_ids == ["t"]
    assert loaded.drop_tenant("t") is True


# ── AsyncSharedMemoryPool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_pool_write_and_union_query():
    pool = AsyncSharedMemoryPool(dimension=DIMS, default_capacity=50)
    await pool.write("alice", _hv(1), _hv(2), "alice-fact")
    await pool.write("bob", _hv(3), _hv(4), "bob-fact")

    top = (await pool.query(_hv(3), top_k=1))[0]
    assert top.agent_id == "bob"
    assert top.label == "bob-fact"
    assert pool.total_size() == 2


@pytest.mark.asyncio
async def test_async_pool_fanout_matches_sync_ranking():
    """The concurrent fan-out must produce exactly the sync pool's ranking."""
    sync = SharedMemoryPool(dimension=DIMS, default_capacity=50)
    apool = AsyncSharedMemoryPool(dimension=DIMS, default_capacity=50)
    for i in range(8):
        k, v, label = _hv(i), _hv(i + 100), f"m{i}"
        sync.write(f"agent_{i}", k, v, label)
        await apool.write(f"agent_{i}", k, v, label)

    probe = _hv(5)
    sync_hits = sync.query(probe, top_k=5)
    async_hits = await apool.query(probe, top_k=5)
    assert [(h.agent_id, h.label) for h in async_hits] == [
        (h.agent_id, h.label) for h in sync_hits
    ]
    assert [h.similarity for h in async_hits] == [h.similarity for h in sync_hits]


@pytest.mark.asyncio
async def test_async_pool_read_scoping_and_empty():
    pool = AsyncSharedMemoryPool(dimension=DIMS, default_capacity=50)
    await pool.write("a", _hv(1), _hv(2), "a")
    await pool.write("b", _hv(1), _hv(3), "b")
    scoped = await pool.query(_hv(1), top_k=5, agents=["a"])
    assert {h.agent_id for h in scoped} == {"a"}
    assert await pool.query(_hv(1), top_k=0) == []
    assert await pool.query(_hv(1), top_k=5, agents=[]) == []


@pytest.mark.asyncio
async def test_async_pool_validation_rejects_near_duplicate():
    pool = AsyncSharedMemoryPool(
        dimension=DIMS, default_capacity=50, duplicate_threshold=0.99
    )
    assert pool.validation_enabled is True
    assert (await pool.write("a", _hv(1), _hv(2), "orig")).accepted
    dup = await pool.write("a", _hv(1), _hv(3), "clone")
    assert not dup.accepted and dup.reason == "near_duplicate"
    assert pool.size("a") == 1


@pytest.mark.asyncio
async def test_async_pool_rate_limit_and_save_load(tmp_path):
    pool = AsyncSharedMemoryPool(
        dimension=DIMS,
        default_capacity=50,
        rate_limit=RateLimit(max_stores=1, window_seconds=60.0),
    )
    assert (await pool.write("a", _hv(1), _hv(2), "1")).accepted
    assert not (await pool.write("a", _hv(3), _hv(4), "2")).accepted

    await pool.save(tmp_path / "p")
    loaded = await AsyncSharedMemoryPool.load(tmp_path / "p")
    assert loaded.size("a") == 1
    assert loaded.validation_enabled is False  # policy not persisted
