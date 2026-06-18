"""Tests for AsyncEpisodicMemory."""

from __future__ import annotations

import asyncio
import pytest
from kohaku._pure import HyperVector, DIMS
from kohaku._async import AsyncEpisodicMemory


def _make_vec(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


@pytest.mark.asyncio
async def test_async_store_returns_int():
    """store() must return an integer entry ID."""
    mem = AsyncEpisodicMemory(capacity=10)
    k = _make_vec(1)
    v = _make_vec(2)
    result = await mem.store(k, v, "test")
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result >= 1


@pytest.mark.asyncio
async def test_async_query_returns_list():
    """query() must return a list of RetrievalResult objects."""
    mem = AsyncEpisodicMemory(capacity=10)
    k = _make_vec(10)
    v = _make_vec(11)
    await mem.store(k, v, "alpha")
    qk = _make_vec(10)
    results = await mem.query(qk, top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_async_query_empty_memory():
    """Querying an empty memory must return an empty list."""
    mem = AsyncEpisodicMemory(capacity=10)
    qk = _make_vec(42)
    results = await mem.query(qk, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_async_store_and_query_roundtrip():
    """The stored key should be the top result when queried with the same key."""
    mem = AsyncEpisodicMemory(capacity=10)
    k = _make_vec(100)
    v = _make_vec(101)
    await mem.store(k, v, "roundtrip")
    # Add noise entries
    for i in range(5):
        ki = _make_vec(i * 200 + 300)
        vi = _make_vec(i * 200 + 301)
        await mem.store(ki, vi, f"noise-{i}")
    results = await mem.query(k, top_k=1)
    assert len(results) == 1
    assert results[0].label == "roundtrip"
    assert results[0].similarity > 0.99


@pytest.mark.asyncio
async def test_async_query_threshold():
    """query_threshold() must return only entries above the threshold."""
    mem = AsyncEpisodicMemory(capacity=10)
    k = _make_vec(50)
    v = _make_vec(51)
    await mem.store(k, v, "above")
    # Add unrelated entries
    for i in range(5):
        ki = _make_vec(i * 500 + 600)
        vi = _make_vec(i * 500 + 601)
        await mem.store(ki, vi, f"below-{i}")
    results = await mem.query_threshold(k, threshold=0.9)
    assert len(results) >= 1
    for r in results:
        assert r.similarity >= 0.9
    assert any(r.label == "above" for r in results)


@pytest.mark.asyncio
async def test_async_clear():
    """clear() must empty the memory."""
    mem = AsyncEpisodicMemory(capacity=10)
    for i in range(5):
        ki = _make_vec(i)
        vi = _make_vec(i + 100)
        await mem.store(ki, vi, f"item-{i}")
    assert len(mem) == 5
    await mem.clear()
    assert len(mem) == 0


@pytest.mark.asyncio
async def test_async_len_after_store():
    """__len__ must accurately reflect the number of stored entries."""
    mem = AsyncEpisodicMemory(capacity=10)
    assert len(mem) == 0
    for i in range(4):
        ki = _make_vec(i * 3)
        vi = _make_vec(i * 3 + 1)
        await mem.store(ki, vi, f"e-{i}")
    assert len(mem) == 4


@pytest.mark.asyncio
async def test_concurrent_stores():
    """10 concurrent stores via asyncio.gather must all complete successfully."""
    mem = AsyncEpisodicMemory(capacity=20)

    async def store_one(i: int) -> int:
        k = _make_vec(i * 17)
        v = _make_vec(i * 17 + 1)
        return await mem.store(k, v, f"concurrent-{i}")

    ids = await asyncio.gather(*[store_one(i) for i in range(10)])
    assert len(ids) == 10
    # All IDs must be integers
    assert all(isinstance(eid, int) for eid in ids)
    # Memory must contain all 10 entries
    assert len(mem) == 10
