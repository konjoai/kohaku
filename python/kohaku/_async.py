"""Async wrappers for EpisodicMemory using asyncio.to_thread."""

from __future__ import annotations

import asyncio
from kohaku._pure import HyperVector, EpisodicMemory
from kohaku._query import RetrievalResult, query, query_threshold


class AsyncEpisodicMemory:
    """Async wrapper around EpisodicMemory. All operations run in a thread pool."""

    def __init__(self, capacity: int = 1000) -> None:
        self._mem = EpisodicMemory(capacity)

    async def store(self, key: HyperVector, value: HyperVector, label: str) -> int:
        return await asyncio.to_thread(self._mem.store, key, value, label)

    async def query(
        self, query_key: HyperVector, top_k: int = 5
    ) -> list[RetrievalResult]:
        return await asyncio.to_thread(query, self._mem, query_key, top_k)

    async def query_threshold(
        self, query_key: HyperVector, threshold: float = 0.5
    ) -> list[RetrievalResult]:
        return await asyncio.to_thread(query_threshold, self._mem, query_key, threshold)

    async def clear(self) -> None:
        return await asyncio.to_thread(self._mem.clear)

    def __len__(self) -> int:
        return len(self._mem)
