"""Async wrappers for EpisodicMemory using asyncio.to_thread."""

from __future__ import annotations

import asyncio
from typing import Iterable, List, Optional

from kohaku._pure import HyperVector, EpisodicMemory
from kohaku._query import RetrievalResult, query, query_threshold
from kohaku.persistence import PathLike
from kohaku.shared import SharedMemoryPool, SharedRetrievalResult
from kohaku.tenant import TenantMemoryStore
from kohaku.validation import RateLimit, ValidationResult


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


class AsyncTenantMemoryStore:
    """Async wrapper around :class:`~kohaku.tenant.TenantMemoryStore`.

    Per-tenant store/retrieve and the file save/load run in the default thread
    pool so a blocking Rust call never stalls the event loop. Read-only,
    non-blocking accessors (sizes, ids) pass straight through.
    """

    def __init__(
        self,
        dimension: int,
        capacity: int = 1000,
        default_capacity: Optional[int] = None,
    ) -> None:
        self._store = TenantMemoryStore(dimension, capacity, default_capacity)

    @classmethod
    def _wrap(cls, store: TenantMemoryStore) -> "AsyncTenantMemoryStore":
        obj = cls.__new__(cls)
        obj._store = store
        return obj

    async def store(
        self, tenant_id: str, key: HyperVector, value: HyperVector, label: str = ""
    ) -> None:
        await asyncio.to_thread(self._store.store, tenant_id, key, value, label)

    async def retrieve(
        self, tenant_id: str, query_key: HyperVector, top_k: int = 1
    ) -> List[RetrievalResult]:
        return await asyncio.to_thread(
            self._store.retrieve, tenant_id, query_key, top_k
        )

    async def save(self, directory: PathLike) -> None:
        await asyncio.to_thread(self._store.save, directory)

    @classmethod
    async def load(cls, directory: PathLike) -> "AsyncTenantMemoryStore":
        store = await asyncio.to_thread(TenantMemoryStore.load, directory)
        return cls._wrap(store)

    @property
    def dimension(self) -> int:
        return self._store.dimension

    @property
    def tenant_ids(self) -> List[str]:
        return self._store.tenant_ids

    def size(self, tenant_id: str) -> int:
        return self._store.size(tenant_id)

    def drop_tenant(self, tenant_id: str) -> bool:
        return self._store.drop_tenant(tenant_id)

    def tenants_count(self) -> int:
        return self._store.tenants_count()


class AsyncSharedMemoryPool:
    """Async wrapper around :class:`~kohaku.shared.SharedMemoryPool`.

    :meth:`query` is a *concurrent* fan-out: each selected namespace is searched
    in its own worker thread and the results are merged with the exact same
    ranking as the sync path. On the Rust backend the popcount kernel releases
    the GIL, so a large fleet's namespaces are searched in parallel rather than
    one after another.
    """

    def __init__(
        self,
        dimension: int,
        default_capacity: int = 1000,
        *,
        duplicate_threshold: Optional[float] = None,
        rate_limit: Optional[RateLimit] = None,
    ) -> None:
        self._pool = SharedMemoryPool(
            dimension,
            default_capacity,
            duplicate_threshold=duplicate_threshold,
            rate_limit=rate_limit,
        )

    @classmethod
    def _wrap(cls, pool: SharedMemoryPool) -> "AsyncSharedMemoryPool":
        obj = cls.__new__(cls)
        obj._pool = pool
        return obj

    async def write(
        self, agent_id: str, key: HyperVector, value: HyperVector, label: str = ""
    ) -> ValidationResult:
        return await asyncio.to_thread(self._pool.write, agent_id, key, value, label)

    async def query(
        self,
        query_key: HyperVector,
        top_k: int = 1,
        agents: Optional[Iterable[str]] = None,
    ) -> List[SharedRetrievalResult]:
        """Concurrent top-k across namespaces, ranked identically to the sync path."""
        if top_k <= 0:
            return []
        scope = self._pool._read_scope(agents)
        per_agent = await asyncio.gather(
            *(
                asyncio.to_thread(self._pool._query_agent, agent_id, query_key, top_k)
                for agent_id in scope
            )
        )
        hits = [hit for sublist in per_agent for hit in sublist]
        return self._pool._merge(hits, top_k)

    async def save(self, directory: PathLike) -> None:
        await asyncio.to_thread(self._pool.save, directory)

    @classmethod
    async def load(cls, directory: PathLike) -> "AsyncSharedMemoryPool":
        pool = await asyncio.to_thread(SharedMemoryPool.load, directory)
        return cls._wrap(pool)

    @property
    def dimension(self) -> int:
        return self._pool.dimension

    @property
    def agent_ids(self) -> List[str]:
        return self._pool.agent_ids

    @property
    def validation_enabled(self) -> bool:
        return self._pool.validation_enabled

    def size(self, agent_id: str) -> int:
        return self._pool.size(agent_id)

    def total_size(self) -> int:
        return self._pool.total_size()

    def drop_agent(self, agent_id: str) -> bool:
        return self._pool.drop_agent(agent_id)

    def agents_count(self) -> int:
        return self._pool.agents_count()
