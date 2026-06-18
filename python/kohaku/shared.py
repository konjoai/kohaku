"""Cross-agent memory sharing — per-agent write namespaces, read-all union.

The dual of :class:`~kohaku.tenant.TenantMemoryStore`. Where a tenant store
isolates *both* reads and writes per tenant, a :class:`SharedMemoryPool`
isolates only *writes* into per-agent namespaces and unions every namespace on
*read*, tagging each hit with the agent that produced it. Together they span
the privacy/collaboration axis: tenants keep memories private; a shared pool
lets a fleet of agents pool what they learn.

Read scoping (`agents=[...]`) narrows the union to a subset of namespaces, so a
pool can still serve a restricted view without a separate store.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from ._pure import EpisodicMemory, HyperVector
from ._query import query as query_topk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedRetrievalResult:
    """A retrieval hit annotated with the agent whose namespace produced it."""

    agent_id: str
    entry_id: int
    label: str
    similarity: float
    value: HyperVector


class SharedMemoryPool:
    """A pool of per-agent memory namespaces with read-all (union) retrieval.

    Each agent writes into its own isolated :class:`EpisodicMemory`. A query
    fans out across the selected namespaces, merges the per-agent top-k hits,
    and returns the global top-k tagged with the originating ``agent_id``.
    Unknown agents are auto-provisioned on first write.
    """

    def __init__(self, dimension: int, default_capacity: int = 1000):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if default_capacity < 1:
            raise ValueError("default_capacity must be >= 1")
        self._dimension = dimension
        self._default_capacity = default_capacity
        self._agents: Dict[str, EpisodicMemory] = {}

    @property
    def agent_ids(self) -> List[str]:
        """All registered agent IDs."""
        return list(self._agents.keys())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_or_create(self, agent_id: str) -> EpisodicMemory:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if agent_id not in self._agents:
            self._agents[agent_id] = EpisodicMemory(capacity=self._default_capacity)
            logger.info("SharedMemoryPool: provisioned agent '%s'", agent_id)
        return self._agents[agent_id]

    def write(
        self, agent_id: str, key: HyperVector, value: HyperVector, label: str = ""
    ) -> None:
        """Store a memory into the given agent's write namespace."""
        self._get_or_create(agent_id).store(key, value, label)

    def _read_scope(self, agents: Optional[Iterable[str]]) -> List[str]:
        """Resolve the set of agent namespaces a query reads from.

        ``None`` means read every namespace. An explicit list silently skips
        unknown agents (they simply contribute nothing to the union).
        """
        if agents is None:
            return list(self._agents.keys())
        scope: List[str] = []
        for agent_id in agents:
            if agent_id in self._agents:
                scope.append(agent_id)
            else:
                logger.debug(
                    "SharedMemoryPool: read scope skips unknown agent '%s'", agent_id
                )
        return scope

    def query(
        self,
        query_key: HyperVector,
        top_k: int = 1,
        agents: Optional[Iterable[str]] = None,
    ) -> List[SharedRetrievalResult]:
        """Retrieve the global top-k across the selected namespaces.

        Each namespace contributes its own top-k; the merged candidates are
        re-sorted by similarity and truncated to ``top_k`` globally. Pass
        ``agents`` to restrict the read view to a subset of namespaces.
        """
        if top_k <= 0:
            return []
        hits: List[SharedRetrievalResult] = []
        for agent_id in self._read_scope(agents):
            for r in query_topk(self._agents[agent_id], query_key, top_k):
                hits.append(
                    SharedRetrievalResult(
                        agent_id=agent_id,
                        entry_id=r.entry_id,
                        label=r.label,
                        similarity=r.similarity,
                        value=r.value,
                    )
                )
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:top_k]

    def size(self, agent_id: str) -> int:
        """Number of entries in an agent's namespace (0 if unknown)."""
        mem = self._agents.get(agent_id)
        return len(mem) if mem is not None else 0

    def total_size(self) -> int:
        """Total entries pooled across every namespace."""
        return sum(len(mem) for mem in self._agents.values())

    def drop_agent(self, agent_id: str) -> bool:
        """Remove an agent's namespace. Returns True if it existed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("SharedMemoryPool: dropped agent '%s'", agent_id)
            return True
        return False

    def agents_count(self) -> int:
        return len(self._agents)
