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
from .persistence import PathLike, save_namespaces, load_namespaces
from .validation import RateLimit, ValidationResult, WriteValidator

logger = logging.getLogger(__name__)

_FORMAT = "kohaku-shared-pool"
_ACCEPTED = ValidationResult(
    accepted=True, reason="accepted", nearest_similarity=0.0, nearest_label=""
)


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

    **Poisoning defense (optional).** Pass ``duplicate_threshold`` and/or
    ``rate_limit`` to gate writes through a per-agent :class:`WriteValidator`,
    so one misbehaving agent can't flood the pool with near-duplicate clones or
    a burst of writes. Each agent is validated against *its own* namespace, so
    legitimate cross-agent overlap of the same true fact is never blocked — only
    an agent re-spamming its own space is. ``write`` then returns a
    :class:`ValidationResult` (rejections are logged and do not store). Validation
    is a runtime policy, not persisted state; a pool loaded via :meth:`load`
    starts unvalidated unless reconstructed with these arguments.
    """

    def __init__(
        self,
        dimension: int,
        default_capacity: int = 1000,
        *,
        duplicate_threshold: Optional[float] = None,
        rate_limit: Optional[RateLimit] = None,
    ):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if default_capacity < 1:
            raise ValueError("default_capacity must be >= 1")
        if duplicate_threshold is not None and not (0.0 < duplicate_threshold <= 1.0):
            raise ValueError("duplicate_threshold must be in (0, 1]")
        self._dimension = dimension
        self._default_capacity = default_capacity
        self._duplicate_threshold = duplicate_threshold
        self._rate_limit = rate_limit
        self._agents: Dict[str, EpisodicMemory] = {}
        self._validators: Dict[str, WriteValidator] = {}

    @property
    def validation_enabled(self) -> bool:
        """Whether writes are gated by novelty and/or rate-limit checks."""
        return self._duplicate_threshold is not None or self._rate_limit is not None

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

    def _validator_for(self, agent_id: str) -> Optional[WriteValidator]:
        """Lazily build the per-agent validator bound to its namespace.

        Returns ``None`` when validation is disabled. Built on demand so agents
        provisioned directly (e.g. via :meth:`load`) still get a validator on
        their first write.
        """
        if not self.validation_enabled:
            return None
        validator = self._validators.get(agent_id)
        if validator is None:
            validator = WriteValidator(
                self._agents[agent_id],
                # threshold 1.0 (exact-duplicate only) when novelty isn't
                # configured but a rate limit is.
                duplicate_threshold=self._duplicate_threshold
                if self._duplicate_threshold is not None
                else 1.0,
                rate_limits={agent_id: self._rate_limit}
                if self._rate_limit is not None
                else None,
            )
            self._validators[agent_id] = validator
        return validator

    def write(
        self, agent_id: str, key: HyperVector, value: HyperVector, label: str = ""
    ) -> ValidationResult:
        """Store a memory into the given agent's write namespace.

        When poisoning defense is enabled (see the class docstring) the write is
        gated by the agent's :class:`WriteValidator`: a near-duplicate or
        rate-limit violation is rejected (logged, not stored). Returns the
        :class:`ValidationResult`; with validation disabled the write always
        succeeds and an ``accepted`` result is returned.
        """
        mem = self._get_or_create(agent_id)
        validator = self._validator_for(agent_id)
        if validator is None:
            mem.store(key, value, label)
            return _ACCEPTED
        result, _ = validator.validate_and_store(key, value, label, source=agent_id)
        if not result.accepted:
            logger.warning(
                "SharedMemoryPool: rejected write to '%s' (%s; nearest=%.3f '%s')",
                agent_id,
                result.reason,
                result.nearest_similarity,
                result.nearest_label,
            )
        return result

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
        """Remove an agent's namespace (and its validator). True if it existed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._validators.pop(agent_id, None)
            logger.info("SharedMemoryPool: dropped agent '%s'", agent_id)
            return True
        return False

    def agents_count(self) -> int:
        return len(self._agents)

    def save(self, directory: PathLike) -> None:
        """Persist every agent namespace to ``directory`` (one ``.hkb`` each).

        Round-trips exactly via :meth:`load`: each agent's pooled memories and
        id counters are preserved, so a fleet's shared memory survives a restart.
        """
        save_namespaces(
            self._agents,
            directory,
            fmt=_FORMAT,
            config={
                "dimension": self._dimension,
                "default_capacity": self._default_capacity,
            },
        )

    @classmethod
    def load(cls, directory: PathLike) -> "SharedMemoryPool":
        """Reconstruct a pool written by :meth:`save`."""
        config, namespaces = load_namespaces(directory, fmt=_FORMAT)
        pool = cls(
            dimension=int(config["dimension"]),
            default_capacity=int(config.get("default_capacity", 1000)),
        )
        pool._agents = namespaces
        return pool
