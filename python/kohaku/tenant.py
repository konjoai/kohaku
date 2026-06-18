"""Multi-tenant memory isolation — scope EpisodicMemory namespaces by tenant ID."""

from __future__ import annotations
import logging
from typing import Dict, Optional, List
from ._pure import EpisodicMemory, HyperVector
from ._query import RetrievalResult, query

logger = logging.getLogger(__name__)


class TenantMemoryStore:
    """A registry of per-tenant EpisodicMemory instances.

    Each tenant has an isolated memory namespace. Tenants cannot access each
    other's memories. Unknown tenants are auto-provisioned on first access.
    """

    def __init__(
        self,
        dimension: int,
        capacity: int = 1000,
        default_capacity: Optional[int] = None,
    ):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._dimension = dimension
        self._capacity = capacity
        self._default_capacity = default_capacity or capacity
        self._tenants: Dict[str, EpisodicMemory] = {}

    @property
    def tenant_ids(self) -> List[str]:
        """Return all registered tenant IDs."""
        return list(self._tenants.keys())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_or_create(self, tenant_id: str) -> EpisodicMemory:
        if not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = EpisodicMemory(capacity=self._default_capacity)
            logger.info("TenantMemoryStore: provisioned tenant '%s'", tenant_id)
        return self._tenants[tenant_id]

    def store(
        self, tenant_id: str, key: HyperVector, value: HyperVector, label: str = ""
    ) -> None:
        """Store a memory for the given tenant."""
        self._get_or_create(tenant_id).store(key, value, label)

    def retrieve(
        self, tenant_id: str, query_key: HyperVector, top_k: int = 1
    ) -> List[RetrievalResult]:
        """Retrieve from the given tenant's memory."""
        mem = self._get_or_create(tenant_id)
        return query(mem, query_key, top_k)

    def size(self, tenant_id: str) -> int:
        """Number of entries for a tenant (0 if tenant unknown)."""
        if tenant_id not in self._tenants:
            return 0
        return len(self._tenants[tenant_id])

    def drop_tenant(self, tenant_id: str) -> bool:
        """Remove all data for a tenant. Returns True if the tenant existed."""
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            logger.info("TenantMemoryStore: dropped tenant '%s'", tenant_id)
            return True
        return False

    def tenants_count(self) -> int:
        return len(self._tenants)
