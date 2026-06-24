"""Multi-agent pool + per-tenant isolated namespaces.

Registered last in ``create_app``, matching the original order: agents
store/query/list/drop, then tenants store/query/list/drop.
"""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import Response

from kohaku import encode_text

from .._helpers import Ctx, RestState
from ..models import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentQueryResult,
    AgentStoreRequest,
    AgentStoreResponse,
    NamespaceInfo,
    NamespaceListResponse,
    TenantQueryRequest,
    TenantQueryResponse,
    TenantQueryResult,
    TenantStoreRequest,
    TenantStoreResponse,
)


def register(app: FastAPI, ctx: Ctx) -> None:
    # ── Multi-agent pool ──────────────────────────────────────────────────
    @app.post("/agents/store", response_model=AgentStoreResponse)
    def agent_store(req: AgentStoreRequest) -> AgentStoreResponse:
        """Store a memory into an agent's private write namespace.

        Unknown agents are auto-provisioned on first write. Returns the
        ValidationResult so callers can detect duplicate/rate-limit rejections.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            hv = encode_text(req.text)
            result = rest.pool.write(req.agent_id, hv, hv, req.label)
            sz = rest.pool.size(req.agent_id)
        return AgentStoreResponse(
            agent_id=req.agent_id,
            stored=result.accepted,
            reason=result.reason,
            size=sz,
        )

    @app.post("/agents/query", response_model=AgentQueryResponse)
    def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
        """Union retrieval across the pool, optionally scoped to named agents.

        Each namespace contributes its own top-k; results are re-ranked globally
        and filtered by ``threshold``. Pass ``agents`` to restrict the read view.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            hv = encode_text(req.text)
            raw = rest.pool.query(hv, top_k=req.top_k, agents=req.agents)
        results = [
            AgentQueryResult(
                agent_id=r.agent_id,
                entry_id=r.entry_id,
                label=r.label,
                similarity=r.similarity,
            )
            for r in raw
            if r.similarity >= req.threshold
        ]
        return AgentQueryResponse(results=results)

    @app.get("/agents", response_model=NamespaceListResponse)
    def list_agents() -> NamespaceListResponse:
        """List every agent namespace and its entry count."""
        rest: RestState = app.state.rest
        pool = rest.pool
        return NamespaceListResponse(
            namespaces=[NamespaceInfo(id=a, size=pool.size(a)) for a in pool.agent_ids],
            count=pool.agents_count(),
            total_size=pool.total_size(),
        )

    @app.delete("/agents", status_code=204)
    def drop_agent(
        agent_id: str = Query(..., min_length=1, max_length=200),
    ) -> Response:
        """Remove an agent namespace and all its memories (no-op if unknown)."""
        rest: RestState = app.state.rest
        with rest.lock:
            rest.pool.drop_agent(agent_id)
        return Response(status_code=204)

    # ── Tenant store ──────────────────────────────────────────────────────
    @app.post("/tenants/store", response_model=TenantStoreResponse)
    def tenant_store(req: TenantStoreRequest) -> TenantStoreResponse:
        """Store a memory into a tenant's fully isolated namespace.

        Unknown tenants are auto-provisioned on first write.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            hv = encode_text(req.text)
            rest.tenants.store(req.tenant_id, hv, hv, req.label)
            sz = rest.tenants.size(req.tenant_id)
        return TenantStoreResponse(
            tenant_id=req.tenant_id,
            stored=True,
            size=sz,
        )

    @app.post("/tenants/query", response_model=TenantQueryResponse)
    def tenant_query(req: TenantQueryRequest) -> TenantQueryResponse:
        """Retrieve from a single tenant's isolated namespace (no cross-read)."""
        rest: RestState = app.state.rest
        with rest.lock:
            hv = encode_text(req.text)
            raw = rest.tenants.retrieve(req.tenant_id, hv, top_k=req.top_k)
        results = [
            TenantQueryResult(
                entry_id=r.entry_id,
                label=r.label,
                similarity=r.similarity,
            )
            for r in raw
            if r.similarity >= req.threshold
        ]
        return TenantQueryResponse(tenant_id=req.tenant_id, results=results)

    @app.get("/tenants", response_model=NamespaceListResponse)
    def list_tenants() -> NamespaceListResponse:
        """List every tenant namespace and its entry count."""
        rest: RestState = app.state.rest
        tenants = rest.tenants
        ids = tenants.tenant_ids
        return NamespaceListResponse(
            namespaces=[NamespaceInfo(id=t, size=tenants.size(t)) for t in ids],
            count=tenants.tenants_count(),
            total_size=sum(tenants.size(t) for t in ids),
        )

    @app.delete("/tenants", status_code=204)
    def drop_tenant(
        tenant_id: str = Query(..., min_length=1, max_length=200),
    ) -> Response:
        """Remove a tenant namespace and all its memories (no-op if unknown)."""
        rest: RestState = app.state.rest
        with rest.lock:
            rest.tenants.drop_tenant(tenant_id)
        return Response(status_code=204)
