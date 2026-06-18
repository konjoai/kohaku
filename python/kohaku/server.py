"""FastAPI REST server for Kohaku HDC episodic memory."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Response
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class StoreRequest(BaseModel):
        """Request body for POST /memory/store."""

        text: str = Field(..., max_length=1000)
        label: str = Field(default="")

    class StoreResponse(BaseModel):
        """Response for a successful store."""

        entry_id: int
        stored: bool

    class QueryRequest(BaseModel):
        """Request body for POST /memory/query."""

        text: str = Field(..., max_length=1000)
        top_k: int = Field(default=5, ge=1, le=50)
        threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    class QueryResult(BaseModel):
        """A single query result."""

        entry_id: int
        label: str
        similarity: float
        source: str = "episodic"

    class QueryResponse(BaseModel):
        """Response for POST /memory/query."""

        results: list[QueryResult]
        query_text_length: int
        top_k: int

    class StatsResponse(BaseModel):
        """Response for GET /memory/stats."""

        capacity: int
        size: int
        utilization: float
        dim: int

    class HealthResponse(BaseModel):
        """Response for GET /health."""

        status: str
        version: str

    # -- multi-agent shared pool (per-agent write, read-all union) --

    class AgentStoreRequest(BaseModel):
        """Request body for POST /agents/store."""

        agent_id: str = Field(..., min_length=1, max_length=200)
        text: str = Field(..., min_length=1, max_length=1000)
        label: str = Field(default="")

    class AgentStoreResponse(BaseModel):
        """Outcome of a pooled write (gated by the pool's poisoning defense)."""

        agent_id: str
        stored: bool
        reason: str  # "accepted" | "near_duplicate" | "rate_limit_exceeded"
        size: int

    class SharedQueryRequest(BaseModel):
        """Request body for POST /agents/query."""

        text: str = Field(..., min_length=1, max_length=1000)
        top_k: int = Field(default=5, ge=1, le=50)
        threshold: float = Field(default=0.0, ge=0.0, le=1.0)
        agents: Optional[list[str]] = None

    class SharedQueryResult(BaseModel):
        """A union-query hit, tagged with the agent whose namespace produced it."""

        agent_id: str
        entry_id: int
        label: str
        similarity: float

    class SharedQueryResponse(BaseModel):
        """Response for POST /agents/query."""

        results: list[SharedQueryResult]
        top_k: int

    # -- multi-tenant store (per-tenant isolation, read + write) --

    class TenantStoreRequest(BaseModel):
        """Request body for POST /tenants/store."""

        tenant_id: str = Field(..., min_length=1, max_length=200)
        text: str = Field(..., min_length=1, max_length=1000)
        label: str = Field(default="")

    class TenantStoreResponse(BaseModel):
        """Response for a successful tenant store."""

        tenant_id: str
        stored: bool
        size: int

    class TenantQueryRequest(BaseModel):
        """Request body for POST /tenants/query."""

        tenant_id: str = Field(..., min_length=1, max_length=200)
        text: str = Field(..., min_length=1, max_length=1000)
        top_k: int = Field(default=5, ge=1, le=50)
        threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    class NamespaceInfo(BaseModel):
        """One namespace (agent or tenant) and its entry count."""

        id: str
        size: int

    class NamespaceListResponse(BaseModel):
        """Response for GET /agents and GET /tenants."""

        namespaces: list[NamespaceInfo]
        count: int
        total_size: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(capacity: int = 1000, dim: int = 1024) -> Any:
    """Create and configure the Kohaku FastAPI application.

    Parameters
    ----------
    capacity:
        Maximum number of memory entries before FIFO eviction.
    dim:
        Hypervector dimensionality used for encoding.

    Returns
    -------
    FastAPI
        Configured application instance.

    Raises
    ------
    ImportError
        If fastapi is not installed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for the REST server. "
            "Install with: pip install kohaku[api]"
        )

    from kohaku._pure import EpisodicMemory
    from kohaku.attention import encode_text
    from kohaku._query import query as memory_query
    from kohaku.shared import SharedMemoryPool
    from kohaku.tenant import TenantMemoryStore

    app = FastAPI(title="Kohaku HDC Memory API", version="0.7.0")

    # Mutable state container so the DELETE /memory/clear handler can replace it.
    state: dict[str, Any] = {
        "memory": EpisodicMemory(capacity=capacity),
        "pool": SharedMemoryPool(dimension=dim, default_capacity=capacity),
        "tenants": TenantMemoryStore(dimension=dim, capacity=capacity),
        "capacity": capacity,
        "dim": dim,
    }

    @app.post("/memory/store", response_model=StoreResponse)
    def store_memory(request: StoreRequest) -> StoreResponse:
        """Encode *request.text* and store the resulting HyperVector."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            entry_id = state["memory"].store(hv, hv, request.label)
            return StoreResponse(entry_id=entry_id, stored=True)
        except Exception as exc:
            log.warning("store failed: %s", exc)
            raise

    @app.post("/memory/query", response_model=QueryResponse)
    def query_memory(request: QueryRequest) -> QueryResponse:
        """Encode *request.text* and query the episodic memory store."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            raw = memory_query(state["memory"], hv, top_k=request.top_k)
            filtered = [r for r in raw if r.similarity >= request.threshold]
            results = [
                QueryResult(
                    entry_id=r.entry_id,
                    label=r.label,
                    similarity=r.similarity,
                )
                for r in filtered
            ]
            return QueryResponse(
                results=results,
                query_text_length=len(request.text),
                top_k=request.top_k,
            )
        except Exception as exc:
            log.warning("query failed: %s", exc)
            raise

    @app.delete("/memory/clear", status_code=204)
    def clear_memory() -> Response:
        """Reset the memory store to empty (same capacity and dim)."""
        state["memory"] = EpisodicMemory(capacity=state["capacity"])
        return Response(status_code=204)

    @app.get("/memory/stats", response_model=StatsResponse)
    def memory_stats() -> StatsResponse:
        """Return current capacity, size, utilization, and dimension."""
        mem = state["memory"]
        size = len(mem)
        cap = state["capacity"]
        return StatsResponse(
            capacity=cap,
            size=size,
            utilization=size / cap if cap > 0 else 0.0,
            dim=state["dim"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness probe — always returns ok."""
        return HealthResponse(status="ok", version="0.7.0")

    # -- shared pool: cross-agent memory (per-agent write, read-all union) --

    @app.post("/agents/store", response_model=AgentStoreResponse)
    def agent_store(request: AgentStoreRequest) -> AgentStoreResponse:
        """Write to an agent's namespace; the pool's validator may reject it."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            result = state["pool"].write(request.agent_id, hv, hv, request.label)
            return AgentStoreResponse(
                agent_id=request.agent_id,
                stored=result.accepted,
                reason=result.reason,
                size=state["pool"].size(request.agent_id),
            )
        except Exception as exc:
            log.warning("agent_store failed: %s", exc)
            raise

    @app.post("/agents/query", response_model=SharedQueryResponse)
    def agent_query(request: SharedQueryRequest) -> SharedQueryResponse:
        """Union retrieval across the pool, optionally scoped to ``agents``."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            raw = state["pool"].query(hv, top_k=request.top_k, agents=request.agents)
            results = [
                SharedQueryResult(
                    agent_id=r.agent_id,
                    entry_id=r.entry_id,
                    label=r.label,
                    similarity=r.similarity,
                )
                for r in raw
                if r.similarity >= request.threshold
            ]
            return SharedQueryResponse(results=results, top_k=request.top_k)
        except Exception as exc:
            log.warning("agent_query failed: %s", exc)
            raise

    @app.get("/agents", response_model=NamespaceListResponse)
    def list_agents() -> NamespaceListResponse:
        """List every agent namespace and its entry count."""
        pool = state["pool"]
        return NamespaceListResponse(
            namespaces=[NamespaceInfo(id=a, size=pool.size(a)) for a in pool.agent_ids],
            count=pool.agents_count(),
            total_size=pool.total_size(),
        )

    @app.delete("/agents", status_code=204)
    def drop_agent(agent_id: str) -> Response:
        """Remove an agent's namespace (no-op if unknown)."""
        state["pool"].drop_agent(agent_id)
        return Response(status_code=204)

    # -- tenant store: isolated per-tenant memory (read + write) --

    @app.post("/tenants/store", response_model=TenantStoreResponse)
    def tenant_store(request: TenantStoreRequest) -> TenantStoreResponse:
        """Store into a tenant's isolated namespace."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            state["tenants"].store(request.tenant_id, hv, hv, request.label)
            return TenantStoreResponse(
                tenant_id=request.tenant_id,
                stored=True,
                size=state["tenants"].size(request.tenant_id),
            )
        except Exception as exc:
            log.warning("tenant_store failed: %s", exc)
            raise

    @app.post("/tenants/query", response_model=QueryResponse)
    def tenant_query(request: TenantQueryRequest) -> QueryResponse:
        """Retrieve from a single tenant's namespace (isolated — no cross-read)."""
        try:
            hv = encode_text(request.text, dims=state["dim"])
            raw = state["tenants"].retrieve(request.tenant_id, hv, top_k=request.top_k)
            results = [
                QueryResult(entry_id=r.entry_id, label=r.label, similarity=r.similarity)
                for r in raw
                if r.similarity >= request.threshold
            ]
            return QueryResponse(
                results=results,
                query_text_length=len(request.text),
                top_k=request.top_k,
            )
        except Exception as exc:
            log.warning("tenant_query failed: %s", exc)
            raise

    @app.get("/tenants", response_model=NamespaceListResponse)
    def list_tenants() -> NamespaceListResponse:
        """List every tenant namespace and its entry count."""
        tenants = state["tenants"]
        ids = tenants.tenant_ids
        return NamespaceListResponse(
            namespaces=[NamespaceInfo(id=t, size=tenants.size(t)) for t in ids],
            count=tenants.tenants_count(),
            total_size=sum(tenants.size(t) for t in ids),
        )

    @app.delete("/tenants", status_code=204)
    def drop_tenant(tenant_id: str) -> Response:
        """Remove a tenant's namespace (no-op if unknown)."""
        state["tenants"].drop_tenant(tenant_id)
        return Response(status_code=204)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    capacity: int = 1000,
    dim: int = 1024,
) -> None:
    """Start the Kohaku HTTP server with uvicorn.

    Parameters
    ----------
    host:
        Bind address (default: 127.0.0.1).
    port:
        Bind port (default: 8080).
    capacity:
        EpisodicMemory capacity (overridden by KOHAKU_CAPACITY env var).
    dim:
        HyperVector dimension (overridden by KOHAKU_DIM env var).
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install kohaku[api]"
        ) from exc

    cap = int(os.environ.get("KOHAKU_CAPACITY", capacity))
    d = int(os.environ.get("KOHAKU_DIM", dim))
    app = create_app(capacity=cap, dim=d)
    uvicorn.run(app, host=host, port=port)
