"""FastAPI REST server for Kohaku HDC episodic memory."""
from __future__ import annotations

import logging
import os
from typing import Any

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

    app = FastAPI(title="Kohaku HDC Memory API", version="0.7.0")

    # Mutable state container so the DELETE /memory/clear handler can replace it.
    state: dict[str, Any] = {
        "memory": EpisodicMemory(capacity=capacity),
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
