"""Core surface: service descriptor, liveness/stats, viz, REST HDC, kyro bridge.

Registered first in ``create_app``. Route registration order within this module
matches the original ``api/main.py`` exactly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Union

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from kohaku import (
    DecayConfig,
    HyperVector,
    _BACKEND,
    query as _kohaku_query,
    query_with_decay,
)
from kohaku import __version__ as KOHAKU_VERSION
from kohaku._pure import DIMS

from .._helpers import (
    DEFAULT_HALF_LIFE,
    DEFAULT_HORIZON,
    DEFAULT_K,
    DEFAULT_STEPS,
    DEFAULT_THRESHOLD,
    LIVE_HTML,
    MEMORY_MAP_HTML,
    Ctx,
    RestState,
    VizState,
)
from ..models import (
    BridgeChunk,
    BridgeIngestRequest,
    BridgeIngestResponse,
    BridgeRetrieveRequest,
    BridgeRetrieveResponse,
    BundleRequest,
    BundleResponse,
    EncodeRequest,
    EncodeResponse,
    HealthResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    StoreRequest,
    StoreResponse,
)


def register(app: FastAPI, ctx: Ctx) -> None:
    # ── Service descriptor ───────────────────────────────────────────────
    @app.get("/")
    def root() -> Dict[str, Any]:
        viz: VizState = app.state.viz
        return {
            "name": "kohaku-api",
            "version": KOHAKU_VERSION,
            "backend": _BACKEND,
            "dims": DIMS,
            "num_concepts": len(viz.memory.entries()),
            "endpoints": [
                "/health",
                "/stats",
                "/viz/graph",
                "/viz/decay",
                "/viz/probe",
                "/viz/memory_map.html",
                "/live",
                "/encode",
                "/store",
                "/query",
                "/bundle",
            ],
        }

    # ── Liveness + stats ─────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", backend=_BACKEND)

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        rest: RestState = app.state.rest
        with rest.lock:
            iterations = sum(p.n_examples for p in rest.semantic._protos.values())
            return StatsResponse(
                backend=_BACKEND,
                version=KOHAKU_VERSION,
                dims=rest.dims,
                episodic_size=len(rest.episodic),
                episodic_capacity=rest.episodic._capacity,
                semantic_concepts=len(rest.semantic),
                learning_iterations=iterations,
                uptime_seconds=time.time() - rest.started_at,
            )

    # ── Visualization endpoints ──────────────────────────────────────────
    @app.get("/viz/graph")
    def viz_graph(
        threshold: float = Query(DEFAULT_THRESHOLD, ge=-1.0, le=1.0),
        k: int = Query(DEFAULT_K, ge=1, le=32),
        half_life: float = Query(DEFAULT_HALF_LIFE, gt=0.0),
    ) -> Dict[str, Any]:
        return app.state.viz.graph(threshold=threshold, k=k, half_life=half_life)

    @app.get("/viz/decay")
    def viz_decay(
        half_life: float = Query(DEFAULT_HALF_LIFE, gt=0.0),
        horizon: int = Query(DEFAULT_HORIZON, ge=1, le=100_000),
        steps: int = Query(DEFAULT_STEPS, ge=2, le=500),
    ) -> Dict[str, Any]:
        return app.state.viz.decay_curves(
            half_life=half_life, horizon=horizon, steps=steps
        )

    @app.post("/viz/probe")
    def viz_probe(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        text = (payload.get("text") or payload.get("query") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="'text' is required")
        try:
            top_k = int(payload.get("top_k") or 5)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid top_k: {exc}"
            ) from exc
        return app.state.viz.probe(text, top_k=top_k)

    @app.get("/viz/memory_map.html", response_class=FileResponse)
    def memory_map_html() -> FileResponse:
        if not MEMORY_MAP_HTML.exists():
            raise HTTPException(status_code=404, detail="memory_map.html not found")
        return FileResponse(MEMORY_MAP_HTML, media_type="text/html")

    @app.get("/live", response_class=FileResponse)
    def live_ui() -> FileResponse:
        if not LIVE_HTML.exists():
            raise HTTPException(status_code=404, detail="kohaku-live.html not found")
        return FileResponse(LIVE_HTML, media_type="text/html")

    # ── REST encoding / storage / retrieval ──────────────────────────────
    @app.post("/encode", response_model=EncodeResponse)
    def encode(req: EncodeRequest) -> EncodeResponse:
        hv = ctx.encode(req.input, req.type)
        return EncodeResponse(vector=hv.data.tolist(), dims=len(hv))

    @app.post("/store", response_model=StoreResponse)
    def store(req: StoreRequest) -> StoreResponse:
        rest: RestState = app.state.rest
        hv = ctx.encode(req.input, req.type)
        with rest.lock:
            entry_id = rest.episodic.store(hv, hv, req.label)
            # Also feed the example into semantic memory so /stats can report
            # online-learning iterations and so prototypes accumulate by label.
            rest.semantic.add(req.label, hv)
            size = len(rest.episodic)
        return StoreResponse(
            id=entry_id, label=req.label, dims=len(hv), episodic_size=size
        )

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(req: QueryRequest) -> QueryResponse:
        rest: RestState = app.state.rest
        with rest.lock:
            if req.label is not None:
                proto = rest.semantic.get(req.label)
                if proto is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"no semantic prototype for label {req.label!r}",
                    )
                probe = proto.vector
            else:
                probe = ctx.encode(req.input, req.type)  # type: ignore[arg-type]

            if rest.episodic.is_empty:
                return QueryResponse(results=[], top_k=req.top_k, decay_applied=False)

            raw_hits = _kohaku_query(rest.episodic, probe, top_k=req.top_k)

            decay_applied = req.half_life is not None
            decayed_map: dict[int, float] = {}
            if decay_applied:
                cfg = DecayConfig(half_life=req.half_life, floor=req.floor)  # type: ignore[arg-type]
                for r in query_with_decay(
                    rest.episodic, probe, top_k=req.top_k, config=cfg
                ):
                    decayed_map[r.entry_id] = r.similarity

        hits = [
            QueryHit(
                entry_id=r.entry_id,
                label=r.label,
                similarity=r.similarity,
                decayed_similarity=decayed_map.get(r.entry_id)
                if decay_applied
                else None,
            )
            for r in raw_hits
        ]
        return QueryResponse(results=hits, top_k=req.top_k, decay_applied=decay_applied)

    @app.post("/bundle", response_model=BundleResponse)
    def bundle(req: BundleRequest) -> BundleResponse:
        hvs = [ctx.encode(item, req.type) for item in req.inputs]
        bundled = HyperVector.bundle_all(hvs)
        return BundleResponse(
            vector=bundled.data.tolist(), dims=len(bundled), n_inputs=len(hvs)
        )

    # ── kyro bridge ──────────────────────────────────────────────────────
    @app.post("/bridge/ingest", response_model=BridgeIngestResponse)
    def bridge_ingest(req: BridgeIngestRequest) -> BridgeIngestResponse:
        """Ingest documents into the HDC retrieval store (kyro RAG backend)."""
        rest: RestState = app.state.rest
        payload: list[Union[str, dict]] = []
        for d in req.documents:
            if isinstance(d, str):
                payload.append(d)
            else:
                payload.append(
                    {"text": d.text, "id": d.id} if d.id else {"text": d.text}
                )
        with rest.lock:
            try:
                ids = rest.bridge.ingest(payload)
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            total = len(rest.bridge)
        return BridgeIngestResponse(entry_ids=ids, total_chunks=total)

    @app.post("/bridge/retrieve", response_model=BridgeRetrieveResponse)
    def bridge_retrieve(req: BridgeRetrieveRequest) -> BridgeRetrieveResponse:
        """HDC-powered top-k retrieval for kyro, with optional Ebbinghaus decay."""
        rest: RestState = app.state.rest
        with rest.lock:
            chunks = rest.bridge.retrieve(
                req.query,
                top_k=req.top_k,
                half_life=req.half_life,
                floor=req.floor,
            )
            total = len(rest.bridge)
        return BridgeRetrieveResponse(
            results=[BridgeChunk(**c.to_dict()) for c in chunks],
            total_chunks=total,
            decay_applied=req.half_life is not None,
        )
