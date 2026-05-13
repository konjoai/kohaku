"""Kohaku unified HTTP surface — FastAPI app exposing both:

  • Visualization API (/viz/*)  — read-only force-directed graph + decay
                                   curves over a sample EpisodicMemory.
  • REST HDC API (/encode, /store, /query, /bundle, /stats, /health)
                               — write-able episodic + semantic memory
                                 driven by the live `kohaku` library.

All numbers are computed by the live kohaku library — no mocks.

Endpoints
=========
GET  /                          — service descriptor
GET  /health                    — liveness probe
GET  /stats                     — runtime stats over the REST-side state
GET  /viz/graph                 — nodes + edges for the force-directed graph
GET  /viz/decay                 — per-concept time-decay curves
POST /viz/probe                 — query → ranked nearest neighbours
GET  /viz/memory_map.html       — serves the interactive viewer
POST /encode                    — text|vector → bipolar ±1 hypervector
POST /store                     — encode + persist (also feeds semantic memory)
POST /query                     — top-k associative retrieval (optional decay)
POST /bundle                    — bundle_all over a list of inputs
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

# Ensure the in-repo kohaku package is importable when this module runs from a
# checkout (no `pip install` required for development / tests).
ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from kohaku import (  # noqa: E402
    DecayConfig,
    EnrichedMemoryStore,
    EnrichedRetrievalResult,
    EpisodicMemory,
    GraphExportConfig,
    HDCRetriever,
    HyperVector,
    ItemMemory,
    MemoryGraphExporter,
    SOURCE_TRUST_WEIGHTS,
    SleepConsolidator,
    SleepReport,
    _BACKEND,
    decay_weight,
    encode_text,
    query as _kohaku_query,
    query_with_decay,
)
from kohaku import __version__ as KOHAKU_VERSION  # noqa: E402
from kohaku._pure import DIMS  # noqa: E402
from kohaku._pure import HyperVector as _PyHyperVector  # noqa: E402

from datetime import datetime, timezone  # noqa: E402

DEMO_DIR = ROOT / "demo"
SAMPLE_PATH = DEMO_DIR / "sample_memory.json"
MEMORY_MAP_HTML = DEMO_DIR / "memory_map.html"

DEFAULT_THRESHOLD = 0.7
DEFAULT_K = 3
DEFAULT_HALF_LIFE = 10.0
DEFAULT_HORIZON = 60
DEFAULT_STEPS = 30


# ═══════════════════════════════════════════════════════════════════════════
#  VIZ — k-means + VizState (read-only sample-backed view)
# ═══════════════════════════════════════════════════════════════════════════

def _kmeans_cosine(
    hvs: List[Any],
    k: int,
    *,
    max_iter: int = 30,
) -> List[int]:
    """Cosine k-means over bipolar hypervectors.

    Centroids are seeded with the first ``k`` hypervectors (deterministic — no
    randomness in the visualisation layer). At each iteration members are
    re-assigned by max cosine and centroids are re-binarised to ±1 by majority
    vote. Returns a list of length ``len(hvs)`` of cluster indices.
    """
    n = len(hvs)
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == 1:
        return [0] * n

    data = np.stack([hv.data.astype(np.float32) for hv in hvs], axis=0)  # (n, D)
    centroids = data[:k].copy()
    assignments = np.full(n, -1, dtype=np.int64)

    for _ in range(max_iter):
        d_norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
        c_norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        sims = (data / d_norm) @ (centroids / c_norm).T
        new_assign = sims.argmax(axis=1)
        if np.array_equal(new_assign, assignments):
            break
        assignments = new_assign
        for ci in range(k):
            members = data[assignments == ci]
            if members.size == 0:
                continue
            summed = members.sum(axis=0)
            centroids[ci] = np.where(summed >= 0.0, 1.0, -1.0)

    return [int(x) for x in assignments]


class VizState:
    """Loads sample concepts into a real EpisodicMemory and serves /viz queries."""

    def __init__(
        self,
        memory: Optional[EpisodicMemory] = None,
        concepts: Optional[List[Dict[str, Any]]] = None,
        sample_path: Path = SAMPLE_PATH,
    ) -> None:
        self.memory: EpisodicMemory = memory or EpisodicMemory(capacity=512)
        self.concepts: List[Dict[str, Any]] = []

        if concepts is None:
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"sample memory file not found: {sample_path}"
                )
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            concepts = payload.get("concepts") or []

        self._load_concepts(concepts)

    def _load_concepts(self, items: List[Dict[str, Any]]) -> None:
        for c in items:
            phrase = c.get("phrase") or c.get("id") or ""
            if not phrase:
                raise ValueError(f"concept missing 'phrase' or 'id': {c!r}")
            hv = encode_text(phrase)
            eid = self.memory.store(hv, hv, label=c["id"])
            self.concepts.append({
                "entry_id": eid,
                "id": c["id"],
                "label": c.get("label", c["id"]),
                "phrase": phrase,
                "cluster_label": c.get("cluster_label", ""),
                "color": c.get("color"),
            })

    # ── /viz/graph ───────────────────────────────────────────────────────
    def graph(self, threshold: float, k: int, half_life: float) -> Dict[str, Any]:
        entries = self.memory.entries()
        hvs = [e.key for e in entries]
        cluster_idx = _kmeans_cosine(hvs, k=k)
        now_clock = self.memory._timestamp

        nodes: List[Dict[str, Any]] = []
        cfg = DecayConfig(half_life=half_life)
        for i, e in enumerate(entries):
            meta = self.concepts[i] if i < len(self.concepts) else {}
            age = max(0, now_clock - 1 - e.timestamp)
            w = decay_weight(age, cfg)
            nodes.append({
                "id": e.label,
                "entry_id": e.id,
                "label": meta.get("label", e.label),
                "cluster": int(cluster_idx[i]) if i < len(cluster_idx) else 0,
                "cluster_label": meta.get("cluster_label", ""),
                "color": meta.get("color"),
                "last_accessed": e.timestamp,
                "age": age,
                "decay_weight": round(float(w), 4),
            })

        edges: List[Dict[str, Any]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = float(entries[i].key.cosine_similarity(entries[j].key))
                if sim >= threshold:
                    edges.append({
                        "source": entries[i].label,
                        "target": entries[j].label,
                        "similarity": round(sim, 4),
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "dims": DIMS,
            "threshold": threshold,
            "num_clusters": int(max(cluster_idx) + 1) if cluster_idx else 0,
            "half_life": half_life,
            "current_clock": now_clock,
        }

    # ── /viz/decay ───────────────────────────────────────────────────────
    def decay_curves(
        self, half_life: float, horizon: int, steps: int
    ) -> Dict[str, Any]:
        entries = self.memory.entries()
        cfg = DecayConfig(half_life=half_life)
        now = max(0, self.memory._timestamp - 1)
        ages = sorted({int(round(a)) for a in np.linspace(0, horizon, steps)})

        concepts_out: List[Dict[str, Any]] = []
        for i, e in enumerate(entries):
            current_age = max(0, now - e.timestamp)
            curve = [
                {"age": int(a), "weight": round(float(decay_weight(int(a), cfg)), 5)}
                for a in ages
            ]
            meta = self.concepts[i] if i < len(self.concepts) else {}
            concepts_out.append({
                "id": e.label,
                "label": meta.get("label", e.label),
                "color": meta.get("color"),
                "last_accessed": int(e.timestamp),
                "current_age": int(current_age),
                "current_weight": round(float(decay_weight(current_age, cfg)), 5),
                "curve": curve,
            })

        return {
            "concepts": concepts_out,
            "half_life": half_life,
            "horizon": horizon,
            "steps": steps,
        }

    # ── /viz/probe ───────────────────────────────────────────────────────
    def probe(self, text: str, top_k: int) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("query text must be non-empty")
        q = encode_text(text)
        ranked: List[Dict[str, Any]] = []
        for e in self.memory.entries():
            ranked.append({
                "id": e.label,
                "entry_id": e.id,
                "similarity": round(float(e.key.cosine_similarity(q)), 4),
            })
        ranked.sort(key=lambda r: r["similarity"], reverse=True)
        return {
            "query": text,
            "top_k": top_k,
            "matches": ranked[:max(0, int(top_k))],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  REST — write-able state + Pydantic models
# ═══════════════════════════════════════════════════════════════════════════

class RestState:
    """Process-wide HDC state for the write-able REST surface. Guarded by a
    lock so requests can run on a threadpool without corrupting the FIFO
    entry list."""

    def __init__(self, capacity: int = 10_000, dims: int = DIMS) -> None:
        self.dims = dims
        self.episodic = EpisodicMemory(capacity=capacity)
        self.semantic = ItemMemory(dims=dims)
        # Separate retriever for the kyro bridge — keeps RAG chunks isolated
        # from the general-purpose episodic store so /query and /bridge/retrieve
        # don't pollute each other.
        self.bridge = HDCRetriever(capacity=capacity, dims=dims)
        # Enriched store (v0.10.0): temporal validity + salience + provenance.
        # Lives alongside the plain `episodic` store so the legacy /store /query
        # endpoints stay unchanged; the new /memories/* endpoints use this.
        self.enriched = EnrichedMemoryStore(capacity=capacity, dims=dims)
        # Sleep-phase consolidation daemon over the enriched store's episodic
        # memory. Started lazily by /consolidate when a background thread is
        # explicitly requested; manual run_once() runs synchronously.
        self.sleep = SleepConsolidator(
            self.enriched.episodic,
            consolidation_interval_minutes=60.0,
            similarity_threshold=0.85,
        )
        self.lock = threading.Lock()
        self.started_at = time.time()


InputType = Literal["text", "vector"]


class EncodeRequest(BaseModel):
    input: Union[str, List[float]]
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "EncodeRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector":
            if not isinstance(self.input, list):
                raise ValueError("input must be a list of floats when type='vector'")
            if len(self.input) != DIMS:
                raise ValueError(f"vector input must have length {DIMS}, got {len(self.input)}")
        return self


class EncodeResponse(BaseModel):
    vector: List[int]
    dims: int


class StoreRequest(BaseModel):
    label: str = Field(..., min_length=1)
    input: Union[str, List[float]]
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "StoreRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector" and (
            not isinstance(self.input, list) or len(self.input) != DIMS
        ):
            raise ValueError(f"vector input must be a list of length {DIMS}")
        return self


class StoreResponse(BaseModel):
    id: int
    label: str
    dims: int
    episodic_size: int


class QueryRequest(BaseModel):
    input: Optional[Union[str, List[float]]] = None
    label: Optional[str] = None
    type: InputType = "text"
    top_k: int = Field(5, ge=1, le=100)
    half_life: Optional[float] = Field(
        default=None,
        description="If set, apply Ebbinghaus decay with this half-life (in store ticks).",
        gt=0.0,
    )
    floor: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_probe(self) -> "QueryRequest":
        if (self.input is None) == (self.label is None):
            raise ValueError("provide exactly one of `input` or `label`")
        if self.type == "vector" and isinstance(self.input, list) and len(self.input) != DIMS:
            raise ValueError(f"vector input must be a list of length {DIMS}")
        return self


class QueryHit(BaseModel):
    entry_id: int
    label: str
    similarity: float
    decayed_similarity: Optional[float] = None


class QueryResponse(BaseModel):
    results: List[QueryHit]
    top_k: int
    decay_applied: bool


class BundleRequest(BaseModel):
    inputs: List[Union[str, List[float]]] = Field(..., min_length=1)
    type: InputType = "text"

    @model_validator(mode="after")
    def _check_shape(self) -> "BundleRequest":
        if self.type == "text" and not all(isinstance(x, str) for x in self.inputs):
            raise ValueError("all inputs must be strings when type='text'")
        if self.type == "vector":
            for v in self.inputs:
                if not isinstance(v, list) or len(v) != DIMS:
                    raise ValueError(f"each vector input must be a list of length {DIMS}")
        return self


class BundleResponse(BaseModel):
    vector: List[int]
    dims: int
    n_inputs: int


class StatsResponse(BaseModel):
    backend: str
    version: str
    dims: int
    episodic_size: int
    episodic_capacity: int
    semantic_concepts: int
    learning_iterations: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str


# ── kyro bridge models ────────────────────────────────────────────────────────

class BridgeDoc(BaseModel):
    text: str = Field(..., min_length=1)
    id: Optional[str] = None


class BridgeIngestRequest(BaseModel):
    documents: List[Union[str, BridgeDoc]] = Field(..., min_length=1)


class BridgeIngestResponse(BaseModel):
    entry_ids: List[int]
    total_chunks: int


class BridgeRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    half_life: Optional[float] = Field(default=None, gt=0.0)
    floor: float = Field(0.0, ge=0.0, le=1.0)


class BridgeChunk(BaseModel):
    entry_id: int
    doc_id: str
    text: str
    similarity: float
    decayed_similarity: Optional[float] = None
    age: int


class BridgeRetrieveResponse(BaseModel):
    results: List[BridgeChunk]
    total_chunks: int
    decay_applied: bool


# ── Enriched memory request/response models ──────────────────────────────────

class EnrichedStoreRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    input: Union[str, List[float]]
    type: InputType = "text"
    source: str = Field("user_input", min_length=1, max_length=50)
    importance: float = Field(0.5, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    @model_validator(mode="after")
    def _check_shape(self) -> "EnrichedStoreRequest":
        if self.type == "text" and not isinstance(self.input, str):
            raise ValueError("input must be a string when type='text'")
        if self.type == "vector" and (
            not isinstance(self.input, list) or len(self.input) != DIMS
        ):
            raise ValueError(f"vector input must be a list of length {DIMS}")
        if self.valid_until is not None and self.valid_from is not None:
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until must be >= valid_from")
        return self


class EnrichedStoreResponse(BaseModel):
    entry_id: int
    label: str
    source: str
    importance: float
    valid_from: str
    valid_until: Optional[str] = None
    total_memories: int


class EnrichedQueryRequest(BaseModel):
    input: Union[str, List[float]]
    type: InputType = "text"
    top_k: int = Field(5, ge=1, le=100)
    sort: Literal["similarity", "salience", "recency"] = "similarity"
    source_filter: Optional[str] = Field(None, max_length=50)
    include_expired: bool = False
    min_similarity: Optional[float] = Field(None, ge=-1.0, le=1.0)
    reinforce_hits: bool = True


class EnrichedQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    top_k: int
    sort: str


class ConsolidateRequest(BaseModel):
    similarity_threshold: Optional[float] = Field(None, ge=-1.0, le=1.0)


class ConsolidateResponse(BaseModel):
    started_at: str
    run_seconds: float
    episodes_before: int
    episodes_after: int
    episodes_consolidated: int
    prototypes_created: int
    memory_freed: int
    similarity_threshold: float


def _vec_input_to_hv(values: List[float]) -> HyperVector:
    """Binarize an arbitrary float vector into a bipolar ±1 hypervector.

    Sign rule matches the rest of the codebase: zero/positive → +1, negative → −1.
    Enforces the project-wide invariant that every HDC operation receives bipolar input.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (DIMS,):
        raise HTTPException(status_code=422, detail=f"vector must have shape ({DIMS},)")
    bits = np.where(arr >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    return HyperVector(bits)


def _encode(input_value: Union[str, List[float]], input_type: InputType) -> HyperVector:
    if input_type == "text":
        if not isinstance(input_value, str):
            raise HTTPException(status_code=422, detail="text input must be a string")
        return encode_text(input_value)
    return _vec_input_to_hv(input_value)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
#  App factory — registers BOTH /viz/* and the REST surface on one app
# ═══════════════════════════════════════════════════════════════════════════

def create_app(
    viz_state: Optional[VizState] = None,
    rest_state: Optional[RestState] = None,
    *,
    state: Optional[VizState] = None,  # legacy alias for `viz_state`
) -> FastAPI:
    if viz_state is None and state is not None:
        viz_state = state
    app = FastAPI(
        title="kohaku HDC API",
        version=KOHAKU_VERSION,
        description=(
            "Unified HTTP surface for kohaku — visualization endpoints over a "
            "sample EpisodicMemory plus a write-able REST API for HDC encoding, "
            "storage, retrieval, and bundling."
        ),
    )
    # Viz state is optional — instantiating loads sample_memory.json which may
    # not exist in every deployment; fall back to an empty state if missing.
    if viz_state is None:
        try:
            viz_state = VizState()
        except FileNotFoundError:
            viz_state = VizState(concepts=[])
    app.state.viz = viz_state

    if rest_state is None:
        capacity = int(os.environ.get("KOHAKU_CAPACITY", "10000"))
        rest_state = RestState(capacity=capacity, dims=DIMS)
    app.state.rest = rest_state

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
                "/health", "/stats",
                "/viz/graph", "/viz/decay", "/viz/probe", "/viz/memory_map.html",
                "/encode", "/store", "/query", "/bundle",
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
            raise HTTPException(status_code=400, detail=f"invalid top_k: {exc}") from exc
        return app.state.viz.probe(text, top_k=top_k)

    @app.get("/viz/memory_map.html", response_class=FileResponse)
    def memory_map_html() -> FileResponse:
        if not MEMORY_MAP_HTML.exists():
            raise HTTPException(status_code=404, detail="memory_map.html not found")
        return FileResponse(MEMORY_MAP_HTML, media_type="text/html")

    # ── REST encoding / storage / retrieval ──────────────────────────────
    @app.post("/encode", response_model=EncodeResponse)
    def encode(req: EncodeRequest) -> EncodeResponse:
        hv = _encode(req.input, req.type)
        return EncodeResponse(vector=hv.data.tolist(), dims=len(hv))

    @app.post("/store", response_model=StoreResponse)
    def store(req: StoreRequest) -> StoreResponse:
        rest: RestState = app.state.rest
        hv = _encode(req.input, req.type)
        with rest.lock:
            entry_id = rest.episodic.store(hv, hv, req.label)
            # Also feed the example into semantic memory so /stats can report
            # online-learning iterations and so prototypes accumulate by label.
            rest.semantic.add(req.label, hv)
            size = len(rest.episodic)
        return StoreResponse(id=entry_id, label=req.label, dims=len(hv), episodic_size=size)

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
                probe = _encode(req.input, req.type)  # type: ignore[arg-type]

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
                decayed_similarity=decayed_map.get(r.entry_id) if decay_applied else None,
            )
            for r in raw_hits
        ]
        return QueryResponse(results=hits, top_k=req.top_k, decay_applied=decay_applied)

    @app.post("/bundle", response_model=BundleResponse)
    def bundle(req: BundleRequest) -> BundleResponse:
        hvs = [_encode(item, req.type) for item in req.inputs]
        bundled = HyperVector.bundle_all(hvs)
        return BundleResponse(vector=bundled.data.tolist(), dims=len(bundled), n_inputs=len(hvs))

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
                payload.append({"text": d.text, "id": d.id} if d.id else {"text": d.text})
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

    # ── Memory graph export ──────────────────────────────────────────────
    @app.get("/export/graph")
    def export_graph(
        format: str = Query("json", description="Output format hint (json or gexf)"),
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Dict[str, Any]:
        """Export the live episodic + semantic memory as a graph (JSON).

        Always returns JSON over HTTP. Use /export/graph/gexf for XML output.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return graph.to_dict()

    @app.get("/export/graph/gexf")
    def export_graph_gexf(
        threshold: float = Query(0.3, ge=-1.0, le=1.0),
    ) -> Response:
        """Export the live episodic + semantic memory as GEXF 1.3 XML."""
        rest: RestState = app.state.rest
        with rest.lock:
            cfg = GraphExportConfig(similarity_threshold=threshold)
            exporter = MemoryGraphExporter(cfg)
            graph = exporter.export(rest.episodic, semantic=rest.semantic)
        return Response(content=graph.to_gexf(), media_type="application/xml")

    # ════════════════════════════════════════════════════════════════════
    #  Enriched memory endpoints — temporal validity + salience + provenance
    # ════════════════════════════════════════════════════════════════════

    @app.post("/memories/store", response_model=EnrichedStoreResponse)
    def memories_store(req: EnrichedStoreRequest) -> EnrichedStoreResponse:
        rest: RestState = app.state.rest
        hv = _encode(req.input, req.type)
        with rest.lock:
            eid = rest.enriched.store(
                hv, hv, req.label,
                source=req.source,
                importance=req.importance,
                valid_from=req.valid_from,
                valid_until=req.valid_until,
            )
            total = len(rest.enriched)
        return EnrichedStoreResponse(
            entry_id=eid,
            label=req.label,
            source=req.source,
            importance=req.importance,
            valid_from=(req.valid_from or datetime.now(timezone.utc)).isoformat(),
            valid_until=req.valid_until.isoformat() if req.valid_until else None,
            total_memories=total,
        )

    @app.post("/memories/query", response_model=EnrichedQueryResponse)
    def memories_query(req: EnrichedQueryRequest) -> EnrichedQueryResponse:
        rest: RestState = app.state.rest
        with rest.lock:
            if len(rest.enriched) == 0:
                return EnrichedQueryResponse(results=[], top_k=req.top_k, sort=req.sort)
            probe = _encode(req.input, req.type)
            try:
                results = rest.enriched.query(
                    probe,
                    top_k=req.top_k,
                    sort=req.sort,
                    source_filter=req.source_filter,
                    include_expired=req.include_expired,
                    min_similarity=req.min_similarity,
                    reinforce_hits=req.reinforce_hits,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return EnrichedQueryResponse(
            results=[r.to_dict() for r in results],
            top_k=req.top_k,
            sort=req.sort,
        )

    @app.get("/memories")
    def memories_list(
        sort: str = Query("salience", description="Sort: salience | recency"),
        source: Optional[str] = Query(None, description="Filter by source"),
        limit: int = Query(50, ge=1, le=1000),
        include_expired: bool = Query(False),
    ) -> Dict[str, Any]:
        rest: RestState = app.state.rest
        if sort not in ("salience", "recency"):
            raise HTTPException(status_code=400, detail="sort must be 'salience' or 'recency'")
        with rest.lock:
            items = rest.enriched.list_memories(
                sort=sort, source_filter=source,
                include_expired=include_expired, limit=limit,
            )
            total = len(rest.enriched)
        return {
            "items": items, "count": len(items), "total": total,
            "sort": sort, "source_filter": source, "include_expired": include_expired,
        }

    @app.post("/memories/expire")
    def memories_expire() -> Dict[str, Any]:
        """Drop all memories whose ``valid_until`` is in the past. Returns the
        list of dropped entry ids."""
        rest: RestState = app.state.rest
        with rest.lock:
            dropped = rest.enriched.expire_old()
            remaining = len(rest.enriched)
        return {"dropped_ids": dropped, "dropped_count": len(dropped),
                "remaining": remaining}

    @app.get("/memories/trust-weights")
    def memories_trust_weights() -> Dict[str, float]:
        rest: RestState = app.state.rest
        # return a copy of the live table so the caller can see what's in effect
        return dict(rest.enriched.trust_weights)

    # ── Sleep-phase consolidation ──────────────────────────────────────────
    @app.post("/consolidate", response_model=ConsolidateResponse)
    def consolidate_endpoint(req: ConsolidateRequest = ConsolidateRequest()) -> ConsolidateResponse:
        """Trigger a one-shot sleep-phase consolidation pass.

        Runs synchronously over the enriched store's episodic memory:
        finds clusters with pairwise cosine >= `similarity_threshold`
        (default 0.85), merges them into semantic prototypes, returns the
        structured `SleepReport`.
        """
        rest: RestState = app.state.rest
        with rest.lock:
            # The daemon owns its own lock too, but we hold the RestState lock
            # to serialize against /memories/store.
            if req.similarity_threshold is not None:
                rest.sleep._threshold = req.similarity_threshold
            try:
                report = rest.sleep.run_once()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ConsolidateResponse(**report.to_dict())

    return app


app = create_app()


def main() -> int:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
