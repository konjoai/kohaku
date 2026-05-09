"""Kohaku Visualization API — exposes the live HDC episodic memory as JSON
suitable for a force-directed graph and Ebbinghaus decay-curve plots.

Endpoints
=========
GET  /                          — service descriptor
GET  /viz/graph                 — nodes + edges for the force-directed graph
GET  /viz/decay                 — per-concept time-decay curves
POST /viz/probe                 — query → ranked nearest neighbours
GET  /viz/memory_map.html       — serves the interactive viewer

The memory is populated from `demo/sample_memory.json` on startup. All numbers
are computed by the live `kohaku` library — no mocks, no shortcuts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure the in-repo kohaku package is importable when this module runs from a
# checkout (no `pip install` required for development / tests).
ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from kohaku import DecayConfig, EpisodicMemory, decay_weight, encode_text
from kohaku._pure import DIMS
from kohaku._pure import HyperVector as _PyHyperVector

DEMO_DIR = ROOT / "demo"
SAMPLE_PATH = DEMO_DIR / "sample_memory.json"
MEMORY_MAP_HTML = DEMO_DIR / "memory_map.html"

DEFAULT_THRESHOLD = 0.7
DEFAULT_K = 3
DEFAULT_HALF_LIFE = 10.0
DEFAULT_HORIZON = 60
DEFAULT_STEPS = 30


# ───────────────────────── clustering ──────────────────────────────────────

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


# ───────────────────────── state ───────────────────────────────────────────

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

    # ── /viz/graph ────────────────────────────────────────────────────────
    def graph(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        k: int = DEFAULT_K,
        half_life: float = DEFAULT_HALF_LIFE,
    ) -> Dict[str, Any]:
        entries = self.memory.entries()
        n = len(entries)
        hvs = [e.key for e in entries]
        clusters = _kmeans_cosine(hvs, k=k)

        cfg = DecayConfig(half_life=half_life)
        now = max(0, self.memory._timestamp - 1)

        nodes: List[Dict[str, Any]] = []
        for idx, e in enumerate(entries):
            age = max(0, now - e.timestamp)
            w = decay_weight(age, cfg)
            meta = self.concepts[idx] if idx < len(self.concepts) else {}
            nodes.append({
                "id": e.label,
                "entry_id": e.id,
                "label": meta.get("label", e.label),
                "phrase": meta.get("phrase", ""),
                "cluster": int(clusters[idx]) if clusters else 0,
                "cluster_label": meta.get("cluster_label", ""),
                "color": meta.get("color"),
                "last_accessed": int(e.timestamp),
                "age": int(age),
                "decay_weight": round(float(w), 4),
            })

        edges: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                s = float(hvs[i].cosine_similarity(hvs[j]))
                if s >= threshold:
                    edges.append({
                        "source": entries[i].label,
                        "target": entries[j].label,
                        "similarity": round(s, 4),
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "dims": DIMS,
            "threshold": threshold,
            "num_clusters": min(k, n) if n else 0,
            "half_life": half_life,
            "current_clock": int(self.memory._timestamp),
        }

    # ── /viz/decay ────────────────────────────────────────────────────────
    def decay_curves(
        self,
        *,
        half_life: float = DEFAULT_HALF_LIFE,
        horizon: int = DEFAULT_HORIZON,
        steps: int = DEFAULT_STEPS,
    ) -> Dict[str, Any]:
        cfg = DecayConfig(half_life=half_life)
        now = max(0, self.memory._timestamp - 1)

        ages = sorted({int(round(a)) for a in np.linspace(0, horizon, steps)})

        concepts_out: List[Dict[str, Any]] = []
        for idx, e in enumerate(self.memory.entries()):
            current_age = max(0, now - e.timestamp)
            curve = [
                {"age": a, "weight": round(decay_weight(a, cfg), 5)}
                for a in ages
            ]
            meta = self.concepts[idx] if idx < len(self.concepts) else {}
            concepts_out.append({
                "id": e.label,
                "label": meta.get("label", e.label),
                "color": meta.get("color"),
                "last_accessed": int(e.timestamp),
                "current_age": int(current_age),
                "current_weight": round(decay_weight(current_age, cfg), 5),
                "curve": curve,
            })

        return {
            "half_life": half_life,
            "horizon": horizon,
            "steps": len(ages),
            "current_clock": int(self.memory._timestamp),
            "concepts": concepts_out,
        }

    # ── /viz/probe ────────────────────────────────────────────────────────
    def probe(self, text: str, *, top_k: int = 5) -> Dict[str, Any]:
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


# ───────────────────────── app factory ──────────────────────────────────────

def create_app(state: Optional[VizState] = None) -> FastAPI:
    app = FastAPI(title="Kohaku Visualization API", version="0.7.0")
    app.state.viz = state or VizState()

    @app.get("/")
    def root() -> Dict[str, Any]:
        viz: VizState = app.state.viz
        return {
            "name": "kohaku-viz",
            "version": "0.7.0",
            "dims": DIMS,
            "num_concepts": len(viz.memory.entries()),
            "endpoints": ["/viz/graph", "/viz/decay", "/viz/probe", "/viz/memory_map.html"],
        }

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

    return app


app = create_app()


def main() -> int:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
