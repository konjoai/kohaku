"""Shared helpers, state containers, and the request-handler context.

Split out of ``api/main.py``. Holds:

  • module-level constants (paths + viz defaults),
  • ``_kmeans_cosine`` (read-only viz clustering),
  • ``VizState`` / ``RestState`` (the two live state objects),
  • ``_vec_input_to_hv`` / ``_encode`` (HDC input adapters),
  • ``Ctx`` — a small carrier passed to every router's ``register(app, ctx)``.

Behaviour is byte-for-byte identical to the original inline definitions.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from fastapi import HTTPException

from kohaku import (
    DecayConfig,
    EnrichedMemoryStore,
    EpisodicMemory,
    HDCRetriever,
    HyperVector,
    ItemMemory,
    ProvenanceGraph,
    RelationshipStore,
    SleepConsolidator,
    VersionStore,
    decay_weight,
    encode_text,
)
from kohaku import (
    EpisodeStore,
    RateLimit,
    WriteValidator,
)
from kohaku import SharedMemoryPool, TenantMemoryStore
from kohaku._pure import DIMS

from .models import InputType

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
SAMPLE_PATH = DEMO_DIR / "sample_memory.json"
MEMORY_MAP_HTML = DEMO_DIR / "memory_map.html"
LIVE_HTML = DEMO_DIR / "kohaku-live.html"

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
                raise FileNotFoundError(f"sample memory file not found: {sample_path}")
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
            self.concepts.append(
                {
                    "entry_id": eid,
                    "id": c["id"],
                    "label": c.get("label", c["id"]),
                    "phrase": phrase,
                    "cluster_label": c.get("cluster_label", ""),
                    "color": c.get("color"),
                }
            )

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
            nodes.append(
                {
                    "id": e.label,
                    "entry_id": e.id,
                    "label": meta.get("label", e.label),
                    "cluster": int(cluster_idx[i]) if i < len(cluster_idx) else 0,
                    "cluster_label": meta.get("cluster_label", ""),
                    "color": meta.get("color"),
                    "last_accessed": e.timestamp,
                    "age": age,
                    "decay_weight": round(float(w), 4),
                }
            )

        edges: List[Dict[str, Any]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = float(entries[i].key.cosine_similarity(entries[j].key))
                if sim >= threshold:
                    edges.append(
                        {
                            "source": entries[i].label,
                            "target": entries[j].label,
                            "similarity": round(sim, 4),
                        }
                    )

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
            concepts_out.append(
                {
                    "id": e.label,
                    "label": meta.get("label", e.label),
                    "color": meta.get("color"),
                    "last_accessed": int(e.timestamp),
                    "current_age": int(current_age),
                    "current_weight": round(float(decay_weight(current_age, cfg)), 5),
                    "curve": curve,
                }
            )

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
            ranked.append(
                {
                    "id": e.label,
                    "entry_id": e.id,
                    "similarity": round(float(e.key.cosine_similarity(q)), 4),
                }
            )
        ranked.sort(key=lambda r: r["similarity"], reverse=True)
        return {
            "query": text,
            "top_k": top_k,
            "matches": ranked[: max(0, int(top_k))],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  REST — write-able state
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
        # Provenance graph — SQLite-backed DAG of memory lineage. Attached
        # to the enriched store so every /memories/store call auto-records.
        self.provenance = ProvenanceGraph()
        # Version store — Phase 16. Every /memories/store records v1; every
        # PUT /memories/{id} appends v2, v3, …
        self.versions = VersionStore()
        # Relationship store — Phase 17. Independent of provenance (lineage):
        # this captures typed semantic edges asserted via /memories/{id}/relate.
        self.relationships = RelationshipStore()
        # Enriched store (v0.10.0): temporal validity + salience + provenance.
        # Lives alongside the plain `episodic` store so the legacy /store /query
        # endpoints stay unchanged; the new /memories/* endpoints use this.
        self.enriched = EnrichedMemoryStore(
            capacity=capacity,
            dims=dims,
            provenance=self.provenance,
            versions=self.versions,
        )
        # Sleep-phase consolidation daemon over the enriched store's episodic
        # memory. Started lazily by /consolidate when a background thread is
        # explicitly requested; manual run_once() runs synchronously.
        # The provenance graph is attached so multi-member clusters write a
        # `record_consolidation` lineage edge on every merge.
        self.sleep = SleepConsolidator(
            self.enriched.episodic,
            consolidation_interval_minutes=60.0,
            similarity_threshold=0.85,
            provenance=self.provenance,
        )
        # Phase 13 P2 stores.
        self.episodes = EpisodeStore(dims=dims, capacity=capacity)
        self.validator = WriteValidator(
            self.episodic,
            rate_limits={
                "agent_inference": RateLimit(max_stores=100, window_seconds=60.0)
            },
        )
        # Multi-agent stores (Phase 17 / v0.32.0 — previously only in server.py).
        self.pool = SharedMemoryPool(dimension=dims, default_capacity=capacity)
        self.tenants = TenantMemoryStore(dimension=dims, capacity=capacity)
        self.lock = threading.Lock()
        self.started_at = time.time()


# ──────────────────────────────────────────────────────────────────────────


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


@dataclass
class Ctx:
    """Carrier handed to every router's ``register(app, ctx)``.

    Holds the HDC input adapters the route handlers reference so handlers
    depend on ``ctx`` rather than module-level enclosing scope. State objects
    are reached through ``app.state.rest`` / ``app.state.viz`` exactly as before.
    """

    encode: Callable[[Union[str, List[float]], InputType], HyperVector]
    vec_input_to_hv: Callable[[List[float]], HyperVector]
