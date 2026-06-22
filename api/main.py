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

The route handlers and pydantic models live in dedicated modules to keep each
file within the project's file-size budget:

  • ``api/models.py``         — request/response models
  • ``api/_helpers.py``       — k-means, VizState/RestState, encode adapters, Ctx
  • ``api/routers/*.py``      — route groups, each exposing ``register(app, ctx)``

``create_app`` builds the two state objects, then calls each router's
``register`` in the exact original registration order — FastAPI matches routes
in registration order, so this ordering is load-bearing and must not change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure the in-repo kohaku package is importable when this module runs from a
# checkout (no `pip install` required for development / tests).
ROOT = Path(__file__).resolve().parent.parent
PY_PKG = ROOT / "python"
if str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))

from fastapi import FastAPI  # noqa: E402

from kohaku import __version__ as KOHAKU_VERSION  # noqa: E402
from kohaku._pure import DIMS  # noqa: E402

from ._helpers import (  # noqa: E402
    Ctx,
    RestState,
    VizState,
    _encode,
    _vec_input_to_hv,
)
from .routers import (  # noqa: E402
    agents,
    analogy,
    core,
    episodes,
    export,
    memories,
    memories_batch,
    memories_extra,
)

__all__ = ["DIMS", "VizState", "app", "create_app", "main"]


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

    # Carrier handed to every router. Route registration order below mirrors the
    # original single-file layout exactly — do not reorder these calls.
    ctx = Ctx(encode=_encode, vec_input_to_hv=_vec_input_to_hv)
    core.register(app, ctx)
    export.register(app, ctx)
    memories.register(app, ctx)
    memories_extra.register(app, ctx)
    memories_batch.register(app, ctx)
    episodes.register(app, ctx)
    analogy.register(app, ctx)
    agents.register(app, ctx)

    return app


app = create_app()


def main() -> int:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8001, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
