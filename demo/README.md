# Kohaku demos

One maintained demo, plus the two pages the servers actually serve. Everything
else has been archived to stop the UI sprawl (C2 in `ROADMAP.md`).

## Maintained

| File | What it is | How to run |
|------|-----------|------------|
| **`kohaku-live.html`** | The canonical, maintained demo — live in-browser HDC memory (ports the kohaku LCG path bit-exactly). | `GET /live` on the FastAPI app (`api/main.py`), or open directly. |

## Load-bearing (served by code — do not move)

| File | Served by |
|------|-----------|
| `memory_map.html` | `api/main.py` → `GET /viz/memory_map.html` (the d3/cosmos viewer over the viz API). |
| `index.html` | `demo/server.py` → `GET /` (standalone demo landing page). |

## Archived (`demo/archive/`)

Redundant iterations of the memory-map viewer, kept for reference but no longer
wired into any server or test:

- `memory_map_cosmic.html`
- `memory_map_cosmos.html`
- `memory_map_dashboard.html`
- `dashboard.html`

> Full collapse to a single page would require rewiring the viz API
> (`/viz/memory_map.html`) and `demo/server.py` onto `kohaku-live.html`; that's
> deferred so the existing endpoints and `api/test_viz.py` keep working.
