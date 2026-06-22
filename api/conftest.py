"""Shared pytest fixtures for the kohaku REST API test suite.

Each test resets in-process state so the suite is order-independent.
The TestClient drives the real FastAPI app — no mocked HDC anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make the repo's in-tree `python/` package and the repo root importable so
# `import api.main` and `import kohaku` both resolve when pytest runs from `api/`.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PY_PKG = _ROOT / "python"
for _p in (str(_PY_PKG), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.main import app  # noqa: E402
from kohaku import EpisodicMemory, HDCRetriever, ItemMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Wipe in-process REST state before every test — full isolation."""
    from kohaku import EnrichedMemoryStore, SleepConsolidator
    from kohaku.analogy import AnalogicalMemory

    rest = app.state.rest
    rest.episodic = EpisodicMemory(capacity=rest.episodic._capacity)
    rest.analogy = AnalogicalMemory(dims=rest.dims)
    rest.semantic = ItemMemory(dims=rest.dims)
    rest.bridge = HDCRetriever(capacity=rest.episodic._capacity, dims=rest.dims)
    rest.enriched = EnrichedMemoryStore(
        capacity=rest.episodic._capacity, dims=rest.dims
    )
    rest.sleep = SleepConsolidator(
        rest.enriched.episodic,
        consolidation_interval_minutes=60.0,
        similarity_threshold=0.85,
    )
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
