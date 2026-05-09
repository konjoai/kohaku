"""Integration tests for the kohaku REST API.

Each test resets in-process state so the suite is order-independent.
The TestClient drives the real FastAPI app — no mocked HDC anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make `api/` importable when pytest is run from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from main import DIMS, app, state  # noqa: E402
from kohaku import EpisodicMemory, ItemMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Wipe global memory before every test — full isolation."""
    state.episodic = EpisodicMemory(capacity=state.episodic._capacity)
    state.semantic = ItemMemory(dims=state.dims)
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] in {"rust", "python"}


# ── /encode ───────────────────────────────────────────────────────────────────

def test_encode_text_returns_bipolar_vector(client: TestClient):
    r = client.post("/encode", json={"input": "hello world", "type": "text"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dims"] == DIMS
    assert len(body["vector"]) == DIMS
    assert set(body["vector"]) <= {-1, 1}


def test_encode_text_deterministic(client: TestClient):
    a = client.post("/encode", json={"input": "the quick brown fox", "type": "text"}).json()
    b = client.post("/encode", json={"input": "the quick brown fox", "type": "text"}).json()
    assert a["vector"] == b["vector"]


def test_encode_distinct_inputs_diverge(client: TestClient):
    a = client.post("/encode", json={"input": "kohaku is amber", "type": "text"}).json()
    b = client.post("/encode", json={"input": "completely unrelated phrase", "type": "text"}).json()
    diff = sum(1 for x, y in zip(a["vector"], b["vector"]) if x != y)
    # Random near-orthogonal HVs differ in ~half the bits — expect well over 25%.
    assert diff > DIMS * 0.25


def test_encode_vector_passthrough_binarizes(client: TestClient):
    raw = [0.1] * DIMS
    raw[0] = -2.0
    r = client.post("/encode", json={"input": raw, "type": "vector"})
    assert r.status_code == 200, r.text
    out = r.json()["vector"]
    assert out[0] == -1
    assert out[1] == 1
    assert set(out) <= {-1, 1}


def test_encode_vector_wrong_length_rejected(client: TestClient):
    r = client.post("/encode", json={"input": [0.0, 1.0, -1.0], "type": "vector"})
    assert r.status_code == 422


# ── /store + /stats ───────────────────────────────────────────────────────────

def test_store_returns_id_and_grows_memory(client: TestClient):
    before = client.get("/stats").json()
    assert before["episodic_size"] == 0
    assert before["learning_iterations"] == 0

    r = client.post("/store", json={"label": "morning_coffee", "input": "coffee is dark and hot"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 1
    assert body["label"] == "morning_coffee"
    assert body["episodic_size"] == 1
    assert body["dims"] == DIMS

    after = client.get("/stats").json()
    assert after["episodic_size"] == 1
    assert after["semantic_concepts"] == 1
    assert after["learning_iterations"] == 1
    assert after["dims"] == DIMS
    assert after["episodic_capacity"] == state.episodic._capacity


def test_store_assigns_increasing_ids(client: TestClient):
    ids = [
        client.post("/store", json={"label": f"L{i}", "input": f"phrase {i}"}).json()["id"]
        for i in range(3)
    ]
    assert ids == [1, 2, 3]


# ── /query ────────────────────────────────────────────────────────────────────

def _seed_three(client: TestClient) -> None:
    client.post("/store", json={"label": "coffee", "input": "coffee is dark and hot"})
    client.post("/store", json={"label": "tea", "input": "tea is leafy and warm"})
    client.post("/store", json={"label": "ocean", "input": "the sea is wide and blue"})


def test_query_text_recovers_self(client: TestClient):
    _seed_three(client)
    r = client.post("/query", json={"input": "coffee is dark and hot", "type": "text", "top_k": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decay_applied"] is False
    assert body["results"][0]["label"] == "coffee"
    # Identical encoding → cosine similarity = 1.0 (within numeric noise).
    assert body["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert body["results"][0]["decayed_similarity"] is None


def test_query_top_k_orders_by_similarity(client: TestClient):
    _seed_three(client)
    r = client.post("/query", json={"input": "coffee is dark and hot", "top_k": 3}).json()
    sims = [hit["similarity"] for hit in r["results"]]
    assert sims == sorted(sims, reverse=True)
    assert len(sims) == 3


def test_query_with_decay_attaches_weighted_score(client: TestClient):
    # 5 stores → newest entry has timestamp ticks above the oldest.
    for i in range(5):
        client.post("/store", json={"label": f"L{i}", "input": f"phrase number {i} carrots"})
    r = client.post(
        "/query",
        json={"input": "phrase number 0 carrots", "top_k": 5, "half_life": 1.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decay_applied"] is True
    for hit in body["results"]:
        assert hit["decayed_similarity"] is not None
        # Decay weight ∈ (0, 1] so |decayed| ≤ |raw| (within numeric noise).
        assert abs(hit["decayed_similarity"]) <= abs(hit["similarity"]) + 1e-6


def test_query_by_label_uses_semantic_prototype(client: TestClient):
    _seed_three(client)
    r = client.post("/query", json={"label": "coffee", "top_k": 1}).json()
    assert r["results"][0]["label"] == "coffee"
    assert r["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_query_unknown_label_404(client: TestClient):
    _seed_three(client)
    r = client.post("/query", json={"label": "does-not-exist", "top_k": 1})
    assert r.status_code == 404


def test_query_requires_exactly_one_probe(client: TestClient):
    r = client.post("/query", json={"input": "x", "label": "y", "top_k": 1})
    assert r.status_code == 422
    r = client.post("/query", json={"top_k": 1})
    assert r.status_code == 422


def test_query_empty_memory_returns_empty(client: TestClient):
    r = client.post("/query", json={"input": "anything", "top_k": 5}).json()
    assert r["results"] == []


# ── /bundle ───────────────────────────────────────────────────────────────────

def test_bundle_returns_bipolar_vector(client: TestClient):
    r = client.post(
        "/bundle",
        json={"inputs": ["alpha", "beta", "gamma"], "type": "text"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dims"] == DIMS
    assert body["n_inputs"] == 3
    assert set(body["vector"]) <= {-1, 1}


def test_bundle_similar_to_each_member(client: TestClient):
    """The bundle of N near-orthogonal vectors should still resemble each member
    above chance (cos > 1/√N - epsilon)."""
    members = ["alpha", "beta", "gamma"]
    bundled = client.post("/bundle", json={"inputs": members, "type": "text"}).json()["vector"]
    for label in members:
        member = client.post("/encode", json={"input": label, "type": "text"}).json()["vector"]
        # Bipolar cosine = dot / D.
        dot = sum(a * b for a, b in zip(bundled, member))
        cos = dot / DIMS
        assert cos > 0.3, f"bundle should resemble {label!r}, got cos={cos:.3f}"


def test_bundle_empty_rejected(client: TestClient):
    r = client.post("/bundle", json={"inputs": [], "type": "text"})
    assert r.status_code == 422
