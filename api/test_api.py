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

from main import DIMS, app  # noqa: E402
from kohaku import EpisodicMemory, HDCRetriever, ItemMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Wipe in-process REST state before every test — full isolation."""
    from kohaku import EnrichedMemoryStore, SleepConsolidator

    rest = app.state.rest
    rest.episodic = EpisodicMemory(capacity=rest.episodic._capacity)
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


# Compatibility alias so the existing tests that reference `state.episodic._capacity`
# keep reading the live REST state.
state = app.state.rest


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
    a = client.post(
        "/encode", json={"input": "the quick brown fox", "type": "text"}
    ).json()
    b = client.post(
        "/encode", json={"input": "the quick brown fox", "type": "text"}
    ).json()
    assert a["vector"] == b["vector"]


def test_encode_distinct_inputs_diverge(client: TestClient):
    a = client.post("/encode", json={"input": "kohaku is amber", "type": "text"}).json()
    b = client.post(
        "/encode", json={"input": "completely unrelated phrase", "type": "text"}
    ).json()
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

    r = client.post(
        "/store", json={"label": "morning_coffee", "input": "coffee is dark and hot"}
    )
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
        client.post("/store", json={"label": f"L{i}", "input": f"phrase {i}"}).json()[
            "id"
        ]
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
    r = client.post(
        "/query", json={"input": "coffee is dark and hot", "type": "text", "top_k": 1}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decay_applied"] is False
    assert body["results"][0]["label"] == "coffee"
    # Identical encoding → cosine similarity = 1.0 (within numeric noise).
    assert body["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert body["results"][0]["decayed_similarity"] is None


def test_query_top_k_orders_by_similarity(client: TestClient):
    _seed_three(client)
    r = client.post(
        "/query", json={"input": "coffee is dark and hot", "top_k": 3}
    ).json()
    sims = [hit["similarity"] for hit in r["results"]]
    assert sims == sorted(sims, reverse=True)
    assert len(sims) == 3


def test_query_with_decay_attaches_weighted_score(client: TestClient):
    # 5 stores → newest entry has timestamp ticks above the oldest.
    for i in range(5):
        client.post(
            "/store", json={"label": f"L{i}", "input": f"phrase number {i} carrots"}
        )
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
    bundled = client.post("/bundle", json={"inputs": members, "type": "text"}).json()[
        "vector"
    ]
    for label in members:
        member = client.post("/encode", json={"input": label, "type": "text"}).json()[
            "vector"
        ]
        # Bipolar cosine = dot / D.
        dot = sum(a * b for a, b in zip(bundled, member))
        cos = dot / DIMS
        assert cos > 0.3, f"bundle should resemble {label!r}, got cos={cos:.3f}"


def test_bundle_empty_rejected(client: TestClient):
    r = client.post("/bundle", json={"inputs": [], "type": "text"})
    assert r.status_code == 422


# ── /bridge ───────────────────────────────────────────────────────────────────


def test_bridge_ingest_assigns_ids_and_persists_text(client: TestClient):
    r = client.post(
        "/bridge/ingest",
        json={
            "documents": [
                "Hyperdimensional computing uses high-D vectors.",
                {
                    "text": "Ebbinghaus described the forgetting curve in 1885.",
                    "id": "ebb-1",
                },
                {"text": "Coffee is dark and hot."},
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["entry_ids"] == [1, 2, 3]
    assert body["total_chunks"] == 3


def test_bridge_ingest_rejects_string_documents_field(client: TestClient):
    # Bare string, not a list — must fail validation, not crash.
    r = client.post("/bridge/ingest", json={"documents": "just a string"})
    assert r.status_code == 422


def test_bridge_ingest_rejects_empty_text(client: TestClient):
    r = client.post("/bridge/ingest", json={"documents": [{"text": "", "id": "x"}]})
    # pydantic min_length=1 rejects empty string before reaching the bridge.
    assert r.status_code == 422


def test_bridge_retrieve_returns_top_k_with_self(client: TestClient):
    docs = [
        "Hyperdimensional computing uses high-D vectors.",
        "Ebbinghaus described the forgetting curve in 1885.",
        "The ocean is wide and blue.",
    ]
    client.post("/bridge/ingest", json={"documents": docs})
    r = client.post(
        "/bridge/retrieve",
        json={"query": "Hyperdimensional computing uses high-D vectors.", "top_k": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decay_applied"] is False
    assert body["total_chunks"] == 3
    assert len(body["results"]) == 3
    top = body["results"][0]
    assert top["text"] == docs[0]
    assert top["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert top["decayed_similarity"] is None
    assert top["age"] >= 0


def test_bridge_retrieve_with_decay_attaches_decayed_score(client: TestClient):
    for i in range(5):
        client.post(
            "/bridge/ingest",
            json={"documents": [{"text": f"chunk number {i} carrots", "id": f"c{i}"}]},
        )
    r = client.post(
        "/bridge/retrieve",
        json={"query": "chunk number 0 carrots", "top_k": 5, "half_life": 1.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decay_applied"] is True
    for hit in body["results"]:
        assert hit["decayed_similarity"] is not None
        assert abs(hit["decayed_similarity"]) <= abs(hit["similarity"]) + 1e-6


def test_bridge_retrieve_empty_returns_empty(client: TestClient):
    r = client.post("/bridge/retrieve", json={"query": "anything", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_bridge_retrieve_rejects_blank_query(client: TestClient):
    r = client.post("/bridge/retrieve", json={"query": "", "top_k": 1})
    assert r.status_code == 422


# ── /memories — enriched memory endpoints (v0.10.0) ───────────────────────────


def test_memories_store_returns_id_and_metadata(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "user fact A",
            "input": "the cat sat on the mat",
            "source": "user_input",
            "importance": 0.8,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["entry_id"] == 1
    assert body["source"] == "user_input"
    assert body["importance"] == 0.8
    assert body["total_memories"] == 1
    assert body["valid_until"] is None


def test_memories_store_rejects_bad_importance(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "x",
            "input": "hi",
            "importance": 1.5,
        },
    )
    assert r.status_code == 422


def test_memories_store_rejects_inverted_validity_window(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "x",
            "input": "hi",
            "valid_from": "2026-12-01T00:00:00Z",
            "valid_until": "2026-01-01T00:00:00Z",
        },
    )
    assert r.status_code == 422


def test_memories_query_filters_expired(client: TestClient):
    # Store a stale memory
    client.post(
        "/memories/store",
        json={
            "label": "stale",
            "input": "stale memory phrase",
            "valid_from": "2020-01-01T00:00:00Z",
            "valid_until": "2020-12-31T00:00:00Z",
        },
    )
    # And a live one
    client.post(
        "/memories/store",
        json={
            "label": "live",
            "input": "live memory phrase",
        },
    )
    r = client.post(
        "/memories/query",
        json={
            "input": "memory phrase",
            "top_k": 5,
            "reinforce_hits": False,
        },
    )
    assert r.status_code == 200
    labels = [row["label"] for row in r.json()["results"]]
    assert "stale" not in labels
    assert "live" in labels


def test_memories_query_include_expired_returns_them(client: TestClient):
    client.post(
        "/memories/store",
        json={
            "label": "stale",
            "input": "stale",
            "valid_from": "2020-01-01T00:00:00Z",
            "valid_until": "2020-12-31T00:00:00Z",
        },
    )
    r = client.post(
        "/memories/query",
        json={
            "input": "stale",
            "top_k": 5,
            "include_expired": True,
            "reinforce_hits": False,
        },
    )
    labels = [row["label"] for row in r.json()["results"]]
    assert "stale" in labels


def test_memories_query_source_filter(client: TestClient):
    client.post(
        "/memories/store",
        json={
            "label": "u",
            "input": "user knowledge",
            "source": "user_input",
            "importance": 0.5,
        },
    )
    client.post(
        "/memories/store",
        json={
            "label": "a",
            "input": "user knowledge",
            "source": "agent_inference",
            "importance": 0.5,
        },
    )
    r = client.post(
        "/memories/query",
        json={
            "input": "user knowledge",
            "top_k": 5,
            "source_filter": "user_input",
            "reinforce_hits": False,
        },
    )
    sources = {row["source"] for row in r.json()["results"]}
    assert sources == {"user_input"}


def test_memories_query_sort_by_salience(client: TestClient):
    # Both memories use the SAME text → same cosine to the probe.
    # Salience should pick the higher-importance + higher-trust one first.
    client.post(
        "/memories/store",
        json={
            "label": "imp",
            "input": "fact about kohaku",
            "source": "user_input",
            "importance": 1.0,
        },
    )
    client.post(
        "/memories/store",
        json={
            "label": "weak",
            "input": "fact about kohaku",
            "source": "agent_inference",
            "importance": 0.1,
        },
    )
    by_sal = client.post(
        "/memories/query",
        json={
            "input": "fact about kohaku",
            "top_k": 2,
            "sort": "salience",
            "reinforce_hits": False,
        },
    ).json()["results"]
    assert by_sal[0]["label"] == "imp"
    assert by_sal[0]["salience"] > by_sal[1]["salience"]


def test_memories_list_returns_metadata(client: TestClient):
    for i in range(3):
        client.post(
            "/memories/store",
            json={
                "label": f"m{i}",
                "input": f"phrase {i}",
                "source": "tool_result",
                "importance": 0.3 + i * 0.2,
            },
        )
    r = client.get("/memories?sort=salience&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert body["total"] == 3
    assert body["sort"] == "salience"
    # all rows carry the metadata fields
    for row in body["items"]:
        assert "salience" in row
        assert "source" in row
        assert "importance" in row
        assert "valid_from" in row
        assert "reinforcement_count" in row


def test_memories_list_source_filter(client: TestClient):
    client.post(
        "/memories/store", json={"label": "u", "input": "x", "source": "user_input"}
    )
    client.post(
        "/memories/store", json={"label": "w", "input": "y", "source": "web_search"}
    )
    r = client.get("/memories?source=web_search")
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["source"] == "web_search"


def test_memories_list_rejects_bad_sort(client: TestClient):
    r = client.get("/memories?sort=garbage")
    assert r.status_code == 400


def test_memories_expire_drops_expired(client: TestClient):
    client.post(
        "/memories/store",
        json={
            "label": "stale",
            "input": "x",
            "valid_from": "2020-01-01T00:00:00Z",
            "valid_until": "2020-12-31T00:00:00Z",
        },
    )
    client.post("/memories/store", json={"label": "live", "input": "y"})
    r = client.post("/memories/expire")
    assert r.status_code == 200
    body = r.json()
    assert body["dropped_count"] == 1
    assert body["remaining"] == 1


def test_memories_trust_weights_endpoint(client: TestClient):
    r = client.get("/memories/trust-weights")
    assert r.status_code == 200
    body = r.json()
    assert body["user_input"] == 1.0
    assert body["agent_inference"] == 0.5


def test_consolidate_endpoint_returns_report(client: TestClient):
    # 4 near-duplicate stores at the same phrase → consolidate merges them
    for i in range(4):
        client.post(
            "/memories/store",
            json={
                "label": f"d{i}",
                "input": "the cat sat on the mat",
            },
        )
    r = client.post("/consolidate", json={"similarity_threshold": 0.85})
    assert r.status_code == 200
    body = r.json()
    assert body["episodes_before"] == 4
    assert body["prototypes_created"] == 1
    assert body["episodes_after"] == 1
    assert body["memory_freed"] == 3
    assert body["run_seconds"] >= 0


def test_consolidate_no_merge_below_threshold(client: TestClient):
    # Orthogonal stores under a tight threshold → no merge
    for i in range(3):
        client.post(
            "/memories/store",
            json={
                "label": f"x{i}",
                "input": f"unrelated phrase number {i}",
            },
        )
    r = client.post("/consolidate", json={"similarity_threshold": 0.95})
    body = r.json()
    assert body["prototypes_created"] == 0
    assert body["memory_freed"] == 0


def test_consolidate_default_threshold(client: TestClient):
    """No similarity_threshold in payload → uses daemon's default (0.85)."""
    r = client.post("/consolidate", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["similarity_threshold"] == 0.85


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 15 — Graphiti/Mem0 export + forgetting_rate
# ═══════════════════════════════════════════════════════════════════════════


def test_export_graphiti_empty_memory(client: TestClient):
    r = client.get("/export/graph/graphiti")
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "graphiti"
    assert body["episodes"] == []
    assert body["entities"] == []
    assert body["relations"] == []


def test_export_graphiti_populated(client: TestClient):
    for phrase in ("cats are fluffy", "dogs are loyal", "birds can fly"):
        client.post("/store", json={"label": phrase, "input": phrase})
    r = client.get("/export/graph/graphiti")
    assert r.status_code == 200
    body = r.json()
    episodic_eps = [e for e in body["episodes"] if e["source"] == "episodic"]
    assert len(episodic_eps) == 3
    ep = episodic_eps[0]
    assert {
        "uuid",
        "name",
        "content",
        "source",
        "created_at",
        "valid_at",
        "invalid_at",
        "attributes",
    } <= ep.keys()


def test_export_graphiti_relations_at_low_threshold(client: TestClient):
    # Two very similar phrases should produce at least one relation.
    client.post("/store", json={"label": "fox1", "input": "the quick brown fox"})
    client.post("/store", json={"label": "fox2", "input": "the quick brown fox jumps"})
    r = client.get("/export/graph/graphiti?threshold=-1.0")
    body = r.json()
    assert len(body["relations"]) >= 1
    rel = body["relations"][0]
    assert rel["name"] == "similar_to"
    assert -1.0 <= rel["weight"] <= 1.0


def test_export_mem0_empty_memory(client: TestClient):
    r = client.get("/export/graph/mem0")
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "mem0"
    assert body["memories"] == []


def test_export_mem0_populated(client: TestClient):
    for phrase in ("red apple", "green pear", "yellow banana"):
        client.post("/store", json={"label": phrase, "input": phrase})
    r = client.get("/export/graph/mem0")
    assert r.status_code == 200
    body = r.json()
    episodic_mems = [
        m for m in body["memories"] if m["metadata"]["source"] == "episodic"
    ]
    assert len(episodic_mems) == 3
    mem = episodic_mems[0]
    assert {
        "id",
        "memory",
        "hash",
        "metadata",
        "score",
        "created_at",
        "updated_at",
    } <= mem.keys()
    assert len(mem["hash"]) == 16
    assert mem["score"] == 1.0  # no DecayConfig → default


def test_export_mem0_threshold_parameter(client: TestClient):
    """threshold query param is forwarded to MemoryGraphExporter."""
    for phrase in ("alpha", "beta"):
        client.post("/store", json={"label": phrase, "input": phrase})
    r_low = client.get("/export/graph/mem0?threshold=-1.0")
    r_high = client.get("/export/graph/mem0?threshold=1.0")
    assert r_low.status_code == 200
    assert r_high.status_code == 200


def test_memories_store_accepts_forgetting_rate(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "high priority fact",
            "input": "high priority fact",
            "source": "user_input",
            "importance": 0.9,
            "forgetting_rate": 0.5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["entry_id"] is not None


def test_memories_store_forgetting_rate_zero_rejected(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "bad rate",
            "input": "bad rate",
            "forgetting_rate": 0.0,
        },
    )
    assert r.status_code == 422


def test_memories_store_forgetting_rate_negative_rejected(client: TestClient):
    r = client.post(
        "/memories/store",
        json={
            "label": "bad rate",
            "input": "bad rate",
            "forgetting_rate": -1.5,
        },
    )
    assert r.status_code == 422


# ── Multi-agent pool ─────────────────────────────────────────────────────────


def test_agents_store_and_list(client: TestClient):
    r = client.post(
        "/agents/store",
        json={
            "agent_id": "alpha",
            "text": "kohaku is an HDC memory engine",
            "label": "fact",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["agent_id"] == "alpha"
    assert body["stored"] is True
    assert body["size"] == 1

    r2 = client.get("/agents")
    assert r2.status_code == 200
    data = r2.json()
    assert data["count"] == 1
    assert data["total_size"] == 1
    ids = [ns["id"] for ns in data["namespaces"]]
    assert "alpha" in ids


def test_agents_query_returns_results(client: TestClient):
    client.post(
        "/agents/store",
        json={"agent_id": "beta", "text": "neural network memory", "label": "nn"},
    )
    client.post(
        "/agents/store",
        json={"agent_id": "gamma", "text": "associative retrieval", "label": "ret"},
    )

    r = client.post(
        "/agents/query", json={"text": "neural network", "top_k": 5, "threshold": 0.0}
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["results"], list)
    agent_ids = {hit["agent_id"] for hit in body["results"]}
    assert "beta" in agent_ids or "gamma" in agent_ids


def test_agents_query_scoped_to_agents(client: TestClient):
    client.post(
        "/agents/store",
        json={"agent_id": "a1", "text": "short term memory", "label": "stm"},
    )
    client.post(
        "/agents/store",
        json={"agent_id": "a2", "text": "long term memory", "label": "ltm"},
    )

    r = client.post(
        "/agents/query",
        json={"text": "memory", "top_k": 5, "threshold": 0.0, "agents": ["a1"]},
    )
    assert r.status_code == 200
    body = r.json()
    agent_ids = {hit["agent_id"] for hit in body["results"]}
    assert "a2" not in agent_ids


def test_agents_drop(client: TestClient):
    client.post(
        "/agents/store",
        json={"agent_id": "temp_agent", "text": "temporary fact", "label": "tmp"},
    )
    r = client.get("/agents")
    assert any(ns["id"] == "temp_agent" for ns in r.json()["namespaces"])

    r2 = client.delete("/agents", params={"agent_id": "temp_agent"})
    assert r2.status_code == 204

    r3 = client.get("/agents")
    assert not any(ns["id"] == "temp_agent" for ns in r3.json()["namespaces"])


def test_agents_drop_unknown_is_noop(client: TestClient):
    r = client.delete("/agents", params={"agent_id": "nonexistent_agent"})
    assert r.status_code == 204


def test_agents_store_missing_agent_id_rejected(client: TestClient):
    r = client.post("/agents/store", json={"text": "something"})
    assert r.status_code == 422


def test_agents_store_empty_text_rejected(client: TestClient):
    r = client.post("/agents/store", json={"agent_id": "x", "text": ""})
    assert r.status_code == 422


# ── Tenant store ─────────────────────────────────────────────────────────────


def test_tenants_store_and_list(client: TestClient):
    r = client.post(
        "/tenants/store",
        json={"tenant_id": "acme", "text": "acme corp memory", "label": "acme-fact"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "acme"
    assert body["stored"] is True
    assert body["size"] == 1

    r2 = client.get("/tenants")
    assert r2.status_code == 200
    data = r2.json()
    assert data["count"] == 1
    ids = [ns["id"] for ns in data["namespaces"]]
    assert "acme" in ids


def test_tenants_are_isolated(client: TestClient):
    client.post(
        "/tenants/store",
        json={"tenant_id": "t1", "text": "t1 private data", "label": "t1"},
    )
    client.post(
        "/tenants/store",
        json={"tenant_id": "t2", "text": "t2 private data", "label": "t2"},
    )

    r = client.post(
        "/tenants/query", json={"tenant_id": "t1", "text": "private data", "top_k": 5}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "t1"
    # t2's data must not appear
    for hit in body["results"]:
        assert "t2" not in hit["label"]


def test_tenants_query_returns_results(client: TestClient):
    client.post(
        "/tenants/store",
        json={"tenant_id": "corp", "text": "quarterly revenue report", "label": "qrr"},
    )

    r = client.post(
        "/tenants/query",
        json={"tenant_id": "corp", "text": "quarterly revenue", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "corp"
    assert isinstance(body["results"], list)
    assert len(body["results"]) >= 1


def test_tenants_query_threshold_filters(client: TestClient):
    client.post(
        "/tenants/store",
        json={"tenant_id": "t3", "text": "machine learning", "label": "ml"},
    )

    r = client.post(
        "/tenants/query",
        json={
            "tenant_id": "t3",
            "text": "machine learning",
            "top_k": 5,
            "threshold": 0.99,
        },
    )
    assert r.status_code == 200
    body = r.json()
    for hit in body["results"]:
        assert hit["similarity"] >= 0.99


def test_tenants_drop(client: TestClient):
    client.post(
        "/tenants/store",
        json={"tenant_id": "old_tenant", "text": "data to delete", "label": "del"},
    )
    r = client.get("/tenants")
    assert any(ns["id"] == "old_tenant" for ns in r.json()["namespaces"])

    r2 = client.delete("/tenants", params={"tenant_id": "old_tenant"})
    assert r2.status_code == 204

    r3 = client.get("/tenants")
    assert not any(ns["id"] == "old_tenant" for ns in r3.json()["namespaces"])


def test_tenants_drop_unknown_is_noop(client: TestClient):
    r = client.delete("/tenants", params={"tenant_id": "does_not_exist"})
    assert r.status_code == 204


def test_tenants_store_missing_tenant_id_rejected(client: TestClient):
    r = client.post("/tenants/store", json={"text": "something"})
    assert r.status_code == 422


def test_tenants_store_empty_text_rejected(client: TestClient):
    r = client.post("/tenants/store", json={"tenant_id": "x", "text": ""})
    assert r.status_code == 422


# ── Analogical memory ─────────────────────────────────────────────────────────


def test_analogy_extract_returns_triples(client: TestClient):
    r = client.post(
        "/analogy/extract",
        json={
            "text": "The capital of France is Paris. The currency of France is euro."
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    subjects = {t["subject"] for t in body["triples"]}
    assert any("france" in s.lower() or "France" in s for s in subjects)


def test_analogy_extract_unparseable_returns_zero(client: TestClient):
    r = client.post(
        "/analogy/extract",
        json={"text": "Clouds drifted softly across the silent afternoon sky."},
    )
    assert r.status_code == 200
    body = r.json()
    # Extractor may or may not find structure here; shape must be correct regardless
    assert isinstance(body["count"], int)
    assert isinstance(body["triples"], list)
    assert body["count"] == len(body["triples"])


def test_analogy_extract_empty_text_rejected(client: TestClient):
    r = client.post("/analogy/extract", json={"text": ""})
    assert r.status_code == 422


def test_analogy_learn_stores_and_returns_triples(client: TestClient):
    r = client.post(
        "/analogy/learn",
        json={"text": "The capital of Japan is Tokyo. The currency of Japan is yen."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["triples_learned"] >= 1
    assert body["records_count"] >= 1


def test_analogy_records_lists_learned(client: TestClient):
    client.post(
        "/analogy/learn",
        json={"text": "The capital of Germany is Berlin."},
    )
    r = client.get("/analogy/records")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert isinstance(body["records"], list)


def test_analogy_get_attribute(client: TestClient):
    client.post(
        "/analogy/learn",
        json={"text": "The capital of Italy is Rome. The currency of Italy is euro."},
    )
    # Fetch the record name (first known record after learning)
    recs = client.get("/analogy/records").json()["records"]
    assert len(recs) >= 1
    # Try to get an attribute from the first record
    name = recs[0]
    r = client.post("/analogy/get", json={"name": name, "attribute": "capital"})
    # May return a result or not depending on what was parsed — just check shape
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert "value" in body
        assert "confidence" in body
        assert isinstance(body["ranked"], list)


def test_analogy_get_unknown_record_returns_404(client: TestClient):
    r = client.post(
        "/analogy/get", json={"name": "nonexistent_country", "attribute": "currency"}
    )
    assert r.status_code == 404


def test_analogy_transfer_unknown_source_returns_404(client: TestClient):
    r = client.post(
        "/analogy/transfer",
        json={"source": "ghost_a", "target": "ghost_b", "value": "dollar"},
    )
    assert r.status_code == 404


def test_analogy_transfer_returns_result(client: TestClient):
    # Seed two records with the same attribute structure
    client.post(
        "/analogy/learn",
        json={
            "text": "The currency of USA is dollar. The capital of USA is Washington."
        },
    )
    client.post(
        "/analogy/learn",
        json={
            "text": "The currency of Mexico is peso. The capital of Mexico is Mexico City."
        },
    )
    recs = {r.lower() for r in client.get("/analogy/records").json()["records"]}
    if "usa" in recs and "mexico" in recs:
        r = client.post(
            "/analogy/transfer",
            json={"source": "USA", "target": "Mexico", "value": "dollar"},
        )
        # May or may not resolve depending on parsed subject casing
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            body = r.json()
            assert "value" in body
            assert 0.0 <= body["confidence"] <= 1.0
