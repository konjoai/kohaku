"""Integration tests for the enriched ``/memories/*`` surface, consolidation,
Graphiti/Mem0 export, and forgetting_rate. Fixtures live in ``conftest.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

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
