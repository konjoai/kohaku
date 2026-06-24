"""Integration tests for the multi-agent pool and per-tenant isolated
namespaces. Fixtures live in ``conftest.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

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
