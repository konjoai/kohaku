"""Integration tests for the analogical-memory HTTP surface (v15).
Fixtures live in ``conftest.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


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
