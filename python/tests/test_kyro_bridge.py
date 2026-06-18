"""Tests for the kyro ↔ kohaku HDC retrieval bridge."""

from __future__ import annotations

import pytest

from kohaku import HDCRetriever, RetrievedChunk


def test_init_defaults():
    r = HDCRetriever()
    assert len(r) == 0
    assert r.dims == 10_000


def test_init_rejects_bad_capacity():
    with pytest.raises(ValueError):
        HDCRetriever(capacity=0)


def test_init_rejects_bad_dims():
    with pytest.raises(ValueError):
        HDCRetriever(dims=0)


def test_ingest_strings_assigns_ids():
    r = HDCRetriever()
    ids = r.ingest(["hyperdimensional computing", "the ocean is blue"])
    assert ids == [1, 2]
    assert len(r) == 2


def test_ingest_dicts_preserves_id():
    r = HDCRetriever()
    r.ingest([{"text": "kohaku amber", "id": "doc-amber"}])
    out = r.retrieve("kohaku amber", top_k=1)
    assert out[0].doc_id == "doc-amber"
    assert out[0].text == "kohaku amber"


def test_ingest_empty_text_rejected():
    r = HDCRetriever()
    with pytest.raises(ValueError):
        r.ingest([{"text": "   "}])


def test_ingest_bad_type_rejected():
    r = HDCRetriever()
    with pytest.raises(TypeError):
        r.ingest([123])  # type: ignore[list-item]


def test_retrieve_self_match_is_one():
    r = HDCRetriever()
    r.ingest(["coffee is dark and hot", "the sea is wide", "stars dim slowly"])
    hits = r.retrieve("coffee is dark and hot", top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "coffee is dark and hot"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[0].decayed_similarity is None


def test_retrieve_orders_by_similarity():
    r = HDCRetriever()
    r.ingest(["coffee is dark and hot", "tea is leafy and warm", "ocean is wide"])
    hits = r.retrieve("coffee is dark and hot", top_k=3)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_with_decay_returns_decayed_score():
    r = HDCRetriever()
    for i in range(5):
        r.ingest([f"phrase number {i} carrots"])
    hits = r.retrieve("phrase number 0 carrots", top_k=5, half_life=1.0)
    for h in hits:
        assert h.decayed_similarity is not None
        assert abs(h.decayed_similarity) <= abs(h.similarity) + 1e-6


def test_retrieve_age_monotone_with_ingest_order():
    r = HDCRetriever()
    r.ingest(["oldest"])
    r.ingest(["middle"])
    r.ingest(["newest"])
    by_text = {h.text: h.age for h in r.retrieve("oldest middle newest", top_k=3)}
    assert by_text["oldest"] > by_text["middle"] > by_text["newest"]


def test_retrieve_empty_returns_empty():
    r = HDCRetriever()
    assert r.retrieve("anything", top_k=3) == []


def test_retrieve_rejects_bad_top_k():
    r = HDCRetriever()
    r.ingest(["x"])
    with pytest.raises(ValueError):
        r.retrieve("x", top_k=0)


def test_retrieve_rejects_blank_query():
    r = HDCRetriever()
    r.ingest(["x"])
    with pytest.raises(ValueError):
        r.retrieve("   ", top_k=1)


def test_clear_drops_all():
    r = HDCRetriever()
    r.ingest(["a", "b"])
    r.clear()
    assert len(r) == 0
    assert r.retrieve("a", top_k=1) == []


def test_chunk_to_dict_round_trip():
    chunk = RetrievedChunk(
        entry_id=1, doc_id="d", text="t", similarity=0.5, decayed_similarity=0.4, age=2
    )
    d = chunk.to_dict()
    assert d == {
        "entry_id": 1,
        "doc_id": "d",
        "text": "t",
        "similarity": 0.5,
        "decayed_similarity": 0.4,
        "age": 2,
    }
