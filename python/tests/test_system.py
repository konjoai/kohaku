"""Tests for the unified system snapshot (B3): save_system / load_system."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from kohaku import (
    EnrichedMemoryStore,
    ProvenanceGraph,
    RelationshipStore,
    VersionStore,
    load_system,
    save_system,
)
from kohaku.attention import encode_text


def _store(dims=2048):
    s = EnrichedMemoryStore(capacity=50, dims=dims)
    for text, imp, src in [
        ("the cat sat on the mat", 0.4, "user_input"),
        ("dogs are loyal companions", 0.8, "web_search"),
        ("paris is the capital of france", 0.6, "tool_result"),
    ]:
        hv = encode_text(text, dims=dims)
        s.store(hv, hv, text, source=src, importance=imp, tags=["t-" + src])
    return s


def test_roundtrip_memory_and_metadata(tmp_path):
    s = _store()
    # reinforce one entry so we can prove counts survive
    first_id = s.episodic.entries()[0].id
    s.reinforce(first_id, delta=4)

    save_system(s, tmp_path)
    bundle = load_system(tmp_path)
    loaded = bundle.store

    assert len(loaded) == len(s)
    # Metadata preserved exactly.
    for e in s.episodic.entries():
        orig = s.get_metadata(e.id)
        new = loaded.get_metadata(e.id)
        assert new is not None
        assert new.source == orig.source
        assert new.importance == orig.importance
        assert new.reinforcement_count == orig.reinforcement_count
        assert new.tags == orig.tags


def test_recall_is_exact_after_roundtrip(tmp_path):
    s = _store()
    save_system(s, tmp_path)
    loaded = load_system(tmp_path).store
    q = encode_text("paris is the capital of france", dims=2048)
    a = s.query(q, top_k=3, reinforce_hits=False)
    b = loaded.query(q, top_k=3, reinforce_hits=False)
    assert [r.entry_id for r in a] == [r.entry_id for r in b]
    assert [round(r.similarity, 6) for r in a] == [round(r.similarity, 6) for r in b]


def test_manifest_contents(tmp_path):
    s = _store()
    save_system(s, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert manifest["dims"] == 2048
    assert manifest["num_memories"] == 3
    assert "memory.hkb" in manifest["components"]
    assert "metadata.json" in manifest["components"]


def test_roundtrip_with_provenance_versions_relationships(tmp_path):
    pg = ProvenanceGraph(":memory:")
    vs = VersionStore(":memory:")
    rel = RelationshipStore(":memory:")
    s = EnrichedMemoryStore(capacity=50, dims=2048, provenance=pg, versions=vs)

    ids = []
    for text in ["root fact", "derived fact"]:
        hv = encode_text(text, dims=2048)
        parents = ids[:1]  # second derives from first
        ids.append(s.store(hv, hv, text, parent_ids=parents))
    rel.record(ids[1], ids[0], "derived_from")

    save_system(s, tmp_path, provenance=pg, versions=vs, relationships=rel)

    # Side-store db files exist and are listed.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for name in ("provenance.db", "versions.db", "relationships.db"):
        assert (tmp_path / name).exists()
        assert name in manifest["components"]

    bundle = load_system(tmp_path)
    assert bundle.provenance is not None
    assert bundle.versions is not None
    assert bundle.relationships is not None
    # Provenance lineage survived (memory_id is stored as text).
    ancestors = bundle.provenance.get_ancestors(ids[1])
    assert str(ids[0]) in {n.memory_id for n in ancestors}
    # Versions survived (v1 snapshot per stored memory).
    assert bundle.versions.count(ids[0]) >= 1
    # Relationship survived.
    out = bundle.relationships.list_outgoing(ids[1])
    assert any(r.target_id == ids[0] and r.relation_type == "derived_from" for r in out)
    # The loaded store has the side stores wired for future writes.
    assert bundle.store.provenance is bundle.provenance
    assert bundle.store.versions is bundle.versions


def test_load_missing_manifest_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_system(tmp_path)


def test_save_uses_attached_side_stores_by_default(tmp_path):
    pg = ProvenanceGraph(":memory:")
    s = EnrichedMemoryStore(capacity=10, dims=1024, provenance=pg)
    hv = encode_text("attached default", dims=1024)
    s.store(hv, hv, "attached default")
    # No explicit provenance= passed; save should pick up store.provenance.
    save_system(s, tmp_path)
    assert (tmp_path / "provenance.db").exists()


def test_expired_metadata_survives_with_validity(tmp_path):
    s = EnrichedMemoryStore(capacity=10, dims=1024)
    now = datetime.now(timezone.utc)
    hv = encode_text("time-boxed", dims=1024)
    s.store(hv, hv, "time-boxed", valid_from=now, valid_until=now)
    save_system(s, tmp_path)
    loaded = load_system(tmp_path).store
    eid = loaded.episodic.entries()[0].id
    meta = loaded.get_metadata(eid)
    assert meta.valid_until is not None
