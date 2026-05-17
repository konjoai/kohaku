"""Tests for kohaku.provenance — SQLite-backed memory lineage DAG."""

from __future__ import annotations

from pathlib import Path

import pytest

from kohaku import EnrichedMemoryStore, encode_text
from kohaku.provenance import ProvenanceGraph, ProvenanceNode


def test_record_creates_node() -> None:
    pg = ProvenanceGraph()
    node = pg.record("m1", source_type="user_input")
    assert isinstance(node, ProvenanceNode)
    assert node.memory_id == "m1"
    assert node.parent_count == 0
    assert pg.has("m1") is True
    assert len(pg) == 1


def test_record_with_parents_creates_edges() -> None:
    pg = ProvenanceGraph()
    pg.record("a", source_type="user_input")
    pg.record("b", source_type="user_input")
    pg.record("c", parent_ids=["a", "b"], source_type="inference")
    assert pg.has("c")
    children = pg.get_descendants("a", max_depth=1)
    assert {n.memory_id for n in children} == {"c"}
    ancestors = pg.get_ancestors("c", max_depth=1)
    assert {n.memory_id for n in ancestors} == {"a", "b"}


def test_get_ancestors_respects_max_depth() -> None:
    pg = ProvenanceGraph()
    # Linear chain a → b → c → d
    pg.record("a", source_type="user_input")
    pg.record("b", parent_ids=["a"], source_type="inference")
    pg.record("c", parent_ids=["b"], source_type="inference")
    pg.record("d", parent_ids=["c"], source_type="inference")

    ancestors_1 = pg.get_ancestors("d", max_depth=1)
    assert {n.memory_id for n in ancestors_1} == {"c"}

    ancestors_3 = pg.get_ancestors("d", max_depth=3)
    assert {n.memory_id for n in ancestors_3} == {"a", "b", "c"}

    # Each node remembers its depth
    by_id = {n.memory_id: n.depth for n in ancestors_3}
    assert by_id == {"c": 1, "b": 2, "a": 3}


def test_get_descendants_branches() -> None:
    pg = ProvenanceGraph()
    pg.record("root", source_type="user_input")
    pg.record("c1", parent_ids=["root"], source_type="consolidation")
    pg.record("c2", parent_ids=["root"], source_type="consolidation")
    pg.record("gc", parent_ids=["c1"], source_type="inference")

    descendants_1 = pg.get_descendants("root", max_depth=1)
    assert {n.memory_id for n in descendants_1} == {"c1", "c2"}

    descendants_2 = pg.get_descendants("root", max_depth=2)
    assert {n.memory_id for n in descendants_2} == {"c1", "c2", "gc"}


def test_get_full_graph_returns_bidirectional_view() -> None:
    pg = ProvenanceGraph()
    pg.record("a", source_type="user_input")
    pg.record("b", source_type="user_input")
    pg.record("merged", parent_ids=["a", "b"], source_type="consolidation")
    pg.record("downstream", parent_ids=["merged"], source_type="inference")

    result = pg.get_full_graph("merged", max_depth=3)
    assert result.root_id == "merged"
    assert {n.memory_id for n in result.ancestors} == {"a", "b"}
    assert {n.memory_id for n in result.descendants} == {"downstream"}
    assert ("a", "merged") in result.edges
    assert ("b", "merged") in result.edges
    assert ("merged", "downstream") in result.edges
    assert {n.memory_id for n in result.nodes} == {"a", "b", "merged", "downstream"}


def test_self_parent_rejected() -> None:
    pg = ProvenanceGraph()
    with pytest.raises(ValueError, match="own parent"):
        pg.record("m1", parent_ids=["m1"], source_type="user_input")


def test_max_depth_validation() -> None:
    pg = ProvenanceGraph()
    pg.record("a", source_type="user_input")
    with pytest.raises(ValueError):
        pg.get_ancestors("a", max_depth=0)
    with pytest.raises(ValueError):
        pg.get_descendants("a", max_depth=0)


def test_unknown_memory_lookups() -> None:
    pg = ProvenanceGraph()
    assert pg.has("ghost") is False
    assert pg.get_ancestors("ghost", max_depth=3) == []
    assert pg.get_descendants("ghost", max_depth=3) == []


def test_record_consolidation_sets_source_type() -> None:
    pg = ProvenanceGraph()
    pg.record("a", source_type="user_input")
    pg.record("b", source_type="user_input")
    node = pg.record_consolidation("merged", source_ids=["a", "b"])
    assert node.source_type == "consolidation"
    assert node.parent_count == 2
    assert node.metadata.get("event") == "sleep_consolidation"


def test_sqlite_persistence_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "prov.sqlite"
    pg1 = ProvenanceGraph(db)
    pg1.record("m1", source_type="user_input", metadata={"label": "first"})
    pg1.record("m2", parent_ids=["m1"], source_type="inference")
    pg1.close()
    pg2 = ProvenanceGraph(db)
    try:
        assert pg2.has("m1") and pg2.has("m2")
        anc = pg2.get_ancestors("m2", max_depth=1)
        assert anc[0].memory_id == "m1"
        # metadata round-trips through SQLite as JSON
        assert anc[0].metadata == {"label": "first"}
    finally:
        pg2.close()


def test_enriched_store_auto_records_provenance() -> None:
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=10, provenance=pg)
    hv = encode_text("the cat sat on the mat")
    eid = store.store(hv, hv, label="cat", source="user_input")
    assert pg.has(eid)
    # The node carries the label in metadata
    full = pg.get_full_graph(eid)
    node = next(n for n in full.nodes if n.memory_id == str(eid))
    assert node.source_type == "user_input"
    assert node.metadata.get("label") == "cat"


def test_enriched_store_records_parent_lineage() -> None:
    pg = ProvenanceGraph()
    store = EnrichedMemoryStore(capacity=10, provenance=pg)
    h1 = encode_text("alpha")
    e1 = store.store(h1, h1, label="alpha", source="user_input")
    h2 = encode_text("derived")
    e2 = store.store(h2, h2, label="derived",
                     source="inference", parent_ids=[e1])
    full = pg.get_full_graph(e2)
    assert str(e1) in {n.memory_id for n in full.ancestors}
    assert (str(e1), str(e2)) in full.edges


def test_upsert_preserves_created_at() -> None:
    pg = ProvenanceGraph()
    node1 = pg.record("m1", source_type="user_input")
    node2 = pg.record("m1", source_type="inference",
                      metadata={"updated": True})
    # created_at is sticky across upserts
    assert node2.created_at == node1.created_at
    assert node2.source_type == "inference"


def test_delete_removes_row() -> None:
    pg = ProvenanceGraph()
    pg.record("a", source_type="user_input")
    pg.record("b", parent_ids=["a"], source_type="inference")
    assert pg.delete("b") is True
    assert pg.has("b") is False
    assert pg.delete("b") is False  # idempotent
