"""Tests for kohaku.relationships — typed semantic edges between memories."""

from __future__ import annotations

from pathlib import Path

import pytest

from kohaku.relationships import (
    KNOWN_RELATIONS,
    Relationship,
    RelationshipStore,
)


def test_empty_store_returns_nothing() -> None:
    rs = RelationshipStore()
    assert len(rs) == 0
    assert rs.list_related(1) == []
    assert rs.list_outgoing(1) == []
    assert rs.list_incoming(1) == []
    assert rs.counts_by_type() == {}


def test_record_returns_relationship() -> None:
    rs = RelationshipStore()
    r = rs.record(1, 2, "supports", metadata={"note": "evidence"})
    assert isinstance(r, Relationship)
    assert r.source_id == 1 and r.target_id == 2
    assert r.relation_type == "supports"
    assert r.metadata == {"note": "evidence"}
    assert len(rs) == 1


def test_self_relationship_rejected() -> None:
    rs = RelationshipStore()
    with pytest.raises(ValueError, match="must differ"):
        rs.record(1, 1, "supports")


def test_negative_id_rejected() -> None:
    rs = RelationshipStore()
    with pytest.raises(ValueError, match=">= 0"):
        rs.record(-1, 2, "supports")
    with pytest.raises(ValueError, match=">= 0"):
        rs.record(1, -2, "supports")


def test_empty_relation_type_rejected() -> None:
    rs = RelationshipStore()
    with pytest.raises(ValueError, match="non-empty"):
        rs.record(1, 2, "   ")


def test_known_relations_vocabulary() -> None:
    expected = {"supports", "contradicts", "extends", "derived_from", "references"}
    assert KNOWN_RELATIONS == expected


def test_upsert_replaces_metadata() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports", metadata={"note": "v1"})
    r2 = rs.record(1, 2, "supports", metadata={"note": "v2"})
    assert r2.metadata == {"note": "v2"}
    assert len(rs) == 1


def test_multiple_types_between_same_pair() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(1, 2, "extends")
    assert len(rs) == 2
    types = {r.relation_type for r in rs.list_outgoing(1)}
    assert types == {"supports", "extends"}


def test_directional_listings_separate_outgoing_and_incoming() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(3, 1, "contradicts")
    out = rs.list_outgoing(1)
    inc = rs.list_incoming(1)
    assert [r.target_id for r in out] == [2]
    assert [r.source_id for r in inc] == [3]


def test_list_related_dedupes_overlap() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(3, 1, "contradicts")
    rs.record(1, 2, "extends")  # same pair, distinct type
    rels = rs.list_related(1)
    triples = {(r.source_id, r.target_id, r.relation_type) for r in rels}
    assert triples == {
        (1, 2, "supports"),
        (1, 2, "extends"),
        (3, 1, "contradicts"),
    }


def test_filter_by_type_in_listings() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(1, 3, "contradicts")
    out = rs.list_outgoing(1, relation_type="contradicts")
    assert len(out) == 1
    assert out[0].target_id == 3


def test_delete_specific_triple() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(1, 2, "extends")
    deleted = rs.delete(1, 2, "supports")
    assert deleted == 1
    assert {r.relation_type for r in rs.list_outgoing(1)} == {"extends"}


def test_delete_without_type_removes_all_between_pair() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(1, 2, "extends")
    rs.record(1, 3, "contradicts")
    deleted = rs.delete(1, 2)
    assert deleted == 2
    assert {r.target_id for r in rs.list_outgoing(1)} == {3}


def test_delete_all_for_strips_both_sides() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(3, 1, "contradicts")
    rs.record(2, 4, "extends")  # untouched
    deleted = rs.delete_all_for(1)
    assert deleted == 2
    assert len(rs) == 1


def test_counts_by_type_aggregates() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(3, 4, "supports")
    rs.record(1, 3, "contradicts")
    assert rs.counts_by_type() == {"supports": 2, "contradicts": 1}


def test_list_by_type_orders_newest_first() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports", metadata={"i": 1})
    rs.record(3, 4, "supports", metadata={"i": 2})
    rels = rs.list_by_type("supports")
    assert len(rels) == 2
    # newest-first via created_at desc — i=2 came second
    assert rels[0].metadata.get("i") == 2


def test_sqlite_persistence_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "rel.sqlite"
    rs1 = RelationshipStore(db)
    rs1.record(1, 2, "supports", metadata={"note": "persist me"})
    rs1.close()
    rs2 = RelationshipStore(db)
    try:
        out = rs2.list_outgoing(1)
        assert len(out) == 1
        assert out[0].metadata == {"note": "persist me"}
    finally:
        rs2.close()


def test_clear_drops_everything() -> None:
    rs = RelationshipStore()
    rs.record(1, 2, "supports")
    rs.record(3, 4, "extends")
    rs.clear()
    assert len(rs) == 0
