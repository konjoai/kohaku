"""Tests for kohaku.bulk_ops — batch update / delete / export."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kohaku import (
    EnrichedMemoryStore,
    ProvenanceGraph,
    RelationshipStore,
    VersionStore,
    encode_text,
)
from kohaku.bulk_ops import (
    BatchDeleteReport,
    BatchUpdateReport,
    batch_delete_by_filter,
    batch_delete_by_ids,
    batch_export,
    batch_update,
)


def _store() -> tuple[EnrichedMemoryStore, VersionStore, ProvenanceGraph, RelationshipStore]:
    pg = ProvenanceGraph()
    vs = VersionStore()
    rs = RelationshipStore()
    store = EnrichedMemoryStore(capacity=30, provenance=pg, versions=vs)
    return store, vs, pg, rs


def _seed_three(s: EnrichedMemoryStore) -> tuple[int, int, int]:
    e1 = s.store(encode_text("alpha"), encode_text("alpha"), label="alpha",
                  source="user_input", importance=0.4)
    e2 = s.store(encode_text("beta"), encode_text("beta"), label="beta",
                  source="user_input", importance=0.6)
    e3 = s.store(encode_text("gamma"), encode_text("gamma"), label="gamma",
                  source="agent_inference", importance=0.5)
    return e1, e2, e3


# ── batch_update ────────────────────────────────────────────────────────────

def test_batch_update_applies_each_row() -> None:
    s, vs, _, _ = _store()
    e1, e2, _ = _seed_three(s)
    report = batch_update(s, vs, [
        {"memory_id": e1, "tags": ["a", "b"]},
        {"memory_id": e2, "importance": 0.9},
    ], editor="curator")
    assert isinstance(report, BatchUpdateReport)
    assert report.processed == 2
    assert report.updated == 2
    assert report.failed == 0
    assert s.get_metadata(e1).tags == {"a", "b"}
    assert s.get_metadata(e2).importance == pytest.approx(0.9)


def test_batch_update_reports_missing_memory_id() -> None:
    s, vs, _, _ = _store()
    report = batch_update(s, vs, [
        {"tags": ["x"]},  # no memory_id
        {"memory_id": "abc"},  # invalid
        {"memory_id": 99999, "tags": ["ghost"]},  # not found
    ])
    assert report.processed == 3
    assert report.updated == 0
    assert report.failed == 3
    msgs = " ".join(err["error"] for err in report.errors)
    assert "memory_id" in msgs
    assert "not found" in msgs


def test_batch_update_rejects_non_list() -> None:
    s, vs, _, _ = _store()
    with pytest.raises(TypeError):
        batch_update(s, vs, "not a list")  # type: ignore[arg-type]


def test_batch_update_invalid_importance_recorded() -> None:
    s, vs, _, _ = _store()
    e1, _, _ = _seed_three(s)
    report = batch_update(s, vs, [{"memory_id": e1, "importance": 9.9}])
    assert report.updated == 0
    assert report.failed == 1
    assert "importance" in report.errors[0]["error"]


def test_batch_update_does_not_abort_on_one_failure() -> None:
    s, vs, _, _ = _store()
    e1, e2, _ = _seed_three(s)
    report = batch_update(s, vs, [
        {"memory_id": e1, "tags": ["t1"]},
        {"memory_id": 99999, "tags": ["ghost"]},
        {"memory_id": e2, "tags": ["t2"]},
    ])
    assert report.updated == 2
    assert report.failed == 1
    assert s.get_metadata(e1).tags == {"t1"}
    assert s.get_metadata(e2).tags == {"t2"}


# ── batch_delete ────────────────────────────────────────────────────────────

def test_batch_delete_by_ids_drops_listed_entries() -> None:
    s, _, _, rs = _store()
    e1, e2, e3 = _seed_three(s)
    rs.record(e1, e2, "supports")  # touched by deletion of e1
    report = batch_delete_by_ids(s, [e1, e3], relationships=rs)
    assert isinstance(report, BatchDeleteReport)
    assert report.deleted == 2
    assert set(report.deleted_ids) == {e1, e3}
    live = {e.id for e in s.episodic.entries()}
    assert live == {e2}
    # relationship for e1 was cleaned up
    assert len(rs) == 0


def test_batch_delete_reports_unknown_ids() -> None:
    s, _, _, _ = _store()
    e1, _, _ = _seed_three(s)
    report = batch_delete_by_ids(s, [e1, 9999, "bogus"])
    assert report.deleted == 1
    assert report.failed == 2
    msgs = " ".join(err["error"] for err in report.errors)
    assert "not found" in msgs and "invalid id" in msgs


def test_batch_delete_by_filter_requires_at_least_one_filter() -> None:
    s, _, _, _ = _store()
    _seed_three(s)
    with pytest.raises(ValueError, match="at least one filter"):
        batch_delete_by_filter(s)


def test_batch_delete_by_filter_source() -> None:
    s, _, _, _ = _store()
    e1, _, e3 = _seed_three(s)
    report = batch_delete_by_filter(s, source="agent_inference")
    assert report.deleted == 1
    live = {e.id for e in s.episodic.entries()}
    assert e3 not in live and e1 in live


def test_batch_delete_by_filter_stale_days() -> None:
    s, _, _, _ = _store()
    now = datetime.now(timezone.utc)
    fresh = s.store(encode_text("fresh"), encode_text("fresh"),
                     label="fresh", valid_from=now)
    stale = s.store(encode_text("old"), encode_text("old"),
                     label="old", valid_from=now - timedelta(days=60))
    report = batch_delete_by_filter(s, stale_days=30, now=now)
    assert report.deleted == 1
    live = {e.id for e in s.episodic.entries()}
    assert stale not in live and fresh in live


def test_batch_delete_by_filter_max_importance() -> None:
    s, _, _, _ = _store()
    e1, e2, e3 = _seed_three(s)
    report = batch_delete_by_filter(s, max_importance=0.5)
    # e1 (0.4) and e3 (0.5) qualify; e2 (0.6) survives
    live = {e.id for e in s.episodic.entries()}
    assert live == {e2}
    assert report.deleted == 2


def test_batch_delete_by_filter_tags_any() -> None:
    s, _, _, _ = _store()
    e1 = s.store(encode_text("a"), encode_text("a"), label="a", tags=["x"])
    s.store(encode_text("b"), encode_text("b"), label="b", tags=["y"])
    report = batch_delete_by_filter(s, tags_any=["x"])
    live = {e.id for e in s.episodic.entries()}
    assert report.deleted == 1
    assert e1 not in live


# ── batch_export ────────────────────────────────────────────────────────────

def test_batch_export_subset_only() -> None:
    s, _, _, _ = _store()
    e1, _, e3 = _seed_three(s)
    bundle = batch_export(s, [e1, e3], fmt="json")
    payload = json.loads(bundle.payload)
    labels = {m["label"] for m in payload["memories"]}
    assert labels == {"alpha", "gamma"}
    assert bundle.memory_count == 2


def test_batch_export_supports_markdown_and_csv() -> None:
    s, _, _, _ = _store()
    e1, _, _ = _seed_three(s)
    md = batch_export(s, [e1], fmt="markdown")
    assert "# Kohaku memory export" in md.payload
    csv_bundle = batch_export(s, [e1], fmt="csv")
    assert "label" in csv_bundle.payload.splitlines()[0]


def test_batch_export_invalid_format() -> None:
    s, _, _, _ = _store()
    e1, _, _ = _seed_three(s)
    with pytest.raises(ValueError):
        batch_export(s, [e1], fmt="rtf")
