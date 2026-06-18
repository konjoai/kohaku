"""Tests for kohaku.versions — version history + edit path."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kohaku import EnrichedMemoryStore, encode_text
from kohaku.versions import (
    MemoryVersion,
    UpdateResult,
    VersionStore,
    update_memory,
)


def _store_with_versions() -> tuple[EnrichedMemoryStore, VersionStore]:
    vs = VersionStore()
    return EnrichedMemoryStore(capacity=20, versions=vs), vs


def test_empty_store_has_no_versions() -> None:
    vs = VersionStore()
    assert vs.count(1) == 0
    assert vs.list_versions(1) == []
    assert vs.latest_version(1) is None
    assert vs.get_version(1, 1) is None


def test_record_returns_monotonic_versions() -> None:
    vs = VersionStore()
    v1 = vs.record(7, label="a", source="user_input", importance=0.5)
    v2 = vs.record(7, label="b", source="user_input", importance=0.6)
    v3 = vs.record(7, label="c", source="user_input", importance=0.7)
    assert (v1.version, v2.version, v3.version) == (1, 2, 3)
    assert isinstance(v1, MemoryVersion)
    assert vs.count(7) == 3


def test_record_tags_are_normalised() -> None:
    vs = VersionStore()
    v = vs.record(
        1,
        label="x",
        source="user_input",
        importance=0.5,
        tags=["  Work ", "URGENT", "work", ""],
    )
    # lowercase, dedup, sorted, empty dropped
    assert v.tags == ("urgent", "work")


def test_list_versions_is_ascending() -> None:
    vs = VersionStore()
    vs.record(1, label="v1", source="user_input", importance=0.5)
    vs.record(1, label="v2", source="user_input", importance=0.5)
    vs.record(1, label="v3", source="user_input", importance=0.5)
    versions = vs.list_versions(1)
    assert [v.version for v in versions] == [1, 2, 3]


def test_latest_version_returns_highest() -> None:
    vs = VersionStore()
    vs.record(2, label="old", source="user_input", importance=0.5)
    vs.record(2, label="new", source="user_input", importance=0.5)
    latest = vs.latest_version(2)
    assert latest is not None
    assert latest.version == 2
    assert latest.label == "new"


def test_negative_memory_id_rejected() -> None:
    vs = VersionStore()
    with pytest.raises(ValueError):
        vs.record(-1, label="x", source="user_input", importance=0.5)


def test_sqlite_persistence_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "versions.sqlite"
    vs1 = VersionStore(db)
    vs1.record(
        1,
        label="first",
        source="user_input",
        importance=0.5,
        tags=["t1"],
        editor="alice",
    )
    vs1.close()
    vs2 = VersionStore(db)
    try:
        loaded = vs2.list_versions(1)
        assert len(loaded) == 1
        assert loaded[0].label == "first"
        assert loaded[0].tags == ("t1",)
        assert loaded[0].editor == "alice"
    finally:
        vs2.close()


def test_delete_removes_all_versions_for_one_memory() -> None:
    vs = VersionStore()
    vs.record(1, label="a", source="user_input", importance=0.5)
    vs.record(1, label="b", source="user_input", importance=0.5)
    vs.record(2, label="c", source="user_input", importance=0.5)
    deleted = vs.delete(1)
    assert deleted == 2
    assert vs.list_versions(1) == []
    assert vs.count(2) == 1


def test_enriched_store_autosnapshots_v1() -> None:
    store, vs = _store_with_versions()
    hv = encode_text("the cat sat on the mat")
    eid = store.store(
        hv, hv, label="cat", source="user_input", importance=0.7, tags=["pet"]
    )
    versions = vs.list_versions(eid)
    assert len(versions) == 1
    assert versions[0].label == "cat"
    assert versions[0].importance == pytest.approx(0.7)
    assert versions[0].tags == ("pet",)
    assert versions[0].editor == "store"


def test_update_label_re_encodes_hv() -> None:
    store, vs = _store_with_versions()
    h = encode_text("user prefers green tea")
    eid = store.store(
        h, h, label="user prefers green tea", source="user_input", importance=0.5
    )
    result = update_memory(store, eid, vs, label="user prefers black coffee")
    assert isinstance(result, UpdateResult)
    assert result.hv_re_encoded is True
    assert "label" in result.changed_fields
    assert result.version == 2
    # The live entry's HV must now match the new label
    entry = next(e for e in store.episodic._entries if e.id == eid)
    expected = encode_text("user prefers black coffee")
    assert (entry.key.data == expected.data).all()
    assert entry.label == "user prefers black coffee"


def test_update_metadata_only_no_re_encode() -> None:
    store, vs = _store_with_versions()
    h = encode_text("ok")
    eid = store.store(h, h, label="ok", source="user_input", importance=0.5)
    result = update_memory(store, eid, vs, importance=0.9, tags=["new"])
    assert result.hv_re_encoded is False
    assert set(result.changed_fields) == {"importance", "tags"}
    assert store.get_metadata(eid).importance == pytest.approx(0.9)
    assert store.get_metadata(eid).tags == {"new"}


def test_update_unknown_memory_raises() -> None:
    store, vs = _store_with_versions()
    with pytest.raises(KeyError):
        update_memory(store, 9999, vs, label="ghost")


def test_update_validates_importance_range() -> None:
    store, vs = _store_with_versions()
    h = encode_text("ok")
    eid = store.store(h, h, label="ok")
    with pytest.raises(ValueError):
        update_memory(store, eid, vs, importance=1.5)


def test_update_can_clear_valid_until() -> None:
    store, vs = _store_with_versions()
    h = encode_text("ok")
    now = datetime.now(timezone.utc)
    eid = store.store(h, h, label="ok", valid_until=now + timedelta(days=1))
    result = update_memory(store, eid, vs, valid_until=None)
    assert "valid_until" in result.changed_fields
    assert store.get_metadata(eid).valid_until is None


def test_update_accepts_iso_valid_until() -> None:
    store, vs = _store_with_versions()
    h = encode_text("ok")
    eid = store.store(h, h, label="ok")
    iso = "2030-01-01T00:00:00Z"
    update_memory(store, eid, vs, valid_until=iso)
    meta = store.get_metadata(eid)
    assert meta.valid_until is not None
    assert meta.valid_until.year == 2030


def test_update_rejects_valid_until_before_valid_from() -> None:
    store, vs = _store_with_versions()
    h = encode_text("ok")
    now = datetime.now(timezone.utc)
    eid = store.store(h, h, label="ok", valid_from=now)
    with pytest.raises(ValueError):
        update_memory(store, eid, vs, valid_until=now - timedelta(days=1))


def test_no_change_still_writes_snapshot() -> None:
    """update_memory is also a 'commit' — even a no-op call snapshots state."""
    store, vs = _store_with_versions()
    h = encode_text("ok")
    eid = store.store(h, h, label="ok", importance=0.5)
    result = update_memory(store, eid, vs)
    assert result.version == 2
    assert result.changed_fields == ()


def test_version_to_dict_shape() -> None:
    vs = VersionStore()
    v = vs.record(
        1, label="x", source="user_input", importance=0.7, tags=["t"], editor="me"
    )
    d = v.to_dict()
    assert {
        "memory_id",
        "version",
        "label",
        "source",
        "importance",
        "tags",
        "edited_at",
        "editor",
    } <= d.keys()
    assert d["tags"] == ["t"]
