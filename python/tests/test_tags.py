"""Tests for the tag layer on :class:`EnrichedMemoryStore`."""

from __future__ import annotations

from kohaku import EnrichedMemoryStore, encode_text


def _store() -> EnrichedMemoryStore:
    return EnrichedMemoryStore(capacity=20)


def test_store_accepts_tags_and_normalises_them() -> None:
    s = _store()
    hv = encode_text("first memory")
    eid = s.store(hv, hv, label="first", tags=[" Work ", "URGENT", "work"])
    # whitespace stripped, lowercased, deduped
    assert s.get_tags(eid) == {"work", "urgent"}


def test_empty_tags_dropped_on_store() -> None:
    s = _store()
    hv = encode_text("m")
    eid = s.store(hv, hv, label="m", tags=["", "   ", "real"])
    assert s.get_tags(eid) == {"real"}


def test_add_tags_appends() -> None:
    s = _store()
    hv = encode_text("m")
    eid = s.store(hv, hv, label="m", tags=["alpha"])
    out = s.add_tags(eid, ["beta", "gamma", "alpha"])  # alpha already present
    assert out == {"alpha", "beta", "gamma"}


def test_remove_tags_drops_known_and_ignores_unknown() -> None:
    s = _store()
    hv = encode_text("m")
    eid = s.store(hv, hv, label="m", tags=["alpha", "beta"])
    out = s.remove_tags(eid, ["alpha", "ghost"])
    assert out == {"beta"}


def test_tag_apis_return_none_for_unknown_entry() -> None:
    s = _store()
    assert s.get_tags(9999) is None
    assert s.add_tags(9999, ["x"]) is None
    assert s.remove_tags(9999, ["x"]) is None


def test_all_tags_returns_counts() -> None:
    s = _store()
    h = encode_text("m")
    s.store(h, h, label="a", tags=["work", "urgent"])
    s.store(h, h, label="b", tags=["work"])
    s.store(h, h, label="c", tags=["personal"])
    counts = s.all_tags()
    assert counts == {"work": 2, "urgent": 1, "personal": 1}


def test_list_memories_tags_any_filter() -> None:
    s = _store()
    h = encode_text("m")
    s.store(h, h, label="a", tags=["work"])
    s.store(h, h, label="b", tags=["personal"])
    s.store(h, h, label="c", tags=["work", "urgent"])
    items = s.list_memories(tags_any=["work"])
    assert {it["label"] for it in items} == {"a", "c"}


def test_list_memories_tags_all_filter() -> None:
    s = _store()
    h = encode_text("m")
    s.store(h, h, label="a", tags=["work"])
    s.store(h, h, label="b", tags=["work", "urgent"])
    s.store(h, h, label="c", tags=["work", "urgent", "today"])
    items = s.list_memories(tags_all=["work", "urgent"])
    assert {it["label"] for it in items} == {"b", "c"}


def test_query_filters_by_tags() -> None:
    s = _store()
    h1 = encode_text("the cat sat on the mat")
    h2 = encode_text("a cat purred on the rug")
    s.store(h1, h1, label="cat-1", tags=["pet", "indoor"])
    s.store(h2, h2, label="cat-2", tags=["wild"])
    probe = encode_text("cat on mat")
    res = s.query(probe, top_k=5, tags_any=["pet"])
    assert len(res) == 1
    assert res[0].label == "cat-1"
    assert "pet" in res[0].tags


def test_list_memories_emits_tags_field() -> None:
    s = _store()
    h = encode_text("m")
    s.store(h, h, label="m", tags=["a", "b"])
    items = s.list_memories()
    assert items[0]["tags"] == ["a", "b"]


def test_normalisation_truncates_long_tags() -> None:
    s = _store()
    h = encode_text("m")
    long_tag = "x" * 200
    eid = s.store(h, h, label="m", tags=[long_tag])
    tags = s.get_tags(eid)
    assert tags is not None
    assert all(len(t) <= 64 for t in tags)
