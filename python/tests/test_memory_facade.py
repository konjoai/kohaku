"""Tests for the ``Memory`` facade — the string-in/string-out front door."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from kohaku import Memory, MemoryHit


def test_store_returns_id_and_increments_len():
    mem = Memory()
    assert len(mem) == 0
    eid = mem.store("User prefers Italian wine")
    assert isinstance(eid, int)
    assert len(mem) == 1


def test_query_returns_memory_hits():
    mem = Memory()
    mem.store("User prefers Italian wine")
    hits = mem.query("Italian wine", reinforce=False)
    assert hits
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].text == "User prefers Italian wine"
    assert 0.0 <= hits[0].similarity <= 1.0


def test_readme_example_runs():
    # The exact shape advertised in the README must work end to end.
    mem = Memory()
    mem.store("User prefers Italian wine")
    mem.store("User is allergic to shellfish", importance=0.9, tags=["health"])
    hits = mem.query("What does the user like to drink?")
    assert all(isinstance(h, MemoryHit) for h in hits)
    assert any("wine" in h.text for h in hits)


def test_self_match_is_strong():
    mem = Memory()
    mem.store("the quick brown fox jumps over the lazy dog")
    hit = mem.query("the quick brown fox jumps over the lazy dog", reinforce=False)[0]
    assert hit.similarity > 0.95


def test_empty_text_rejected():
    mem = Memory()
    with pytest.raises(ValueError):
        mem.store("   ")
    with pytest.raises(ValueError):
        mem.query("")


def test_tag_filtering():
    mem = Memory()
    mem.store("Paris is the capital of France", tags=["geo"])
    mem.store("Photosynthesis converts light to energy", tags=["science"])
    geo = mem.query("capital", tags_any=["geo"], reinforce=False)
    assert geo and all("geo" in h.tags for h in geo)
    none = mem.query("capital", tags_all=["geo", "science"], reinforce=False)
    assert none == []


def test_source_filter():
    mem = Memory()
    mem.store("fact from the web", source="web_search")
    mem.store("fact from a tool", source="tool_result")
    web = mem.query("fact", source="web_search", reinforce=False)
    assert web and all(h.source == "web_search" for h in web)


def test_salience_sort_uses_salience_as_score():
    mem = Memory()
    mem.store("low importance note", importance=0.1)
    mem.store("high importance note", importance=1.0)
    hits = mem.query("note", sort="salience", reinforce=False)
    assert hits[0].score == hits[0].salience
    # The high-importance memory should outrank the low-importance one.
    assert hits[0].text == "high importance note"


def test_reinforce_raises_salience():
    mem = Memory()
    eid = mem.store("reinforce me", importance=0.5)
    before = mem.query("reinforce", sort="salience", reinforce=False)[0].salience
    mem.reinforce(eid)
    mem.reinforce(eid)
    after = mem.query("reinforce", sort="salience", reinforce=False)[0].salience
    assert after > before


def test_query_reinforces_by_default():
    mem = Memory()
    mem.store("auto reinforced")
    s0 = mem.query("auto", reinforce=False)[0].salience
    mem.query("auto")  # default reinforce=True bumps the hit
    s1 = mem.query("auto", reinforce=False)[0].salience
    assert s1 > s0


def test_expire_drops_past_validity():
    mem = Memory()
    now = datetime.now(timezone.utc)
    mem.store("ephemeral", valid_until=now + timedelta(hours=1))
    mem.store("permanent")
    # Advance "now" past the ephemeral memory's validity window.
    dropped = mem.expire(now=now + timedelta(days=1))
    assert len(dropped) == 1
    assert len(mem) == 1
    assert mem.query("permanent", reinforce=False)[0].text == "permanent"


def test_clear():
    mem = Memory()
    mem.store("a")
    mem.store("b")
    mem.clear()
    assert len(mem) == 0


def test_capacity_eviction():
    mem = Memory(capacity=2)
    mem.store("one")
    mem.store("two")
    mem.store("three")
    assert len(mem) == 2


def test_save_load_roundtrip_exact(tmp_path):
    mem = Memory(capacity=50, half_life_days=10.0)
    mem.store("User prefers Italian wine", importance=0.7, tags=["pref"])
    mem.store("Paris is the capital of France", source="web_search")
    eid = mem.store("reinforced fact")
    mem.reinforce(eid, delta=3)

    path = str(tmp_path / "mem.json")
    mem.save(path)
    restored = Memory.load(path)

    assert len(restored) == len(mem)
    # Similarity is reproduced exactly (deterministic re-encode).
    orig = mem.query("Italian wine", reinforce=False)[0]
    again = restored.query("Italian wine", reinforce=False)[0]
    assert again.text == orig.text
    assert again.similarity == pytest.approx(orig.similarity)
    assert again.tags == orig.tags
    # Reinforcement count survives the round-trip.
    rf = restored.query("reinforced fact", sort="salience", reinforce=False)[0]
    assert rf.salience > 0


def test_store_escape_hatch_exposes_enriched_store():
    from kohaku import EnrichedMemoryStore

    mem = Memory()
    mem.store("hello world")
    assert isinstance(mem.store_, EnrichedMemoryStore)
    assert len(mem.store_) == 1


def test_memory_hit_to_dict():
    mem = Memory()
    mem.store("serialise me")
    hit = mem.query("serialise", reinforce=False)[0]
    d = hit.to_dict()
    assert set(d) >= {"id", "text", "score", "similarity", "salience", "source", "tags"}


# ── relational reasoning via the facade (Track D2) ───────────────────────────


def _countries_facade() -> Memory:
    mem = Memory()
    mem.add_record("USA", {"currency": "dollar", "capital": "washington"})
    mem.add_record("Mexico", {"currency": "peso", "capital": "mexico_city"})
    return mem


def test_facade_attribute_and_analogy():
    mem = _countries_facade()
    assert mem.attribute("USA", "currency").value == "dollar"
    assert mem.analogy("USA", "Mexico", "dollar").value == "peso"


def test_facade_analogical_records_survive_save_load(tmp_path):
    mem = _countries_facade()
    mem.store("episodic memory still works")  # episodic + analogical coexist
    path = str(tmp_path / "mem.json")
    mem.save(path)

    restored = Memory.load(path)
    assert restored.analogy("USA", "Mexico", "dollar").value == "peso"
    assert restored.attribute("Mexico", "capital").value == "mexico_city"
    assert len(restored) == 1  # episodic entry preserved too


def test_facade_without_records_saves_and_loads(tmp_path):
    mem = Memory()
    mem.store("just episodic, no records")
    path = str(tmp_path / "m.json")
    mem.save(path)
    restored = Memory.load(path)
    assert len(restored) == 1
