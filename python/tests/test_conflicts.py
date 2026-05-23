"""Tests for kohaku.conflicts — contradiction signal detection."""

from __future__ import annotations

import pytest

from kohaku import EnrichedMemoryStore, ProvenanceGraph, encode_text
from kohaku.conflicts import (
    ConflictPair,
    detect_conflicts,
    resolve_conflict,
)


def _store(provenance: bool = False) -> EnrichedMemoryStore:
    pg = ProvenanceGraph() if provenance else None
    return EnrichedMemoryStore(capacity=30, provenance=pg)


def test_no_conflicts_in_empty_store() -> None:
    assert detect_conflicts(_store()) == []


def test_negation_pair_flagged() -> None:
    s = _store()
    a = "the user is currently asleep"
    b = "the user is not currently asleep"
    ha, hb = encode_text(a), encode_text(b)
    s.store(ha, ha, label=a)
    s.store(hb, hb, label=b)
    pairs = detect_conflicts(s)
    assert len(pairs) == 1
    assert "one side negated" in " ".join(pairs[0].reasons)


def test_numeric_divergence_flagged() -> None:
    s = _store()
    s.store(encode_text("the meeting is at 3pm"),
            encode_text("the meeting is at 3pm"),
            label="the meeting is at 3pm")
    s.store(encode_text("the meeting is at 5pm"),
            encode_text("the meeting is at 5pm"),
            label="the meeting is at 5pm")
    pairs = detect_conflicts(s)
    assert pairs
    assert any("numeric divergence" in r for r in pairs[0].reasons)


def test_predicate_divergence_caught_via_jaccard() -> None:
    """user prefers X / user prefers Y — share 'morning' but mostly diverge."""
    s = _store()
    a = "user prefers green tea in the morning"
    b = "user prefers black coffee in the morning"
    s.store(encode_text(a), encode_text(a), label=a)
    s.store(encode_text(b), encode_text(b), label=b)
    pairs = detect_conflicts(s)
    assert pairs, "preference flip should be flagged at default threshold"
    assert any("Jaccard" in r for r in pairs[0].reasons)


def test_unrelated_pair_not_flagged() -> None:
    s = _store()
    s.store(encode_text("cats sleep most of the day"),
            encode_text("cats sleep most of the day"),
            label="cats sleep most of the day")
    s.store(encode_text("the espresso machine is broken"),
            encode_text("the espresso machine is broken"),
            label="the espresso machine is broken")
    assert detect_conflicts(s) == []


def test_thresholds_validate_inputs() -> None:
    s = _store()
    with pytest.raises(ValueError):
        detect_conflicts(s, similarity_threshold=0.0)
    with pytest.raises(ValueError):
        detect_conflicts(s, contradiction_threshold=1.5)
    with pytest.raises(ValueError):
        detect_conflicts(s, max_pairs=-1)


def test_returns_sorted_by_score_descending() -> None:
    s = _store()
    # Strong: full negation pair
    s.store(encode_text("the door is open"),
            encode_text("the door is open"),
            label="the door is open")
    s.store(encode_text("the door is not open"),
            encode_text("the door is not open"),
            label="the door is not open")
    # Weaker: preference flip
    a = "user prefers tea"
    b = "user prefers coffee"
    s.store(encode_text(a), encode_text(a), label=a)
    s.store(encode_text(b), encode_text(b), label=b)
    pairs = detect_conflicts(s, contradiction_threshold=0.40)
    assert len(pairs) >= 2
    assert pairs[0].contradiction_score >= pairs[-1].contradiction_score


def test_conflict_pair_to_dict_shape() -> None:
    s = _store()
    s.store(encode_text("door open"), encode_text("door open"), label="door open")
    s.store(encode_text("door not open"), encode_text("door not open"),
            label="door not open")
    pair = detect_conflicts(s)[0]
    d = pair.to_dict()
    assert {"a_id", "b_id", "label_a", "label_b", "similarity",
            "contradiction_score", "reasons"} <= d.keys()
    assert isinstance(d["reasons"], list)


def test_resolve_keep_a_drops_b() -> None:
    s = _store(provenance=True)
    a_id = s.store(encode_text("door open"), encode_text("door open"),
                    label="door open")
    b_id = s.store(encode_text("door not open"), encode_text("door not open"),
                    label="door not open")
    outcome = resolve_conflict(s, a_id=a_id, b_id=b_id, keep="a")
    assert outcome.action == "keep_a"
    assert outcome.kept_id == a_id
    assert outcome.removed_ids == (b_id,)
    live = {e.id for e in s.episodic.entries()}
    assert live == {a_id}


def test_resolve_keep_both_makes_no_changes() -> None:
    s = _store()
    a_id = s.store(encode_text("x"), encode_text("x"), label="x")
    b_id = s.store(encode_text("y"), encode_text("y"), label="y")
    before = {e.id for e in s.episodic.entries()}
    outcome = resolve_conflict(s, a_id=a_id, b_id=b_id, keep="both")
    after = {e.id for e in s.episodic.entries()}
    assert before == after
    assert outcome.kept_id is None and outcome.removed_ids == ()


def test_resolve_validates_inputs() -> None:
    s = _store()
    with pytest.raises(ValueError):
        resolve_conflict(s, a_id=1, b_id=1, keep="a")
    with pytest.raises(ValueError):
        resolve_conflict(s, a_id=1, b_id=2, keep="bogus")
    # unknown ids
    a_id = s.store(encode_text("x"), encode_text("x"), label="x")
    with pytest.raises(ValueError):
        resolve_conflict(s, a_id=a_id, b_id=9999, keep="a")


def test_max_pairs_caps_results() -> None:
    s = _store()
    pairs_in = [
        ("door open", "door not open"),
        ("light on", "light not on"),
        ("user awake", "user not awake"),
    ]
    for a, b in pairs_in:
        s.store(encode_text(a), encode_text(a), label=a)
        s.store(encode_text(b), encode_text(b), label=b)
    out = detect_conflicts(s, max_pairs=2)
    assert len(out) <= 2
