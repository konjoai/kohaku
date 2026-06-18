"""Tests for kohaku.extraction — heuristic free-text → triple extraction.

The extractor is deliberately high-precision: it must parse the supported
memory-statement forms exactly, and — just as important — yield *nothing* for
prose it cannot confidently structure (no fabricated facts). These tests also
cover the integration points that let analogical reasoning run on learned text:
:meth:`AnalogicalMemory.learn` and :meth:`Memory.learn`.
"""

from __future__ import annotations

import pytest

from kohaku import (
    AnalogicalMemory,
    Memory,
    Triple,
    extract_triples,
    records_from_texts,
)


# ── pattern coverage ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The capital of France is Paris", ("france", "capital", "paris")),
        ("Tokyo is the capital of Japan", ("japan", "capital", "tokyo")),
        ("France's currency is the euro", ("france", "currency", "euro")),
        ("Mexico's currency: peso", ("mexico", "currency", "peso")),
        ("Fido is a dog", ("fido", "type", "dog")),
        ("The user prefers aisle seats", ("user", "prefers", "aisle seats")),
    ],
)
def test_each_pattern_extracts_expected_triple(text, expected):
    triples = extract_triples(text)
    assert len(triples) == 1
    t = triples[0]
    assert (t.subject, t.attribute, t.value) == expected


def test_articles_and_case_are_normalised():
    (t,) = extract_triples("The Currency of The USA is The Dollar")
    assert (t.subject, t.attribute, t.value) == ("usa", "currency", "dollar")


def test_confidence_is_reported_and_bounded():
    (t,) = extract_triples("The capital of France is Paris")
    assert isinstance(t, Triple)
    assert 0.0 < t.confidence <= 1.0


# ── precision: no fabrication ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "I went hiking yesterday",
        "What time is it?",
        "Please remember to call mom",
        "",
        "   ",
    ],
)
def test_unparseable_text_yields_nothing(text):
    assert extract_triples(text) == []


# ── multi-clause ──────────────────────────────────────────────────────────────


def test_multiple_clauses_extract_multiple_triples():
    triples = extract_triples(
        "The capital of Japan is Tokyo and the capital of France is Paris"
    )
    pairs = {(t.subject, t.value) for t in triples}
    assert ("japan", "tokyo") in pairs
    assert ("france", "paris") in pairs


# ── grouping helper ───────────────────────────────────────────────────────────


def test_records_from_texts_groups_by_subject():
    records = records_from_texts(
        [
            "The capital of USA is Washington",
            "USA's currency is the dollar",
            "The capital of Mexico is Mexico City",
        ]
    )
    assert records["usa"] == {"capital": "washington", "currency": "dollar"}
    assert records["mexico"] == {"capital": "mexico city"}


def test_records_from_texts_later_mention_overwrites():
    records = records_from_texts(
        ["France's currency is franc", "France's currency is the euro"]
    )
    assert records["france"]["currency"] == "euro"


# ── AnalogicalMemory.learn integration ────────────────────────────────────────


def test_analogical_learn_builds_records_for_reasoning():
    m = AnalogicalMemory()
    for text in [
        "USA's currency is the dollar",
        "The capital of USA is Washington",
        "Mexico's currency is the peso",
        "The capital of Mexico is Mexico City",
    ]:
        m.learn(text)
    # attribute recall works on extracted records
    assert m.get("usa", "currency").value == "dollar"
    # the canonical analogy now runs on text the agent "read", not hand records
    assert m.analogy("usa", "mexico", "dollar").value == "peso"


def test_analogical_learn_merges_into_existing_subject():
    m = AnalogicalMemory()
    m.learn("USA's currency is the dollar")
    m.learn("The capital of USA is Washington")
    assert m.fields("usa") == {"currency": "dollar", "capital": "washington"}


def test_analogical_learn_returns_triples_and_skips_noise():
    m = AnalogicalMemory()
    learned = m.learn("I had coffee this morning")
    assert learned == []
    assert "usa" not in m


# ── Memory facade learn integration ───────────────────────────────────────────


def test_facade_learn_stores_episodic_and_structured():
    mem = Memory(dims=1024)
    eid, triples = mem.learn("The capital of France is Paris")
    # episodic side: text is retrievable
    hits = mem.query("capital of France")
    assert any(h.text == "The capital of France is Paris" for h in hits)
    # structured side: attribute recall works
    assert (triples[0].subject, triples[0].value) == ("france", "paris")
    assert mem.attribute("france", "capital").value == "paris"
    assert isinstance(eid, int)


def test_facade_learn_with_noise_still_stores_text():
    mem = Memory(dims=1024)
    eid, triples = mem.learn("Remember the milk")
    assert triples == []
    assert isinstance(eid, int)
    assert mem.query("milk")
