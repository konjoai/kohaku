"""Tests for kohaku.analogy — relational reasoning via VSA binding algebra.

Covers the two headline operations (attribute query, analogical transfer), the
determinism that makes them reproducible, confidence/margin reporting, and the
error/edge cases. The "dollar of Mexico" analogy is the canonical correctness
check (Kanerva 2010).
"""

from __future__ import annotations

import pytest

from kohaku import AnalogicalMemory, AnalogyResult


def _countries() -> AnalogicalMemory:
    m = AnalogicalMemory()
    m.add_record(
        "USA", {"currency": "dollar", "capital": "washington", "language": "english"}
    )
    m.add_record(
        "Mexico", {"currency": "peso", "capital": "mexico_city", "language": "spanish"}
    )
    m.add_record(
        "France", {"currency": "euro", "capital": "paris", "language": "french"}
    )
    m.add_record(
        "Japan", {"currency": "yen", "capital": "tokyo", "language": "japanese"}
    )
    return m


# ── attribute query ──────────────────────────────────────────────────────────


def test_attribute_query_recovers_value():
    m = _countries()
    assert m.get("USA", "currency").value == "dollar"
    assert m.get("France", "capital").value == "paris"
    assert m.get("Japan", "language").value == "japanese"


def test_attribute_query_confidence_is_top_and_positive():
    m = _countries()
    res = m.get("Mexico", "currency")
    assert res.value == "peso"
    assert res.confidence > 0.0
    assert res.ranked[0] == ("peso", res.confidence)
    assert res.margin > 0.0  # peso clearly separated from runner-up


# ── analogical transfer (the unique capability) ──────────────────────────────


def test_dollar_of_mexico():
    m = _countries()
    assert m.analogy("USA", "Mexico", "dollar").value == "peso"


def test_analogy_across_multiple_attributes_and_pairs():
    m = _countries()
    assert m.analogy("USA", "France", "washington").value == "paris"
    assert m.analogy("Japan", "USA", "yen").value == "dollar"
    assert m.analogy("France", "Japan", "french").value == "japanese"


def test_analogy_excludes_the_probe_value():
    m = _countries()
    # The analog of a value is never the value itself.
    assert m.analogy("USA", "Mexico", "dollar").value != "dollar"


# ── determinism ──────────────────────────────────────────────────────────────


def test_results_are_deterministic_across_instances():
    a = _countries().analogy("USA", "Mexico", "dollar")
    b = _countries().analogy("USA", "Mexico", "dollar")
    assert a.value == b.value
    assert a.confidence == b.confidence


# ── introspection / API surface ──────────────────────────────────────────────


def test_records_and_fields_roundtrip():
    m = _countries()
    assert set(m.records()) == {"USA", "Mexico", "France", "Japan"}
    assert m.fields("USA")["currency"] == "dollar"
    assert "USA" in m
    assert len(m) == 4


def test_re_adding_replaces_record():
    m = AnalogicalMemory()
    m.add_record("X", {"a": "one"})
    m.add_record("X", {"a": "two"})
    assert m.get("X", "a").value == "two"
    assert len(m) == 1


def test_result_is_falsy_when_empty():
    assert not AnalogyResult("", 0.0, ())
    assert AnalogyResult("peso", 0.2, (("peso", 0.2),))


# ── errors / edge cases ──────────────────────────────────────────────────────


def test_unknown_record_raises():
    m = _countries()
    with pytest.raises(ValueError, match="unknown record"):
        m.get("Atlantis", "currency")
    with pytest.raises(ValueError, match="unknown record"):
        m.analogy("USA", "Atlantis", "dollar")


def test_empty_fields_rejected():
    m = AnalogicalMemory()
    with pytest.raises(ValueError, match="at least one field"):
        m.add_record("Empty", {})


def test_invalid_dims_rejected():
    with pytest.raises(ValueError, match="dims"):
        AnalogicalMemory(dims=0)


def test_single_field_record_attribute_query():
    m = AnalogicalMemory()
    m.add_record("solo", {"color": "blue"})
    assert m.get("solo", "color").value == "blue"


def test_to_dict_from_dict_roundtrip():
    m = _countries()
    rebuilt = AnalogicalMemory.from_dict(m.to_dict())
    assert set(rebuilt.records()) == set(m.records())
    assert rebuilt.analogy("USA", "Mexico", "dollar").value == "peso"
    assert rebuilt.get("France", "capital").value == "paris"
