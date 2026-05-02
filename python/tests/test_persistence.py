"""Tests for kohaku.persistence — JSON + binary `.hkb` round-trips."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.persistence import (
    _MAGIC,
    load,
    load_binary,
    load_json,
    save,
    save_binary,
    save_json,
)


def _make_memory(n: int = 5, capacity: int = 10) -> EpisodicMemory:
    mem = EpisodicMemory(capacity=capacity)
    for i in range(n):
        k = HyperVector.random(DIMS, seed=100 + i)
        v = HyperVector.random(DIMS, seed=200 + i)
        mem.store(k, v, label=f"entry-{i}")
    return mem


def _assert_memories_equal(a: EpisodicMemory, b: EpisodicMemory) -> None:
    assert len(a) == len(b)
    assert a._capacity == b._capacity
    assert a._next_id == b._next_id
    assert a._timestamp == b._timestamp
    for ea, eb in zip(a.entries(), b.entries()):
        assert ea.id == eb.id
        assert ea.timestamp == eb.timestamp
        assert ea.label == eb.label
        assert np.array_equal(ea.key.data, eb.key.data)
        assert np.array_equal(ea.value.data, eb.value.data)


def test_json_roundtrip(tmp_path: Path) -> None:
    mem = _make_memory(n=4)
    p = tmp_path / "mem.json"
    save_json(mem, p)
    loaded = load_json(p)
    _assert_memories_equal(mem, loaded)


def test_json_is_human_readable(tmp_path: Path) -> None:
    mem = _make_memory(n=2)
    p = tmp_path / "mem.json"
    save_json(mem, p)
    payload = json.loads(p.read_text())
    assert payload["format"] == "kohaku-json"
    assert payload["dims"] == DIMS
    assert len(payload["entries"]) == 2
    assert payload["entries"][0]["label"] == "entry-0"


def test_binary_roundtrip(tmp_path: Path) -> None:
    mem = _make_memory(n=6)
    p = tmp_path / "mem.hkb"
    save_binary(mem, p)
    loaded = load_binary(p)
    _assert_memories_equal(mem, loaded)


def test_binary_magic_header(tmp_path: Path) -> None:
    mem = _make_memory(n=1)
    p = tmp_path / "mem.hkb"
    save_binary(mem, p)
    assert p.read_bytes()[:4] == _MAGIC


def test_binary_is_smaller_than_json(tmp_path: Path) -> None:
    mem = _make_memory(n=8)
    pj = tmp_path / "mem.json"
    pb = tmp_path / "mem.hkb"
    save_json(mem, pj)
    save_binary(mem, pb)
    # Bipolar JSON has ~3 chars per int (-1,...) so binary should be at least 10x smaller.
    assert pb.stat().st_size * 10 < pj.stat().st_size


def test_load_binary_rejects_bad_magic(tmp_path: Path) -> None:
    p = tmp_path / "garbage.hkb"
    p.write_bytes(b"NOPE" + b"\x00" * 100)
    with pytest.raises(ValueError, match="bad magic"):
        load_binary(p)


def test_load_binary_rejects_truncated(tmp_path: Path) -> None:
    p = tmp_path / "tiny.hkb"
    p.write_bytes(b"KH")
    with pytest.raises(ValueError, match="too small"):
        load_binary(p)


def test_save_and_load_dispatch_by_extension(tmp_path: Path) -> None:
    mem = _make_memory(n=3)
    pj = tmp_path / "m.json"
    pb = tmp_path / "m.hkb"
    save(mem, pj)
    save(mem, pb)
    _assert_memories_equal(mem, load(pj))
    _assert_memories_equal(mem, load(pb))


def test_save_unknown_extension_raises(tmp_path: Path) -> None:
    mem = _make_memory(n=1)
    with pytest.raises(ValueError, match="Unknown extension"):
        save(mem, tmp_path / "m.bin")


def test_empty_memory_roundtrip(tmp_path: Path) -> None:
    mem = EpisodicMemory(capacity=5)
    pj = tmp_path / "empty.json"
    pb = tmp_path / "empty.hkb"
    save(mem, pj)
    save(mem, pb)
    _assert_memories_equal(mem, load(pj))
    _assert_memories_equal(mem, load(pb))


def test_unicode_labels_roundtrip(tmp_path: Path) -> None:
    mem = EpisodicMemory(capacity=3)
    for label in ["ቆንጆ", "根性", "कोहजो", "konjo 🚀"]:
        k = HyperVector.random(DIMS, seed=hash(label) & 0xFFFF)
        v = HyperVector.random(DIMS, seed=(hash(label) + 1) & 0xFFFF)
        mem.store(k, v, label)
    pb = tmp_path / "unicode.hkb"
    save_binary(mem, pb)
    loaded = load_binary(pb)
    assert [e.label for e in loaded.entries()] == [e.label for e in mem.entries()]


def test_binary_preserves_query_results(tmp_path: Path) -> None:
    """Round-trip must preserve associative recall semantics, not just bytes."""
    from kohaku._query import query

    mem = _make_memory(n=8)
    # Use one of the keys as the query so we know the top hit
    target = mem.entries()[3].key
    expected = query(mem, target, top_k=3)

    pb = tmp_path / "recall.hkb"
    save_binary(mem, pb)
    loaded = load_binary(pb)
    actual = query(loaded, target, top_k=3)

    assert [r.entry_id for r in actual] == [r.entry_id for r in expected]
    for a, e in zip(actual, expected):
        assert a.similarity == pytest.approx(e.similarity, abs=1e-9)
