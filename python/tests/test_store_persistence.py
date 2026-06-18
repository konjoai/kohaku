"""Persistence round-trips for the namespaced stores.

TenantMemoryStore and SharedMemoryPool persist as a directory of per-namespace
``.hkb`` files plus a ``manifest.json``. These tests prove the round-trip is
exact (memories, ids, config, retrieval) and that the format tag guards against
loading one store kind as the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from kohaku import SharedMemoryPool, TenantMemoryStore
from kohaku._pure import HyperVector


def _hv(seed: int, dims: int = 64) -> HyperVector:
    rng = np.random.default_rng(seed)
    return HyperVector(np.where(rng.random(dims) > 0.5, np.int8(1), np.int8(-1)))


# ── TenantMemoryStore ────────────────────────────────────────────────────────


def test_tenant_round_trip_preserves_memories_and_retrieval(tmp_path):
    store = TenantMemoryStore(dimension=64, capacity=50, default_capacity=20)
    k_a, v_a = _hv(1), _hv(2)
    store.store("tenant-a", k_a, v_a, "a-fact")
    store.store("tenant-a", _hv(3), _hv(4), "a-other")
    store.store("tenant-b", _hv(5), _hv(6), "b-fact")

    store.save(tmp_path / "snap")
    loaded = TenantMemoryStore.load(tmp_path / "snap")

    assert sorted(loaded.tenant_ids) == ["tenant-a", "tenant-b"]
    assert loaded.dimension == 64
    assert loaded.size("tenant-a") == 2
    assert loaded.size("tenant-b") == 1
    # Retrieval still works and isolation holds (a's key not in b's namespace).
    hit = loaded.retrieve("tenant-a", k_a, top_k=1)[0]
    assert hit.label == "a-fact"
    assert hit.similarity == pytest.approx(1.0, abs=1e-6)


def test_tenant_round_trip_preserves_config_and_ids(tmp_path):
    store = TenantMemoryStore(dimension=32, capacity=7, default_capacity=3)
    store.store("t", _hv(9, 32), _hv(10, 32), "x")
    original_id = store._tenants["t"].entries()[0].id

    store.save(tmp_path / "s")
    loaded = TenantMemoryStore.load(tmp_path / "s")

    assert loaded._capacity == 7
    assert loaded._default_capacity == 3
    assert loaded._tenants["t"].entries()[0].id == original_id
    # Per-namespace capacity survives (it lives in the .hkb body).
    assert loaded._tenants["t"]._capacity == 3


def test_tenant_empty_store_round_trips(tmp_path):
    TenantMemoryStore(dimension=64).save(tmp_path / "empty")
    loaded = TenantMemoryStore.load(tmp_path / "empty")
    assert loaded.tenants_count() == 0
    assert loaded.dimension == 64


# ── SharedMemoryPool ─────────────────────────────────────────────────────────


def test_shared_round_trip_preserves_pool_and_tagging(tmp_path):
    pool = SharedMemoryPool(dimension=64, default_capacity=100)
    probe = _hv(11)
    pool.write("agent-1", probe, _hv(12), "from-1")
    pool.write("agent-2", _hv(13), _hv(14), "from-2")

    pool.save(tmp_path / "pool")
    loaded = SharedMemoryPool.load(tmp_path / "pool")

    assert sorted(loaded.agent_ids) == ["agent-1", "agent-2"]
    assert loaded.total_size() == 2
    # Union read still tags the originating agent.
    top = loaded.query(probe, top_k=1)[0]
    assert top.agent_id == "agent-1"
    assert top.label == "from-1"
    assert top.similarity == pytest.approx(1.0, abs=1e-6)


def test_shared_read_scoping_after_load(tmp_path):
    pool = SharedMemoryPool(dimension=64)
    pool.write("a", _hv(1), _hv(2), "a")
    pool.write("b", _hv(3), _hv(4), "b")
    pool.save(tmp_path / "p")
    loaded = SharedMemoryPool.load(tmp_path / "p")
    # Restrict the union to one namespace.
    hits = loaded.query(_hv(3), top_k=5, agents=["b"])
    assert {h.agent_id for h in hits} == {"b"}


def test_shared_empty_pool_round_trips(tmp_path):
    SharedMemoryPool(dimension=128, default_capacity=42).save(tmp_path / "e")
    loaded = SharedMemoryPool.load(tmp_path / "e")
    assert loaded.agents_count() == 0
    assert loaded._default_capacity == 42


# ── format guards & ids ──────────────────────────────────────────────────────


def test_unicode_and_pathlike_namespace_ids_round_trip(tmp_path):
    # ids that would be unsafe as raw filenames must survive (they live in the
    # manifest; files are index-named).
    pool = SharedMemoryPool(dimension=64)
    for nid in ("a/b", "café", "../escape"):
        pool.write(nid, _hv(7), _hv(8), nid)
    pool.save(tmp_path / "u")
    loaded = SharedMemoryPool.load(tmp_path / "u")
    assert set(loaded.agent_ids) == {"a/b", "café", "../escape"}


def test_loading_tenant_dir_as_shared_pool_raises(tmp_path):
    TenantMemoryStore(dimension=64).save(tmp_path / "t")
    with pytest.raises(ValueError, match="format"):
        SharedMemoryPool.load(tmp_path / "t")


def test_loading_missing_manifest_raises(tmp_path):
    (tmp_path / "nope").mkdir()
    with pytest.raises(FileNotFoundError):
        TenantMemoryStore.load(tmp_path / "nope")
