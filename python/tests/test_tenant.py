"""Tests for kohaku.tenant — TenantMemoryStore multi-tenant isolation."""
from __future__ import annotations

import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.tenant import TenantMemoryStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _hv(seed: int) -> HyperVector:
    return HyperVector.random(DIMS, seed=seed)


def _store() -> TenantMemoryStore:
    return TenantMemoryStore(dimension=DIMS, capacity=50)


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------

def test_init_defaults() -> None:
    ts = _store()
    assert ts.dimension == DIMS
    assert ts.tenants_count() == 0
    assert ts.tenant_ids == []


def test_init_bad_dimension_raises() -> None:
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=0, capacity=10)
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=-1, capacity=10)


def test_init_bad_capacity_raises() -> None:
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=DIMS, capacity=0)
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=DIMS, capacity=-5)


# ---------------------------------------------------------------------------
# empty-ID rejection
# ---------------------------------------------------------------------------

def test_empty_tenant_id_raises_on_store() -> None:
    ts = _store()
    with pytest.raises(ValueError):
        ts.store("", _hv(1), _hv(2))


def test_empty_tenant_id_raises_on_retrieve() -> None:
    ts = _store()
    with pytest.raises(ValueError):
        ts.retrieve("", _hv(1))


# ---------------------------------------------------------------------------
# auto-provisioning
# ---------------------------------------------------------------------------

def test_unknown_tenant_auto_provisioned_on_store() -> None:
    ts = _store()
    ts.store("alice", _hv(1), _hv(2), label="hello")
    assert "alice" in ts.tenant_ids
    assert ts.tenants_count() == 1


def test_unknown_tenant_size_returns_zero() -> None:
    ts = _store()
    assert ts.size("nonexistent") == 0
    # Checking size for a non-existent tenant should NOT auto-provision
    assert ts.tenants_count() == 0


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------

def test_tenant_isolation_store_and_retrieve() -> None:
    """Memories stored under tenant A must not appear when querying tenant B."""
    ts = _store()
    key_a = _hv(seed=1)
    val_a = _hv(seed=2)
    key_b = _hv(seed=3)
    val_b = _hv(seed=4)

    ts.store("alice", key_a, val_a, label="alice-memory")
    ts.store("bob", key_b, val_b, label="bob-memory")

    # Alice queries with her own key — should get her memory
    alice_results = ts.retrieve("alice", key_a, top_k=1)
    assert len(alice_results) == 1
    assert alice_results[0].label == "alice-memory"

    # Bob queries with Alice's key — should not get Alice's memory
    bob_results = ts.retrieve("bob", key_a, top_k=1)
    if bob_results:  # bob has 1 entry (key_b), so result will be from that
        assert bob_results[0].label != "alice-memory"


def test_size_per_tenant() -> None:
    ts = _store()
    for i in range(3):
        ts.store("alice", _hv(seed=i), _hv(seed=i + 100), label=f"a{i}")
    for i in range(5):
        ts.store("bob", _hv(seed=i + 50), _hv(seed=i + 150), label=f"b{i}")

    assert ts.size("alice") == 3
    assert ts.size("bob") == 5
    assert ts.tenants_count() == 2


# ---------------------------------------------------------------------------
# drop_tenant
# ---------------------------------------------------------------------------

def test_drop_tenant_removes_all_data() -> None:
    ts = _store()
    ts.store("alice", _hv(1), _hv(2), label="x")
    assert ts.size("alice") == 1
    result = ts.drop_tenant("alice")
    assert result is True
    assert ts.size("alice") == 0
    assert "alice" not in ts.tenant_ids


def test_drop_nonexistent_tenant_returns_false() -> None:
    ts = _store()
    assert ts.drop_tenant("nobody") is False


# ---------------------------------------------------------------------------
# multiple tenants
# ---------------------------------------------------------------------------

def test_multiple_tenants_independent() -> None:
    """Ten tenants each storing and retrieving their own memories."""
    ts = TenantMemoryStore(dimension=DIMS, capacity=20)
    keys = {f"user_{i}": _hv(seed=i * 7 + 1) for i in range(10)}
    vals = {f"user_{i}": _hv(seed=i * 7 + 2) for i in range(10)}

    for tid, k in keys.items():
        ts.store(tid, k, vals[tid], label=tid)

    assert ts.tenants_count() == 10

    for tid, k in keys.items():
        results = ts.retrieve(tid, k, top_k=1)
        assert len(results) == 1
        assert results[0].label == tid, (
            f"tenant {tid} got wrong label: {results[0].label}"
        )
