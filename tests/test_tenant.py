"""Tests for TenantMemoryStore."""
import pytest
from kohaku.tenant import TenantMemoryStore
from kohaku._pure import HyperVector


DIM = 64


def rand_hv(seed=0):
    return HyperVector.random(dims=DIM, seed=seed)


def test_tenant_store_init():
    store = TenantMemoryStore(dimension=DIM, capacity=100)
    assert store.dimension == DIM
    assert store.tenants_count() == 0


def test_tenant_bad_dimension():
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=0)


def test_tenant_bad_capacity():
    with pytest.raises(ValueError):
        TenantMemoryStore(dimension=DIM, capacity=0)


def test_tenant_empty_id_rejected():
    store = TenantMemoryStore(dimension=DIM)
    k = rand_hv(0)
    v = rand_hv(1)
    with pytest.raises(ValueError):
        store.store("", k, v)


def test_tenant_auto_provision():
    store = TenantMemoryStore(dimension=DIM)
    store.store("alice", rand_hv(0), rand_hv(1))
    assert "alice" in store.tenant_ids
    assert store.tenants_count() == 1


def test_tenant_isolation():
    store = TenantMemoryStore(dimension=DIM)
    ka = rand_hv(0)
    va = rand_hv(1)
    kb = rand_hv(2)
    vb = rand_hv(3)
    store.store("alice", ka, va, label="alice-entry")
    store.store("bob", kb, vb, label="bob-entry")
    alice_results = store.retrieve("alice", ka, top_k=1)
    assert len(alice_results) == 1
    assert alice_results[0].label == "alice-entry"
    bob_results = store.retrieve("bob", kb, top_k=1)
    assert len(bob_results) == 1
    assert bob_results[0].label == "bob-entry"


def test_tenant_size():
    store = TenantMemoryStore(dimension=DIM)
    assert store.size("unknown") == 0
    store.store("alice", rand_hv(0), rand_hv(1))
    store.store("alice", rand_hv(2), rand_hv(3))
    assert store.size("alice") == 2


def test_tenant_drop():
    store = TenantMemoryStore(dimension=DIM)
    store.store("alice", rand_hv(0), rand_hv(1))
    assert store.drop_tenant("alice") is True
    assert "alice" not in store.tenant_ids


def test_tenant_drop_nonexistent():
    store = TenantMemoryStore(dimension=DIM)
    assert store.drop_tenant("ghost") is False


def test_tenant_multiple():
    store = TenantMemoryStore(dimension=DIM)
    for i, name in enumerate(["alice", "bob", "carol"]):
        store.store(name, rand_hv(i), rand_hv(i + 10))
    assert store.tenants_count() == 3
    assert set(store.tenant_ids) == {"alice", "bob", "carol"}


def test_tenant_cross_isolation():
    """Alice's retrieval should never return Bob's entries."""
    store = TenantMemoryStore(dimension=DIM)
    alice_key = rand_hv(0)
    store.store("alice", alice_key, rand_hv(10), label="alice")
    store.store("bob", rand_hv(5), rand_hv(15), label="bob")
    results = store.retrieve("alice", alice_key, top_k=5)
    for r in results:
        assert r.label == "alice"
