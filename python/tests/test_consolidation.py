"""Tests for kohaku.consolidation — semantic clustering via bundle-of-bundles."""
from __future__ import annotations

import numpy as np
import pytest

from kohaku._pure import DIMS, EpisodicMemory, HyperVector
from kohaku.consolidation import consolidate, consolidate_to_memory


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    """Flip `flip_frac` of bits to make a noisy variant of `base`."""
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def test_consolidate_isolates_unrelated_entries() -> None:
    mem = EpisodicMemory(capacity=10)
    for s in range(5):
        k = HyperVector.random(DIMS, seed=10_000 + s)
        v = HyperVector.random(DIMS, seed=20_000 + s)
        mem.store(k, v, label=f"item-{s}")
    clusters = consolidate(mem, similarity_threshold=0.3)
    # Random vectors in 10k-D should be near-orthogonal → 5 singleton clusters.
    assert len(clusters) == 5
    for c in clusters:
        assert c.size == 1


def test_consolidate_merges_noisy_variants() -> None:
    mem = EpisodicMemory(capacity=10)
    base_k = HyperVector.random(DIMS, seed=1)
    base_v = HyperVector.random(DIMS, seed=2)
    # 4 close variants — 5% bit flips → cosine ≈ 0.9
    for i in range(4):
        mem.store(_noisy(base_k, 0.05, seed=100 + i), base_v, label="cat")
    # 1 unrelated entry
    mem.store(HyperVector.random(DIMS, seed=999), base_v, label="dog")

    clusters = consolidate(mem, similarity_threshold=0.3)
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [1, 4]


def test_consolidate_centroid_concentrates() -> None:
    """The bundled centroid should be more similar to the prototype than any single member."""
    mem = EpisodicMemory(capacity=20)
    base_k = HyperVector.random(DIMS, seed=42)
    base_v = HyperVector.random(DIMS, seed=43)
    members = [_noisy(base_k, 0.20, seed=500 + i) for i in range(10)]
    for m in members:
        mem.store(m, base_v, label="x")

    clusters = consolidate(mem, similarity_threshold=0.1)
    # All 10 should land in one cluster
    assert len(clusters) == 1
    centroid = clusters[0].centroid_key
    centroid_sim = base_k.cosine_similarity(centroid)
    avg_member_sim = np.mean([base_k.cosine_similarity(m) for m in members])
    assert centroid_sim > avg_member_sim


def test_consolidate_to_memory_capacity_and_labels() -> None:
    mem = EpisodicMemory(capacity=20)
    base = HyperVector.random(DIMS, seed=7)
    val = HyperVector.random(DIMS, seed=8)
    for i in range(3):
        mem.store(_noisy(base, 0.05, seed=i), val, label="alpha")
    mem.store(HyperVector.random(DIMS, seed=999), val, label="beta")

    out = consolidate_to_memory(mem, similarity_threshold=0.3)
    labels = [e.label for e in out.entries()]
    assert any("alpha (n=3)" in lab for lab in labels)
    assert any("beta (n=1)" in lab for lab in labels)
    assert len(out) == 2


def test_consolidate_empty_memory() -> None:
    mem = EpisodicMemory(capacity=5)
    assert consolidate(mem) == []
    out = consolidate_to_memory(mem)
    assert len(out) == 0


def test_consolidate_threshold_validation() -> None:
    mem = EpisodicMemory(capacity=2)
    with pytest.raises(ValueError):
        consolidate(mem, similarity_threshold=2.0)
    with pytest.raises(ValueError):
        consolidate(mem, similarity_threshold=-2.0)


def test_consolidate_member_ids_track_originals() -> None:
    mem = EpisodicMemory(capacity=10)
    base = HyperVector.random(DIMS, seed=11)
    val = HyperVector.random(DIMS, seed=12)
    ids = []
    for i in range(3):
        ids.append(mem.store(_noisy(base, 0.04, seed=200 + i), val, "g"))
    clusters = consolidate(mem, similarity_threshold=0.3)
    assert len(clusters) == 1
    assert sorted(clusters[0].member_ids) == sorted(ids)
