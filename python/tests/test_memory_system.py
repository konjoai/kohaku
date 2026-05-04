"""Tests for kohaku.memory_system — combined episodic + semantic stores."""
from __future__ import annotations

import numpy as np
import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.decay import DecayConfig
from kohaku.memory_system import CombinedRecall, MemorySystem


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def test_empty_system() -> None:
    ms = MemorySystem(episodic_capacity=10)
    assert ms.num_episodes == 0
    assert ms.num_concepts == 0
    assert len(ms) == 0
    assert ms.recall(HyperVector.random(DIMS, seed=1)) == []


def test_repr() -> None:
    ms = MemorySystem(episodic_capacity=10)
    s = repr(ms)
    assert "episodes=0/10" in s
    assert "concepts=0" in s


def test_store_episode_increments_episodic() -> None:
    ms = MemorySystem(episodic_capacity=10)
    k = HyperVector.random(DIMS, seed=1)
    eid = ms.store_episode(k, k, label="a")
    assert eid == 1
    assert ms.num_episodes == 1
    assert ms.num_concepts == 0


def test_reinforce_concept_creates_semantic_only() -> None:
    ms = MemorySystem(episodic_capacity=10)
    ms.reinforce_concept("dog", HyperVector.random(DIMS, seed=1))
    assert ms.num_concepts == 1
    assert ms.num_episodes == 0


def test_consolidate_promotes_clusters_to_semantic() -> None:
    """Episodic clusters get pushed into semantic memory as prototypes."""
    ms = MemorySystem(episodic_capacity=20)
    cat = HyperVector.random(DIMS, seed=1)
    dog = HyperVector.random(DIMS, seed=2)
    val = HyperVector.random(DIMS, seed=99)
    # 4 noisy cats, 3 noisy dogs
    for i in range(4):
        ms.store_episode(_noisy(cat, 0.05, seed=100 + i), val, label="cat")
    for i in range(3):
        ms.store_episode(_noisy(dog, 0.05, seed=200 + i), val, label="dog")

    n_promoted = ms.consolidate_to_semantic(similarity_threshold=0.3)
    assert n_promoted == 2
    assert ms.num_concepts == 2
    # The semantic prototypes should be very close to the latent prototypes.
    cat_proto = ms.semantic.get("cat").vector
    dog_proto = ms.semantic.get("dog").vector
    assert cat.cosine_similarity(cat_proto) > 0.95
    assert dog.cosine_similarity(dog_proto) > 0.95


def test_recall_merges_episodic_and_semantic() -> None:
    ms = MemorySystem(episodic_capacity=20)
    cat = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=99)
    # Episodic side
    ms.store_episode(_noisy(cat, 0.05, seed=10), val, label="cat-episode-1")
    ms.store_episode(_noisy(cat, 0.05, seed=11), val, label="cat-episode-2")
    # Semantic side
    ms.reinforce_concept("cat-concept", cat)

    results = ms.recall(cat, top_k=4)
    assert len(results) > 0
    sources = {r.source for r in results}
    assert "episodic" in sources
    assert "semantic" in sources
    # semantic prototype should be the strongest match (it's the exact prototype)
    assert results[0].source == "semantic"
    assert results[0].similarity == pytest.approx(1.0)
    # Sorted descending
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_recall_with_decay_uses_provided_config() -> None:
    """use_decay=True with a DecayConfig dampens episodic similarity by age."""
    ms = MemorySystem(episodic_capacity=50)
    cat = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=99)
    # Store target as the *first* episode → it will be the oldest.
    ms.store_episode(cat, val, label="oldest")
    # Pad with unrelated noise to age it
    for i in range(10):
        ms.store_episode(HyperVector.random(DIMS, seed=500 + i), val, label=f"f{i}")

    # No-decay recall
    plain = ms.recall(cat, top_k=1, use_decay=False)
    assert plain[0].similarity == pytest.approx(1.0, abs=1e-6)

    # Heavily-decayed recall
    decayed = ms.recall(cat, top_k=1, use_decay=True,
                        decay_config=DecayConfig(half_life=2.0))
    assert decayed[0].similarity < plain[0].similarity


def test_default_decay_config_used_when_none_provided() -> None:
    ms = MemorySystem(
        episodic_capacity=10,
        decay_config=DecayConfig(half_life=5.0),
    )
    cat = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=99)
    ms.store_episode(cat, val, label="a")
    for i in range(8):
        ms.store_episode(HyperVector.random(DIMS, seed=600 + i), val, label="f")
    # use_decay=True with no explicit config → falls back to instance default
    decayed = ms.recall(cat, top_k=1, use_decay=True)
    assert decayed[0].similarity < 1.0


def test_teach_supervises_semantic() -> None:
    ms = MemorySystem(episodic_capacity=5)
    proto = HyperVector.random(DIMS, seed=1)
    ms.teach("x", proto, correct=True)
    assert ms.semantic.get("x").n_examples == 1
    ms.teach("x", proto, correct=False)
    assert ms.semantic.get("x").n_examples == 2


def test_combined_recall_dataclass_shape() -> None:
    ms = MemorySystem(episodic_capacity=5)
    p = HyperVector.random(DIMS, seed=1)
    ms.reinforce_concept("c", p)
    rs = ms.recall(p, top_k=1)
    assert isinstance(rs[0], CombinedRecall)
    assert rs[0].source in ("episodic", "semantic")
    assert isinstance(rs[0].value, HyperVector)


def test_consolidate_repeated_calls_are_idempotent_in_count() -> None:
    """Calling consolidate twice should not duplicate semantic concepts (same labels)."""
    ms = MemorySystem(episodic_capacity=20)
    cat = HyperVector.random(DIMS, seed=1)
    val = HyperVector.random(DIMS, seed=99)
    for i in range(4):
        ms.store_episode(_noisy(cat, 0.05, seed=100 + i), val, label="cat")
    ms.consolidate_to_semantic(similarity_threshold=0.3)
    n1 = ms.num_concepts
    ms.consolidate_to_semantic(similarity_threshold=0.3)
    assert ms.num_concepts == n1  # same label → same prototype, just reinforced
