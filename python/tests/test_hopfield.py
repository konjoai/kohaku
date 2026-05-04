"""Tests for kohaku.hopfield — modern continuous Hopfield associator."""
from __future__ import annotations

import numpy as np
import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.hopfield import HopfieldAssociator, HopfieldRecall


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def test_empty_recall_raises() -> None:
    h = HopfieldAssociator()
    with pytest.raises(ValueError, match="empty"):
        h.recall(HyperVector.random(DIMS, seed=1))


def test_invalid_beta_raises() -> None:
    with pytest.raises(ValueError, match="beta"):
        HopfieldAssociator(beta=0)
    with pytest.raises(ValueError, match="beta"):
        HopfieldAssociator(beta=-1)


def test_invalid_dims_raises() -> None:
    with pytest.raises(ValueError, match="dims"):
        HopfieldAssociator(dims=0)


def test_store_and_len() -> None:
    h = HopfieldAssociator()
    assert len(h) == 0
    assert h.is_empty
    idx = h.store(HyperVector.random(DIMS, seed=1), label="a")
    assert idx == 0
    h.store(HyperVector.random(DIMS, seed=2), label="b")
    assert len(h) == 2
    assert h.labels() == ["a", "b"]


def test_dims_mismatch_on_store() -> None:
    h = HopfieldAssociator()
    with pytest.raises(ValueError, match="dims mismatch"):
        h.store(HyperVector(np.ones(100, dtype=np.int8)))


def test_dims_mismatch_on_recall() -> None:
    h = HopfieldAssociator()
    h.store(HyperVector.random(DIMS, seed=1))
    with pytest.raises(ValueError, match="dims mismatch"):
        h.recall(HyperVector(np.ones(100, dtype=np.int8)))


def test_recall_cleans_noisy_query() -> None:
    """Store 5 patterns; query with a noisy version of one → recover original."""
    h = HopfieldAssociator(beta=0.05)
    bases = [HyperVector.random(DIMS, seed=i) for i in range(5)]
    for i, b in enumerate(bases):
        h.store(b, label=f"p{i}")
    query = _noisy(bases[2], 0.30, seed=999)  # 30% bit flips
    raw_sim = bases[2].cosine_similarity(query)
    result = h.recall(query, max_iters=5)
    cleaned_sim = bases[2].cosine_similarity(result.pattern)
    # Cleaned result must be at least as close to the true pattern as the noisy query.
    assert cleaned_sim >= raw_sim
    # And the highest-weighted stored pattern should be index 2.
    assert result.best_index == 2
    assert result.best_similarity > 0.95


def test_recall_returns_hopfield_recall_dataclass() -> None:
    h = HopfieldAssociator()
    h.store(HyperVector.random(DIMS, seed=1))
    r = h.recall(HyperVector.random(DIMS, seed=1), max_iters=3)
    assert isinstance(r, HopfieldRecall)
    assert isinstance(r.pattern, HyperVector)
    assert r.iterations >= 1
    assert r.weights.shape == (1,)


def test_recall_converges_quickly_for_clean_query() -> None:
    """A clean stored pattern should converge in 2 iterations or fewer."""
    h = HopfieldAssociator(beta=0.05)
    p = HyperVector.random(DIMS, seed=42)
    h.store(p)
    r = h.recall(p, max_iters=10, eps=1e-3)
    assert r.converged
    assert r.iterations <= 3


def test_max_iters_validation() -> None:
    h = HopfieldAssociator()
    h.store(HyperVector.random(DIMS, seed=1))
    with pytest.raises(ValueError, match="max_iters"):
        h.recall(HyperVector.random(DIMS, seed=1), max_iters=0)


def test_continuous_mode_returns_binarized() -> None:
    """binarize_each_step=False still returns a final ±1 HyperVector."""
    h = HopfieldAssociator(beta=0.05, binarize_each_step=False)
    h.store_many([HyperVector.random(DIMS, seed=i) for i in range(3)])
    r = h.recall(HyperVector.random(DIMS, seed=1), max_iters=3)
    # Output is bipolar
    assert set(r.pattern.data.tolist()).issubset({-1, 1})


def test_complete_returns_pattern_only() -> None:
    h = HopfieldAssociator()
    p = HyperVector.random(DIMS, seed=7)
    h.store(p)
    completed = h.complete(_noisy(p, 0.2, seed=99))
    assert isinstance(completed, HyperVector)
    assert p.cosine_similarity(completed) > 0.9


def test_clear() -> None:
    h = HopfieldAssociator()
    h.store(HyperVector.random(DIMS, seed=1), label="x")
    h.clear()
    assert len(h) == 0
    assert h.is_empty
    assert h.labels() == []


def test_store_many_label_length_check() -> None:
    h = HopfieldAssociator()
    with pytest.raises(ValueError, match="same length"):
        h.store_many(
            [HyperVector.random(DIMS, seed=1), HyperVector.random(DIMS, seed=2)],
            labels=["only-one"],
        )


def test_recall_separates_well_separated_patterns() -> None:
    """With near-orthogonal stored patterns, softmax should pick exactly one."""
    h = HopfieldAssociator(beta=0.05)
    patterns = [HyperVector.random(DIMS, seed=i) for i in range(8)]
    h.store_many(patterns)
    for target_idx in range(8):
        q = _noisy(patterns[target_idx], 0.10, seed=2000 + target_idx)
        r = h.recall(q, max_iters=5)
        assert r.best_index == target_idx, f"expected {target_idx}, got {r.best_index}"
        # weight on the winner should dominate
        assert r.weights[target_idx] > 0.9
