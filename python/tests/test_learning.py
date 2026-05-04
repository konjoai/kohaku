"""Tests for kohaku.learning — online HDC item memory."""
from __future__ import annotations

import numpy as np
import pytest

from kohaku._pure import DIMS, HyperVector
from kohaku.learning import ItemMemory, Prototype


def _noisy(base: HyperVector, flip_frac: float, seed: int) -> HyperVector:
    rng = np.random.default_rng(seed)
    n_flip = int(flip_frac * len(base))
    idx = rng.choice(len(base), size=n_flip, replace=False)
    data = base.data.copy()
    data[idx] *= -1
    return HyperVector(data)


def test_empty_item_memory() -> None:
    im = ItemMemory()
    assert len(im) == 0
    assert im.labels() == []
    assert im.predict(HyperVector.random(DIMS, seed=1), top_k=3) == []
    assert "anything" not in im
    assert im.get("missing") is None


def test_add_creates_prototype() -> None:
    im = ItemMemory()
    v = HyperVector.random(DIMS, seed=1)
    proto = im.add("cat", v)
    assert isinstance(proto, Prototype)
    assert proto.label == "cat"
    assert proto.n_examples == 1
    assert "cat" in im
    assert len(im) == 1
    # Single positive example → prototype matches the input exactly.
    assert proto.vector.cosine_similarity(v) == pytest.approx(1.0)


def test_prototype_concentrates_with_repeated_examples() -> None:
    """Many noisy variants of one prototype → learned prototype gets close to the latent."""
    im = ItemMemory()
    base = HyperVector.random(DIMS, seed=42)
    # Observe 12 variants with 20% bit flips
    for i in range(12):
        im.add("animal", _noisy(base, 0.2, seed=100 + i))
    learned = im.get("animal").vector
    sim_to_truth = base.cosine_similarity(learned)
    assert sim_to_truth > 0.95, f"expected concentration > 0.95, got {sim_to_truth:.3f}"


def test_negative_feedback_pushes_away() -> None:
    """update(sign=-1) should pull the prototype away from the example."""
    im = ItemMemory()
    base = HyperVector.random(DIMS, seed=1)
    other = HyperVector.random(DIMS, seed=2)  # near-orthogonal in 10k-D
    im.add("x", base)
    sim_before = im.get("x").vector.cosine_similarity(other)
    # Hammer with negative examples of `other` repeatedly
    for _ in range(20):
        im.update("x", other, sign=-1, weight=2.0)
    sim_after = im.get("x").vector.cosine_similarity(other)
    assert sim_after < sim_before


def test_train_from_feedback_correct_vs_wrong() -> None:
    im = ItemMemory()
    cat_proto = HyperVector.random(DIMS, seed=1)
    dog_proto = HyperVector.random(DIMS, seed=2)
    im.add("cat", cat_proto)
    # Wrong: telling it dog is also cat → reinforce
    im.train_from_feedback("cat", dog_proto, correct=True)
    sim_after_pos = im.get("cat").vector.cosine_similarity(dog_proto)
    # Then wrong: tell it dog is NOT cat repeatedly → suppress
    for _ in range(15):
        im.train_from_feedback("cat", dog_proto, correct=False, weight=2.0)
    sim_after_neg = im.get("cat").vector.cosine_similarity(dog_proto)
    assert sim_after_neg < sim_after_pos


def test_predict_picks_correct_label() -> None:
    """Three labels with three latent prototypes — predict picks the right one."""
    im = ItemMemory()
    protos = {f"p{i}": HyperVector.random(DIMS, seed=1000 + i) for i in range(3)}
    for label, p in protos.items():
        for j in range(8):
            im.add(label, _noisy(p, 0.1, seed=int(label[1:]) * 100 + j))
    # Query with a fresh noisy version of p1
    q = _noisy(protos["p1"], 0.15, seed=99999)
    results = im.predict(q, top_k=3)
    assert len(results) == 3
    assert results[0].label == "p1"
    assert results[0].similarity > results[1].similarity


def test_dims_validation() -> None:
    im = ItemMemory(dims=DIMS)
    bad = HyperVector(np.ones(100, dtype=np.int8))
    with pytest.raises(ValueError, match="dims mismatch"):
        im.add("x", bad)
    im.add("ok", HyperVector.random(DIMS, seed=1))
    with pytest.raises(ValueError, match="dims mismatch"):
        im.predict(bad, top_k=1)


def test_invalid_dims_raises() -> None:
    with pytest.raises(ValueError, match="dims must be > 0"):
        ItemMemory(dims=0)


def test_invalid_sign_raises() -> None:
    im = ItemMemory()
    with pytest.raises(ValueError, match="sign must be"):
        im.update("x", HyperVector.random(DIMS, seed=1), sign=0)


def test_invalid_weight_raises() -> None:
    im = ItemMemory()
    with pytest.raises(ValueError, match="weight must be"):
        im.update("x", HyperVector.random(DIMS, seed=1), weight=0.0)


def test_clear_resets() -> None:
    im = ItemMemory()
    im.add("a", HyperVector.random(DIMS, seed=1))
    im.add("b", HyperVector.random(DIMS, seed=2))
    assert len(im) == 2
    im.clear()
    assert len(im) == 0
    assert im.labels() == []


def test_n_examples_tracks_updates() -> None:
    im = ItemMemory()
    v = HyperVector.random(DIMS, seed=1)
    im.add("x", v)
    im.add("x", v)
    im.update("x", v, sign=-1)
    assert im.get("x").n_examples == 3
