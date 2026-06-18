"""Tests for kohaku.compositional — multi-cue composition and Hopfield cleanup.

Covers the two primitives (`compose`, `complete_cue`) and the `Memory`
facade's `recall_composite`: soft-conjunction retrieval and robust recall of a
noisy cue via pattern completion.
"""

from __future__ import annotations

import numpy as np
import pytest

from kohaku import Memory, compose, complete_cue
from kohaku._pure import HyperVector


def _flip(hv: HyperVector, frac: float, seed: int) -> HyperVector:
    """Corrupt ``frac`` of a bipolar vector's components (deterministic)."""
    rng = np.random.default_rng(seed)
    data = hv.data.copy()
    n = int(len(data) * frac)
    idx = rng.choice(len(data), size=n, replace=False)
    data[idx] *= -1
    return HyperVector(data)


# ── compose ──────────────────────────────────────────────────────────────────


def test_compose_single_cue_is_identity():
    hv = HyperVector.random(seed=1)
    assert compose([hv]).cosine_similarity(hv) == 1.0


def test_compose_bundle_is_similar_to_each_cue():
    a = HyperVector.random(seed=1)
    b = HyperVector.random(seed=2)
    composite = compose([a, b])
    # The bundle correlates positively with both ingredients.
    assert composite.cosine_similarity(a) > 0.3
    assert composite.cosine_similarity(b) > 0.3


def test_compose_empty_raises():
    with pytest.raises(ValueError, match="at least one cue"):
        compose([])


# ── complete_cue (Hopfield pattern completion) ───────────────────────────────


def test_complete_cue_no_keys_returns_input():
    hv = HyperVector.random(seed=3)
    assert complete_cue(hv, []).cosine_similarity(hv) == 1.0


def test_complete_cue_recovers_corrupted_key():
    keys = [HyperVector.random(seed=s) for s in range(10)]
    target = keys[4]
    noisy = _flip(target, 0.2, seed=99)  # 20% corruption
    cleaned = complete_cue(noisy, keys)
    # Cleanup should move the cue closer to the stored attractor than the noise.
    assert cleaned.cosine_similarity(target) > noisy.cosine_similarity(target)


# ── facade: recall_composite ─────────────────────────────────────────────────


def _facade() -> Memory:
    mem = Memory()
    mem.store("User loves Italian red wine from Tuscany")
    mem.store("User enjoys French cheese and baguettes")
    mem.store("User prefers Japanese green tea in the morning")
    mem.store("User dislikes Italian coffee but likes espresso")
    return mem


def test_recall_composite_soft_conjunction():
    mem = _facade()
    hits = mem.recall_composite(
        ["Italian", "wine", "Tuscany"], top_k=1, reinforce=False
    )
    assert hits[0].text.startswith("User loves Italian red wine")


def test_recall_composite_cleanup_returns_results():
    mem = _facade()
    hits = mem.recall_composite(
        ["Japanese", "tea"], top_k=1, cleanup=True, reinforce=False
    )
    assert hits[0].text.startswith("User prefers Japanese green tea")


def test_recall_composite_respects_top_k():
    mem = _facade()
    assert len(mem.recall_composite(["Italian"], top_k=2, reinforce=False)) == 2


def test_recall_composite_empty_cues_raises():
    mem = _facade()
    with pytest.raises(ValueError, match="at least one non-empty cue"):
        mem.recall_composite(["", "   "])


def test_recall_composite_cleanup_empty_store():
    # cleanup against an empty store is a no-op, not a crash
    mem = Memory()
    assert mem.recall_composite(["anything"], cleanup=True, reinforce=False) == []
