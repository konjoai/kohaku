"""Compositional & robust recall — query memory by several cues at once, with
optional Hopfield pattern-completion to denoise a partial or noisy composite cue.

Plain cosine retrieval ranks by similarity to a *single* query vector. Real
recall is often multi-constraint and fragmentary — "the meeting about pricing,
last quarter". The HDC substrate handles this natively:

* **Compose** the cue vectors into one composite (``bundle``) — a soft
  conjunction, where the memory closest to *all* the cues scores highest.
* **Complete** a noisy/partial composite by letting a Hopfield associator pull
  it toward the nearest stored attractor before ranking (pattern completion),
  so recall degrades gracefully as the cue gets corrupted.

Both are pure-Python, deterministic, and need no model call. ``complete_cue`` is
``O(N·D)`` in the number of stored keys, so it is opt-in.
"""
from __future__ import annotations

from typing import Sequence

from kohaku._pure import HyperVector
from kohaku.hopfield import HopfieldAssociator


def compose(cues: Sequence[HyperVector]) -> HyperVector:
    """Bundle cue vectors into one composite query (a soft conjunction).

    The memory closest to *all* cues scores highest. A single cue is returned
    unchanged. Raises ``ValueError`` on an empty cue list.
    """
    items = list(cues)
    if not items:
        raise ValueError("compose requires at least one cue")
    if len(items) == 1:
        return items[0]
    return HyperVector.bundle_all(items)


def complete_cue(
    cue: HyperVector, keys: Sequence[HyperVector], *, max_iters: int = 5
) -> HyperVector:
    """Pull a noisy/partial ``cue`` toward the nearest stored key (Hopfield).

    Returns the cue unchanged when there are no keys to clean against. ``keys``
    are the stored memory hypervectors that act as attractors.
    """
    patterns = list(keys)
    if not patterns:
        return cue
    assoc = HopfieldAssociator(dims=len(cue))
    assoc.store_many(patterns)
    return assoc.complete(cue, max_iters=max_iters)
