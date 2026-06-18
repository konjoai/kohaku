"""Memory consolidation — semantic clustering via bundle-of-bundles.

Greedy single-pass clustering over an EpisodicMemory:

  for each entry e (insertion order):
    find existing cluster c whose centroid has cosine(c.centroid, e.key) ≥ threshold.
    if found and similarity is best among clusters:
      add e to c, recompute centroid by bundling all member keys (majority vote).
    else:
      seed a new cluster from e.

Bundle-of-bundles: each cluster centroid IS a bundle of its member keys; the consolidated
memory is then a memory of those centroids. This reduces N noisy episodic traces to K
semantic centroids while preserving associative recall.

The mathematical formulation: for cluster C with members {k_1, ..., k_n},

    centroid(C) = sign( Σ k_i )      (majority vote — same as bundle_all)

For bipolar ±1 vectors this is the maximum-likelihood estimate of the latent prototype
under independent symmetric noise. Cosine similarity to the centroid concentrates as
n grows (Johnson-Lindenstrauss-style concentration in 10k-D).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from kohaku._index import RetrievalIndex
from kohaku._pure import EpisodicMemory, HyperVector


@dataclass
class Cluster:
    centroid_key: HyperVector
    centroid_value: HyperVector
    member_ids: List[int] = field(default_factory=list)
    label: str = ""

    @property
    def size(self) -> int:
        return len(self.member_ids)


def consolidate(
    memory: EpisodicMemory,
    similarity_threshold: float = 0.3,
) -> List[Cluster]:
    """Cluster *memory* entries by key cosine similarity.

    Parameters
    ----------
    memory:
        Source episodic memory. Not mutated.
    similarity_threshold:
        Minimum cosine similarity to merge an entry into an existing cluster.
        Lower → fewer, larger clusters. Higher → more, smaller clusters.

    Returns
    -------
    List of :class:`Cluster`, in seed-insertion order.
    """
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")

    clusters: List[Cluster] = []
    # Track raw member key/value vectors per cluster to recompute centroids
    member_keys: List[List[HyperVector]] = []
    member_vals: List[List[HyperVector]] = []

    for entry in memory.entries():
        best_idx: int = -1
        if clusters:
            # Batch the entry-vs-centroid scan: one packed pass over the current
            # centroids picks the nearest. Centroids mutate as clusters grow, so
            # the index is rebuilt per entry (cheap — bit-packing, not float MAC).
            centroids = np.stack([c.centroid_key.data for c in clusters])
            ranked = RetrievalIndex(centroids).topk(entry.key.data, 1)
            # Merge only on a strict improvement over the threshold, matching the
            # original `sim > best_sim` (top-1 tiebreak is lowest index, as before).
            if ranked and ranked[0][1] > similarity_threshold:
                best_idx = ranked[0][0]

        if best_idx == -1:
            # Seed a new cluster from this entry alone
            clusters.append(
                Cluster(
                    centroid_key=entry.key,
                    centroid_value=entry.value,
                    member_ids=[entry.id],
                    label=entry.label,
                )
            )
            member_keys.append([entry.key])
            member_vals.append([entry.value])
        else:
            member_keys[best_idx].append(entry.key)
            member_vals[best_idx].append(entry.value)
            clusters[best_idx].member_ids.append(entry.id)
            clusters[best_idx].centroid_key = HyperVector.bundle_all(member_keys[best_idx])
            clusters[best_idx].centroid_value = HyperVector.bundle_all(member_vals[best_idx])

    return clusters


def consolidate_to_memory(
    memory: EpisodicMemory,
    similarity_threshold: float = 0.3,
    capacity: int | None = None,
) -> EpisodicMemory:
    """Run :func:`consolidate` and return a fresh EpisodicMemory of the centroids.

    The label of each consolidated entry is ``f"{seed_label} (n={size})"`` so consumers
    can see how much was merged.
    """
    clusters = consolidate(memory, similarity_threshold)
    cap = capacity if capacity is not None else max(1, len(clusters))
    out = EpisodicMemory(capacity=cap)
    for c in clusters:
        out.store(c.centroid_key, c.centroid_value, f"{c.label} (n={c.size})")
    return out
