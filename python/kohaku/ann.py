"""Approximate nearest-neighbour retrieval via bipolar LSH (B2).

Brute-force cosine retrieval is ``O(N·D)`` per query — fine to ~10⁴ memories,
a wall beyond that. This module adds an optional locality-sensitive hashing
index that narrows each query to a small candidate set before exact ranking,
turning the common case sub-linear while keeping exact cosine as the
correctness baseline (the index never *invents* a match — it only proposes
candidates that are then scored exactly).

The scheme is SimHash / random-hyperplane LSH: each of ``num_tables`` tables
hashes a vector to a ``hash_bits``-bit bucket via the signs of fixed random
projections. Vectors close in cosine collide in at least one table with high
probability; more tables raise recall, more bits raise precision.

Pure NumPy — no FAISS/hnswlib dependency, so it runs everywhere the rest of
kohaku does. Drop it into the facade with ``Memory(ann=True)`` or use it
directly:

    >>> from kohaku.ann import LSHIndex
    >>> idx = LSHIndex(dims=10_000)
    >>> idx.add(1, hv_a); idx.add(2, hv_b)            # doctest: +SKIP
    >>> idx.query(hv_query, top_k=5)                  # doctest: +SKIP
    [(2, 0.83), (1, 0.41)]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from kohaku._pure import DIMS, EpisodicMemory, HyperVector

# Defaults favour recall over precision: because every candidate is re-ranked
# with exact cosine, extra candidates are cheap, but a missed bucket is a lost
# result. More tables + fewer bits per table widen recall (≈0.9 at 5% noise).
DEFAULT_NUM_TABLES = 16
DEFAULT_HASH_BITS = 12
DEFAULT_SEED = 0x1071_5EED


def _as_float_vector(vector: "HyperVector | np.ndarray") -> "np.ndarray":
    data = vector.data if isinstance(vector, HyperVector) else vector
    return np.asarray(data, dtype=np.float32).reshape(-1)


class LSHIndex:
    """Random-hyperplane LSH over bipolar hypervectors.

    Parameters
    ----------
    dims:
        Hypervector dimensionality.
    num_tables:
        Independent hash tables. More tables → higher recall, more memory.
    hash_bits:
        Bits per bucket key (≤ 63). More bits → finer buckets, higher
        precision but lower recall per table.
    seed:
        Seed for the fixed projection planes (keep stable across runs).
    """

    def __init__(
        self,
        dims: int = DIMS,
        *,
        num_tables: int = DEFAULT_NUM_TABLES,
        hash_bits: int = DEFAULT_HASH_BITS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if not 0 < hash_bits <= 63:
            raise ValueError("hash_bits must be in 1..63")
        if num_tables <= 0:
            raise ValueError("num_tables must be > 0")
        self.dims = dims
        self.num_tables = num_tables
        self.hash_bits = hash_bits
        rng = np.random.default_rng(seed)
        # Planes: (num_tables, hash_bits, dims). Fixed for the index lifetime.
        self._planes = rng.standard_normal((num_tables, hash_bits, dims)).astype(np.float32)
        self._powers = (1 << np.arange(hash_bits, dtype=np.int64))
        self._tables: List[Dict[int, List[int]]] = [{} for _ in range(num_tables)]
        self._vectors: Dict[int, "np.ndarray"] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, entry_id: int) -> bool:
        return entry_id in self._vectors

    def _bucket_keys(self, vec: "np.ndarray") -> "np.ndarray":
        # (num_tables, hash_bits) projections → sign bits → packed int per table.
        projections = self._planes @ vec  # (num_tables, hash_bits)
        bits = (projections >= 0.0).astype(np.int64)
        return bits @ self._powers  # (num_tables,)

    def add(self, entry_id: int, vector: "HyperVector | np.ndarray") -> None:
        """Index a vector under ``entry_id`` (replacing any prior entry)."""
        vec = _as_float_vector(vector)
        if vec.shape[0] != self.dims:
            raise ValueError(f"vector dim {vec.shape[0]} != index dim {self.dims}")
        if entry_id in self._vectors:
            self.remove(entry_id)
        self._vectors[entry_id] = vec
        for table, key in zip(self._tables, self._bucket_keys(vec)):
            table.setdefault(int(key), []).append(entry_id)

    def remove(self, entry_id: int) -> bool:
        """Drop ``entry_id`` from the index. Returns False if it was absent."""
        vec = self._vectors.pop(entry_id, None)
        if vec is None:
            return False
        for table, key in zip(self._tables, self._bucket_keys(vec)):
            bucket = table.get(int(key))
            if bucket and entry_id in bucket:
                bucket.remove(entry_id)
                if not bucket:
                    del table[int(key)]
        return True

    def clear(self) -> None:
        self._tables = [{} for _ in range(self.num_tables)]
        self._vectors = {}

    def candidates(self, vector: "HyperVector | np.ndarray") -> set:
        """Union of entry ids sharing a bucket with ``vector`` in any table."""
        vec = _as_float_vector(vector)
        found: set = set()
        for table, key in zip(self._tables, self._bucket_keys(vec)):
            bucket = table.get(int(key))
            if bucket:
                found.update(bucket)
        return found

    def query(
        self, vector: "HyperVector | np.ndarray", top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Return up to ``top_k`` ``(entry_id, cosine)`` pairs, ranked exactly.

        Candidates come from the LSH buckets; ranking is exact cosine over that
        candidate set. Returns ``[]`` when no candidate shares a bucket.
        """
        if top_k <= 0:
            return []
        vec = _as_float_vector(vector)
        cand = self.candidates(vec)
        if not cand:
            return []
        denom = float(np.linalg.norm(vec)) or 1.0
        scored = [
            (cid, float(self._vectors[cid] @ vec) / (denom * (np.linalg.norm(self._vectors[cid]) or 1.0)))
            for cid in cand
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    @classmethod
    def from_memory(
        cls,
        memory: EpisodicMemory,
        *,
        num_tables: int = DEFAULT_NUM_TABLES,
        hash_bits: int = DEFAULT_HASH_BITS,
        seed: int = DEFAULT_SEED,
    ) -> "LSHIndex":
        """Build an index over every entry's key in an ``EpisodicMemory``."""
        entries = memory.entries()
        dims = entries[0].key.data.shape[0] if entries else DIMS
        idx = cls(dims, num_tables=num_tables, hash_bits=hash_bits, seed=seed)
        for e in entries:
            idx.add(e.id, e.key)
        return idx
