"""Resident packed retrieval index — amortizes key packing across queries.

Slice-1 benchmarking showed the per-call PyO3 list marshaling, not the math,
was the bottleneck. Slice 2 fixes that two ways:

* **Zero-copy FFI** (``kohaku._accel.rust_cosine_topk``) hands Rust a borrowed
  ``int8`` array instead of a Python list-of-lists — the one-shot path is now
  ~parity with NumPy instead of ~4× slower.
* **Resident packing** (this module) packs the keys to one bit per component
  *once*; each subsequent query marshals only the probe across the boundary and
  costs ``n_rows · dims / 64`` XOR+popcount words. Measured ~200× over
  re-streaming every float through BLAS for the repeated-query workload.

Pure-Python (NumPy) stays the correctness baseline: without the extension the
index caches a contiguous ``float32`` matrix and uses the same NumPy kernel, so
rankings are identical either way (proven by the parity tests).
"""
from __future__ import annotations

import weakref
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kohaku._accel import HAS_RUST, _numpy_cosine_topk
from kohaku._pure import EpisodicMemory

if HAS_RUST:
    from kohaku import _kohaku_rs as _rs


class RetrievalIndex:
    """Cosine top-k over a fixed key matrix, packing the keys once.

    Build it over an ``(N, D)`` bipolar (±1) key matrix, then call
    :meth:`topk` / :meth:`all_scores` as many times as you like — the expensive
    bit-packing happens in ``__init__`` and is reused on every query. Use it
    whenever the same key set is probed more than once (retrieval loops,
    consolidation scans, batch evaluation).
    """

    __slots__ = ("_rust", "_mat", "_n")

    def __init__(self, keys: np.ndarray) -> None:
        mat = np.asarray(keys)
        self._n = int(mat.shape[0]) if mat.ndim == 2 else 0
        if HAS_RUST and self._n:
            packed = np.ascontiguousarray(mat, dtype=np.int8)
            self._rust = _rs.PackedIndex(packed)
            self._mat: Optional[np.ndarray] = None
        else:
            self._rust = None
            self._mat = np.ascontiguousarray(mat, dtype=np.float32) if self._n else None

    def __len__(self) -> int:
        return self._n

    def topk(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Return ``top_k`` ``(row_index, cosine)`` pairs, cosine descending.

        Ties break by ascending row index — identical ordering on both backends.
        """
        if top_k <= 0 or self._n == 0:
            return []
        if self._rust is not None:
            q = np.ascontiguousarray(query, dtype=np.int8)
            return [(int(i), float(s)) for i, s in self._rust.topk(q, top_k)]
        return _numpy_cosine_topk(query, self._mat, top_k)

    def all_pairs(self) -> np.ndarray:
        """Full symmetric ``(N, N)`` cosine matrix over every pair of rows.

        ``M[i, j]`` is the cosine of row ``i`` against row ``j``; the diagonal is
        ``1.0``. For all-pairs scans (conflict / duplicate detection) this is the
        batched dual of calling :meth:`all_scores` once per row — it collapses
        ``N`` FFI crossings and ``N`` sorts into a single packed-popcount pass in
        Rust, and a single BLAS ``MMᵀ`` on the NumPy baseline. Both backends agree
        bit-for-bit for ±1 inputs (proven by the parity tests).
        """
        if self._n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self._rust is not None:
            flat = self._rust.all_pairs()
            return np.asarray(flat, dtype=np.float32).reshape(self._n, self._n)
        # NumPy baseline: bipolar rows share norm √dims, so cosine = MMᵀ / dims.
        # Normalise per-row to stay exact even if a row isn't perfectly ±1.
        mat = self._mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = mat / norms
        sims = (unit @ unit.T).astype(np.float32)
        np.fill_diagonal(sims, 1.0)
        return sims

    def all_scores(self, query: np.ndarray) -> np.ndarray:
        """Cosine of ``query`` against every row, in row order.

        Agrees bit-for-bit with :meth:`topk` on whichever backend is active —
        used by temporal-decay re-ranking, which needs every similarity.
        """
        sims = np.zeros(self._n, dtype=np.float32)
        for i, s in self.topk(query, self._n):
            sims[i] = s
        return sims


# Per-memory index cache. Keyed weakly by the memory object so it never keeps a
# store alive; the value carries the memory's generation counter so a stale
# index is rebuilt the moment the memory changes.
_INDEX_CACHE: "weakref.WeakKeyDictionary[EpisodicMemory, Tuple[int, RetrievalIndex]]"
_INDEX_CACHE = weakref.WeakKeyDictionary()


def index_over(entries: Sequence) -> RetrievalIndex:
    """Build a one-off :class:`RetrievalIndex` over the keys of ``entries``.

    For all-pairs / batch similarity scans (consolidation, conflict, importance,
    duplicate detection) where the key set is fixed for the duration of the
    scan: pack once, then call ``all_scores`` / ``topk`` per row. Not cached —
    use :func:`index_for` for the repeated single-memory query path.
    """
    if not entries:
        return RetrievalIndex(np.empty((0, 0), dtype=np.int8))
    return RetrievalIndex(np.stack([e.key.data for e in entries]))


def index_for(memory: EpisodicMemory, entries: list) -> RetrievalIndex:
    """Return a :class:`RetrievalIndex` over ``entries``, rebuilt only on change.

    ``entries`` must be ``memory.entries()`` (passed in so callers that already
    fetched it don't pay for a second copy). Repeated queries against an
    unchanged memory reuse the packed index; any store or clear bumps the
    memory's ``_generation`` counter and invalidates it.
    """
    if not entries:
        return RetrievalIndex(np.empty((0, 0), dtype=np.int8))
    version = memory._generation
    cached = _INDEX_CACHE.get(memory)
    if cached is not None and cached[0] == version:
        return cached[1]
    idx = RetrievalIndex(np.stack([e.key.data for e in entries]))
    _INDEX_CACHE[memory] = (version, idx)
    return idx
