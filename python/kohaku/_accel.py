"""Optional Rust acceleration for the cosine top-k hot loop.

Pure-Python (NumPy) is the correctness baseline. When the compiled
``kohaku._kohaku_rs`` extension is present, the batched cosine top-k runs in
Rust (bit-packed XOR + popcount over bipolar vectors). Both paths return
identical rankings for ±1 inputs — proven by the parity tests in
``test_accel.py``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from kohaku import _kohaku_rs as _rs

    HAS_RUST = True
except ImportError:  # pragma: no cover - exercised only without the built ext
    _rs = None
    HAS_RUST = False


def cosine_topk(
    query: np.ndarray,
    keys: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    """Return ``top_k`` ``(row_index, cosine)`` pairs over ``keys``.

    Parameters
    ----------
    query:
        1-D bipolar (±1) array of length ``D``.
    keys:
        2-D array shaped ``(N, D)`` of bipolar key vectors.
    top_k:
        Number of results to return.

    Ranking is by cosine descending, ties broken by ascending row index —
    matching the pure-Python ``HyperVector.cosine_similarity`` ordering.

    The batch path uses NumPy (``asarray`` + BLAS), which is currently faster
    than the Rust kernel: routing a batch through PyO3 marshals a Python
    list-of-lists per call, and that overhead dwarfs the bit-packed popcount win
    (measured ~3× slower — see ``benchmarks/bench_backends.py``). The Rust
    kernel stays built, parity-tested, and reachable via :func:`rust_cosine_topk`;
    it will become the default once slice 2 lands zero-copy NumPy FFI.
    """
    n = len(keys)
    if top_k <= 0 or n == 0:
        return []
    return _numpy_cosine_topk(query, keys, top_k)


def rust_cosine_topk(
    query: np.ndarray, keys: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    """Cosine top-k via the Rust bit-packed popcount kernel.

    Requires the compiled extension (:data:`HAS_RUST`). Returns the same
    ranking as :func:`cosine_topk`; kept available for parity testing and for
    the zero-copy re-wiring planned in slice 2.
    """
    if not HAS_RUST:
        raise RuntimeError("Rust extension not available")
    if top_k <= 0 or len(keys) == 0:
        return []
    ranked = _rs.cosine_topk(
        np.asarray(query, dtype=np.int8).tolist(),
        np.asarray(keys, dtype=np.int8).tolist(),
        top_k,
    )
    return [(int(i), float(s)) for i, s in ranked]


def cosine_all(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Cosine of ``query`` against every row of ``keys``, in row order.

    Uses the same kernel as :func:`cosine_topk`, so callers that need *all*
    similarities (e.g. temporal decay re-ranking) agree bit-for-bit with the
    top-k path on whichever backend is active.
    """
    n = len(keys)
    sims = np.zeros(n, dtype=np.float32)
    for i, s in cosine_topk(query, keys, n):
        sims[i] = s
    return sims


def _numpy_cosine_topk(
    query: np.ndarray, keys: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    q = np.asarray(query, dtype=np.float32)
    mat = np.asarray(keys, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    kn = np.linalg.norm(mat, axis=1)
    denom = kn * qn
    dots = mat @ q
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(denom > 1e-8, dots / denom, 0.0).astype(np.float32)
    # Stable sort on the negative similarity → descending, ties keep ascending index.
    order = np.argsort(-sims, kind="stable")[:top_k]
    return [(int(i), float(sims[i])) for i in order]
