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

    This one-shot path uses NumPy (``asarray`` + BLAS). Slice 2's zero-copy FFI
    (:func:`rust_cosine_topk`) brought the Rust one-shot kernel to ~parity, but
    re-packing keys every call is no clear win over BLAS — so NumPy stays the
    default here. The decisive ~200× speedup comes from amortizing the packing:
    use :class:`kohaku.RetrievalIndex` (resident bit-packed index) when the same
    keys are probed more than once. ``kohaku.query`` / ``query_with_decay``
    already route through it.
    """
    n = len(keys)
    if top_k <= 0 or n == 0:
        return []
    return _numpy_cosine_topk(query, keys, top_k)


def rust_cosine_topk(
    query: np.ndarray, keys: np.ndarray, top_k: int
) -> List[Tuple[int, float]]:
    """Cosine top-k via the Rust bit-packed popcount kernel (zero-copy FFI).

    Requires the compiled extension (:data:`HAS_RUST`). The query and key matrix
    are passed as contiguous ``int8`` arrays and borrowed directly by Rust —
    no per-element Python marshaling — so the popcount kernel finally beats the
    NumPy path (slice 1 measured the list-of-lists marshaling at ~4× slower).
    Returns the same ranking as :func:`cosine_topk`.
    """
    if not HAS_RUST:
        raise RuntimeError("Rust extension not available")
    if top_k <= 0 or len(keys) == 0:
        return []
    q = np.ascontiguousarray(query, dtype=np.int8)
    k = np.ascontiguousarray(keys, dtype=np.int8)
    ranked = _rs.cosine_topk(q, k, top_k)
    return [(int(i), float(s)) for i, s in ranked]


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
