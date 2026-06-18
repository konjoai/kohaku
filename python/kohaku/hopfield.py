"""Hopfield network associator — modern continuous Hopfield retrieval.

Classical discrete Hopfield (Hopfield 1982) requires a D×D weight matrix. At
D=10 000 that is 100M floats — 400 MB. Useless at our scale.

Modern Hopfield networks (Ramsauer et al. 2020, "Hopfield Networks is All You
Need") replace the explicit weight matrix with an attention-style softmax over
stored patterns. Storage = O(N·D); recall is N·D matmul + softmax. Provably
exponential storage capacity in D and convergence to the closest stored
pattern in O(1) iterations for well-separated patterns.

Recall dynamics
---------------

Given stored patterns ``X ∈ R^{N×D}`` and query ``q ∈ R^D``:

    p* = softmax(β · X · q) · X            (one Hopfield update)

Iterate ``q ← p*`` for a few steps (usually 1–5 are enough). β is the inverse
temperature; higher β makes recall more selective. For bipolar ±1 patterns we
optionally binarize ``p*`` after each step to keep retrieval inside the
hypercube; the implementation provides both modes.

Why ship this on top of :func:`kohaku._query.query`?
    `query` returns the *closest* stored pattern — but its similarity is the raw
    cosine to a single noisy episodic trace. Hopfield recall *cleans* the query
    by averaging over stored patterns weighted by similarity, yielding a
    reconstructed pattern that converges to the true latent prototype. Useful
    for de-noising and for completing partial cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from kohaku._pure import DIMS, HyperVector


@dataclass
class HopfieldRecall:
    """Result of one :meth:`HopfieldAssociator.recall` call."""

    pattern: HyperVector  # final binarized state q*
    iterations: int  # how many updates ran
    converged: bool  # True if update delta ≤ eps
    weights: np.ndarray  # final softmax(β X q*) — length N
    best_index: int  # argmax of final weights
    best_similarity: float  # cosine(q*, X[best_index])


class HopfieldAssociator:
    """Modern continuous Hopfield retrieval over stored bipolar patterns.

    Parameters
    ----------
    beta:
        Inverse temperature for the softmax. Higher → more selective recall.
        Default 0.05 is calibrated for D=10 000 bipolar vectors so a single
        positive cosine of 0.5 dominates the softmax appropriately
        (β · D · cos = 0.05 · 10 000 · 0.5 = 250 → strong winner).
    binarize_each_step:
        If True (default), states are clipped to ±1 after every update —
        keeps the dynamics inside the hypercube and matches the discrete
        Hopfield convergence story. If False, states are left continuous
        until the final return.
    """

    def __init__(
        self,
        beta: float = 0.05,
        binarize_each_step: bool = True,
        dims: int = DIMS,
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if dims <= 0:
            raise ValueError("dims must be > 0")
        self.beta = float(beta)
        self.binarize_each_step = bool(binarize_each_step)
        self._dims = int(dims)
        # Stored patterns kept as one float32 matrix (N, D) for fast matmul.
        self._patterns: List[np.ndarray] = []
        self._labels: List[str] = []

    # ── basic accessors ────────────────────────────────────────────────────
    @property
    def dims(self) -> int:
        return self._dims

    def __len__(self) -> int:
        return len(self._patterns)

    @property
    def is_empty(self) -> bool:
        return not self._patterns

    def labels(self) -> List[str]:
        return list(self._labels)

    def clear(self) -> None:
        self._patterns.clear()
        self._labels.clear()

    # ── storage ────────────────────────────────────────────────────────────
    def store(self, vector: HyperVector, label: str = "") -> int:
        """Add a bipolar pattern. Returns its index."""
        if len(vector) != self._dims:
            raise ValueError(
                f"vector dims mismatch: expected {self._dims}, got {len(vector)}"
            )
        self._patterns.append(vector.data.astype(np.float32))
        self._labels.append(label)
        return len(self._patterns) - 1

    def store_many(
        self, vectors: List[HyperVector], labels: Optional[List[str]] = None
    ) -> None:
        if labels is None:
            labels = [""] * len(vectors)
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        for v, lab in zip(vectors, labels):
            self.store(v, lab)

    # ── recall ─────────────────────────────────────────────────────────────
    def recall(
        self,
        query: HyperVector,
        max_iters: int = 5,
        eps: float = 1e-3,
    ) -> HopfieldRecall:
        """Iteratively refine *query* toward the closest stored pattern.

        Each iteration computes ``p* = softmax(β · X q) · X``. By default the
        result is binarized to ±1 after each step. Iteration stops early when
        the L2-normalized state changes by less than ``eps``.

        Returns a :class:`HopfieldRecall` with the final pattern, weights, and
        convergence diagnostics.
        """
        if max_iters <= 0:
            raise ValueError("max_iters must be > 0")
        if self.is_empty:
            raise ValueError("Cannot recall from empty HopfieldAssociator")
        if len(query) != self._dims:
            raise ValueError(
                f"query dims mismatch: expected {self._dims}, got {len(query)}"
            )

        X = np.stack(self._patterns, axis=0)  # (N, D) float32
        q = query.data.astype(np.float32).copy()  # (D,)
        prev: Optional[np.ndarray] = None
        weights = np.zeros(X.shape[0], dtype=np.float32)
        converged = False
        n_iters = 0

        for step in range(max_iters):
            n_iters = step + 1
            scores = X @ q  # (N,) — dot products
            # numerically stable softmax
            scaled = self.beta * scores
            scaled -= scaled.max()
            w = np.exp(scaled, dtype=np.float64)
            w /= w.sum()
            weights = w.astype(np.float32)
            new_q = (weights @ X).astype(np.float32)  # (D,)
            if self.binarize_each_step:
                new_q = np.where(new_q >= 0.0, np.float32(1), np.float32(-1))
            # convergence: change in normalized state below eps
            if prev is not None:
                delta = float(
                    np.linalg.norm(new_q - prev) / max(1.0, np.linalg.norm(new_q))
                )
                if delta <= eps:
                    converged = True
                    q = new_q
                    break
            prev = new_q.copy()
            q = new_q

        # Always return a binarized HyperVector — even if continuous mode was used
        final_bits = np.where(q >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
        final_hv = HyperVector(final_bits)
        best_idx = int(np.argmax(weights))
        best_sim = float(
            final_hv.cosine_similarity(
                HyperVector(self._patterns[best_idx].astype(np.int8))
            )
        )
        return HopfieldRecall(
            pattern=final_hv,
            iterations=n_iters,
            converged=converged,
            weights=weights,
            best_index=best_idx,
            best_similarity=best_sim,
        )

    def complete(self, partial_cue: HyperVector, max_iters: int = 5) -> HyperVector:
        """Pattern-completion convenience: return only the cleaned pattern."""
        return self.recall(partial_cue, max_iters=max_iters).pattern
