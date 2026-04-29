"""Attention-guided HDC encoding — weight token hypervectors by attention scores."""
from __future__ import annotations

import numpy as np

from kohaku._pure import (
    HyperVector,
    DIMS,
    _LCG_MUL,
    _LCG_ADD,
    _MASK64,
    _SEED_XOR,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_token(token: str) -> int:
    """Stable 64-bit hash for a token string."""
    h: int = 0
    for ch in token:
        h = (h * 31 + ord(ch)) & _MASK64
    return h


def _token_to_hv(token: str, dims: int) -> HyperVector:
    """Deterministically encode a single token as a bipolar hypervector via LCG."""
    seed = _hash_token(token) & _MASK64
    state = (seed ^ _SEED_XOR) & _MASK64
    bits = np.empty(dims, dtype=np.int8)
    for i in range(dims):
        state = (state * _LCG_MUL + _LCG_ADD) & _MASK64
        bits[i] = np.int8(1) if (state >> 63) == 0 else np.int8(-1)
    return HyperVector(bits)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attention_weighted_encode(
    tokens: list[str],
    weights: list[float],
    dims: int = DIMS,
) -> HyperVector:
    """Bundle token hypervectors weighted by attention scores.

    Parameters
    ----------
    tokens:
        List of token strings.  Must be non-empty and match length of *weights*.
    weights:
        Attention weights.  Normalized to sum=1 before use (simple softmax-free
        division — sufficient for weighting without distorting relative magnitudes).
        Must be non-empty and the same length as *tokens*.
    dims:
        Hypervector dimensionality (default 10 000).

    Returns
    -------
    HyperVector
        Attended hypervector with all values in {+1, -1}.

    Raises
    ------
    ValueError
        If *tokens* or *weights* are empty, or if their lengths differ, or if the
        weight sum is zero.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")
    if not weights:
        raise ValueError("weights must be non-empty")
    if len(tokens) != len(weights):
        raise ValueError(
            f"tokens and weights must have the same length, got {len(tokens)} vs {len(weights)}"
        )

    weight_arr = np.array(weights, dtype=np.float64)
    total = float(weight_arr.sum())
    if abs(total) < 1e-12:
        raise ValueError("weights sum to zero — cannot normalize")
    weight_arr = weight_arr / total  # normalize to sum=1

    # Weighted accumulation: sum_d = Σ w_i * hv_i[d]
    accumulator = np.zeros(dims, dtype=np.float64)
    for token, w in zip(tokens, weight_arr):
        hv = _token_to_hv(token, dims)
        accumulator += w * hv.data.astype(np.float64)

    # Binarize: sign(accumulator), tie-break to +1
    result = np.where(accumulator >= 0.0, np.int8(1), np.int8(-1)).astype(np.int8)
    return HyperVector(result)


def encode_text(text: str, dims: int = DIMS) -> HyperVector:
    """Uniform-weighted encode of whitespace-split tokens.

    Equivalent to ``attention_weighted_encode(tokens, [1.0]*len(tokens), dims)``.
    Returns a deterministic hypervector for empty text (seed=0).
    """
    tokens = text.split()
    if not tokens:
        return HyperVector.random(dims, seed=0)
    uniform = [1.0] * len(tokens)
    return attention_weighted_encode(tokens, uniform, dims=dims)
