"""Tests for kohaku.attention — attention_weighted_encode and encode_text."""

from __future__ import annotations

import pytest
from kohaku.attention import attention_weighted_encode, encode_text
from kohaku._pure import DIMS


# ---------------------------------------------------------------------------
# 1. Uniform weights == encode_text (cosine similarity > 0.99)
# ---------------------------------------------------------------------------


def test_uniform_weights_matches_encode_text():
    """attention_weighted_encode with uniform weights must match encode_text closely."""
    tokens = ["the", "quick", "brown", "fox"]
    uniform = [1.0] * len(tokens)
    hv_weighted = attention_weighted_encode(tokens, uniform)
    hv_encode = encode_text(" ".join(tokens))
    sim = hv_weighted.cosine_similarity(hv_encode)
    assert sim > 0.99, (
        f"Uniform-weighted encode should match encode_text, got sim={sim:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Higher weight on a token → result more similar to that token's vector
# ---------------------------------------------------------------------------


def test_high_weight_token_dominates():
    """The high-weight token's hypervector should be more similar to the result."""
    from kohaku.attention import _token_to_hv

    tokens = ["alpha", "beta", "gamma"]
    # Give 'alpha' a very high weight
    weights_alpha = [10.0, 1.0, 1.0]
    weights_gamma = [1.0, 1.0, 10.0]

    hv_alpha = attention_weighted_encode(tokens, weights_alpha)
    hv_gamma = attention_weighted_encode(tokens, weights_gamma)

    alpha_hv = _token_to_hv("alpha", DIMS)

    # Result weighted toward 'alpha' should be more similar to alpha than gamma result
    sim_alpha_to_alpha = hv_alpha.cosine_similarity(alpha_hv)
    sim_gamma_to_alpha = hv_gamma.cosine_similarity(alpha_hv)
    assert sim_alpha_to_alpha > sim_gamma_to_alpha, (
        f"High weight on alpha should yield higher similarity to alpha HV: "
        f"alpha={sim_alpha_to_alpha:.4f}, gamma={sim_gamma_to_alpha:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Output is a valid HyperVector (all values ±1)
# ---------------------------------------------------------------------------


def test_output_is_valid_bipolar_hypervector():
    """Result of attention_weighted_encode must have all values in {+1, -1}."""
    tokens = ["neural", "network", "memory"]
    weights = [0.5, 0.3, 0.2]
    hv = attention_weighted_encode(tokens, weights)
    unique = set(hv.data.tolist())
    assert unique <= {1, -1}, f"HyperVector must be bipolar ±1, got values: {unique}"
    assert len(hv) == DIMS


# ---------------------------------------------------------------------------
# 4. Empty weight list raises ValueError
# ---------------------------------------------------------------------------


def test_empty_weights_raises_value_error():
    """attention_weighted_encode with empty tokens/weights must raise ValueError."""
    with pytest.raises(ValueError):
        attention_weighted_encode([], [])

    with pytest.raises(ValueError):
        attention_weighted_encode(["word"], [])

    with pytest.raises(ValueError):
        attention_weighted_encode([], [1.0])
