"""Tests for the cosine-top-k accelerator (C1).

The NumPy path is always exercised. When the Rust extension is built, a parity
test asserts it agrees with NumPy exactly for bipolar inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from kohaku import _BACKEND
from kohaku._accel import HAS_RUST, cosine_topk, rust_cosine_topk
from kohaku._accel import _numpy_cosine_topk
from kohaku._pure import HyperVector


def _bipolar(n, dims, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((n, dims)) > 0.5, np.int8(1), np.int8(-1))


def test_empty_and_zero_k():
    keys = _bipolar(3, 64, 0)
    assert cosine_topk(np.ones(64, dtype=np.int8), keys, 0) == []
    assert cosine_topk(np.ones(64, dtype=np.int8), keys[:0], 5) == []


def test_self_match_is_one():
    keys = _bipolar(5, 128, 1)
    out = cosine_topk(keys[2], keys, 5)
    assert out[0][0] == 2
    assert out[0][1] == pytest.approx(1.0, abs=1e-6)


def test_matches_pure_python_cosine():
    keys = _bipolar(20, 256, 2)
    q = keys[7]
    out = cosine_topk(q, keys, 20)
    qhv = HyperVector(q)
    for idx, sim in out:
        expected = HyperVector(keys[idx]).cosine_similarity(qhv)
        assert sim == pytest.approx(expected, abs=1e-5)


def test_descending_with_index_tiebreak():
    # rows 0 and 3 identical to the query → both cosine 1.0, index 0 first.
    q = np.array([1, 1, 1, 1], dtype=np.int8)
    keys = np.array(
        [[1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, -1, -1], [1, 1, 1, 1]],
        dtype=np.int8,
    )
    out = cosine_topk(q, keys, 4)
    assert [i for i, _ in out] == [0, 3, 2, 1]


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")
def test_rust_matches_numpy_exactly():
    keys = _bipolar(50, 512, 3)
    for qi in (0, 11, 49):
        q = keys[qi]
        rust = rust_cosine_topk(q, keys, 10)
        numpy = _numpy_cosine_topk(q, keys, 10)
        assert [i for i, _ in rust] == [i for i, _ in numpy]
        for (_, rs), (_, ns) in zip(rust, numpy):
            assert rs == pytest.approx(ns, abs=1e-5)


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")
def test_backend_flag_reports_rust():
    assert _BACKEND == "rust-accel"
    from kohaku import _kohaku_rs

    # Zero-copy FFI: query and keys are contiguous int8 arrays, not lists.
    q = np.array([1, -1], dtype=np.int8)
    keys = np.array([[1, -1]], dtype=np.int8)
    assert _kohaku_rs.cosine_topk(q, keys, 1)[0][1] == pytest.approx(1.0)


@pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")
def test_cosine_topk_rejects_dim_mismatch():
    from kohaku import _kohaku_rs

    q = np.array([1, -1, 1], dtype=np.int8)
    keys = np.array([[1, -1]], dtype=np.int8)  # dims 2 != query 3
    with pytest.raises(ValueError):
        _kohaku_rs.cosine_topk(q, keys, 1)
