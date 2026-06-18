"""Performance gates (C3).

These are *invariants*, not wall-clock thresholds — they catch scaling
regressions (ANN recall collapse, candidate-pruning loss, compression loss,
inexact round-trips) without being flaky on shared CI runners. The human-facing
timing benchmark lives in ``benchmarks/run_benchmarks.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from kohaku import LSHIndex, Memory, load_binary, save_binary, save_json
from kohaku._pure import EpisodicMemory, HyperVector

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "run_benchmarks.py"


def _planted_dataset(n: int, dims: int, n_queries: int, seed: int = 0):
    """Return (stored vectors by id, [(query, true_id)]) with planted neighbours."""
    rng = np.random.default_rng(seed)
    vectors = {i: HyperVector.random(dims, seed=1000 + i) for i in range(n)}
    queries = []
    for q in range(n_queries):
        target = int(rng.integers(0, n))
        data = vectors[target].data.copy()
        flip = rng.choice(dims, size=dims // 20, replace=False)  # ~5% noise
        data[flip] *= -1
        queries.append((HyperVector(data), target))
    return vectors, queries


def test_ann_recall_at_10_is_high():
    # Uses the library default index params — this gate locks them at a config
    # that achieves strong recall, not just precision.
    dims, n = 4096, 300
    vectors, queries = _planted_dataset(n, dims, n_queries=60, seed=1)
    idx = LSHIndex(dims)
    for cid, hv in vectors.items():
        idx.add(cid, hv)
    hits = 0
    for q, true_id in queries:
        top = [cid for cid, _ in idx.query(q, top_k=10)]
        if true_id in top:
            hits += 1
    recall = hits / len(queries)
    assert recall >= 0.8, f"ANN recall@10 regressed to {recall:.2f}"


def test_ann_prunes_candidate_set():
    # The whole point of the index: scan far fewer than N entries.
    dims, n = 4096, 400
    vectors, queries = _planted_dataset(n, dims, n_queries=20, seed=2)
    idx = LSHIndex(dims, num_tables=8, hash_bits=16)
    for cid, hv in vectors.items():
        idx.add(cid, hv)
    avg_candidates = np.mean([len(idx.candidates(q)) for q, _ in queries])
    assert avg_candidates < n * 0.5, (
        f"ANN scanned {avg_candidates:.0f}/{n} — pruning lost"
    )


def test_hkb_is_smaller_than_json(tmp_path):
    mem = EpisodicMemory(capacity=200)
    for i in range(100):
        hv = HyperVector.random(4096, seed=i)
        mem.store(hv, hv, f"memory number {i}")
    hkb = tmp_path / "m.hkb"
    js = tmp_path / "m.json"
    save_binary(mem, hkb)
    save_json(mem, js)
    assert hkb.stat().st_size < js.stat().st_size
    # Packed bits should be at least ~5x smaller than the int-list JSON.
    assert js.stat().st_size / hkb.stat().st_size > 5.0


def test_binary_roundtrip_recall_is_exact(tmp_path):
    mem = EpisodicMemory(capacity=50)
    for i in range(20):
        hv = HyperVector.random(2048, seed=i)
        mem.store(hv, hv, f"m{i}")
    path = tmp_path / "m.hkb"
    save_binary(mem, path)
    restored = load_binary(path)
    q = HyperVector.random(2048, seed=5)
    before = [(e.id, round(e.key.cosine_similarity(q), 6)) for e in mem.entries()]
    after = [(e.id, round(e.key.cosine_similarity(q), 6)) for e in restored.entries()]
    assert before == after


def test_ann_memory_matches_exact_top1():
    # End-to-end through the facade: ANN top-1 == exact top-1.
    exact = Memory(dims=4096)
    approx = Memory(dims=4096, ann=True)
    phrases = [f"distinct concept number {i} about topic {i % 7}" for i in range(80)]
    for p in phrases:
        exact.store(p)
        approx.store(p)
    agree = 0
    for p in phrases:
        e = exact.query(p, top_k=1, reinforce=False)[0].text
        a = approx.query(p, top_k=1, reinforce=False)[0].text
        agree += int(e == a)
    assert agree / len(phrases) >= 0.95


def test_benchmark_script_runs():
    # Smoke-test the reproducible bench so it can't silently break.
    spec = importlib.util.spec_from_file_location("run_benchmarks", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    retrieval = mod.bench_retrieval([50], dims=1024, queries=5)
    storage = mod.bench_storage([50], dims=1024)
    assert retrieval[0]["n"] == 50
    assert retrieval[0]["ann_top1_agreement"] >= 0.9
    assert storage[0]["ratio"] > 5.0
