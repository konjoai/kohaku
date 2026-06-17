#!/usr/bin/env python3
"""Rust vs NumPy kernel benchmark for cosine top-k (Track C1 evidence).

Measures the two ``kohaku._accel`` backends head-to-head *in the same process*
so the comparison is apples-to-apples. Follows the repo benchmarking rules:
≥5 warmup runs, p50/p95/p99 + stddev reported, results written under
``benchmarks/results/<timestamp>_backends/`` (never overwritten).

Requires the Rust extension to be built (``pip install .``); otherwise only the
NumPy path is timed and the speedup column is blank.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Prefer an installed (possibly Rust-accelerated) kohaku; only fall back to the
# source tree when the package isn't installed. Inserting python/ unconditionally
# would shadow the compiled extension and force the NumPy path.
try:
    import kohaku  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku._accel import HAS_RUST, _numpy_cosine_topk, rust_cosine_topk  # noqa: E402
from kohaku import RetrievalIndex  # noqa: E402

WARMUP = 5
RUNS = 30
TOP_K = 10


def _percentiles(samples_ms):
    s = sorted(samples_ms)
    return {
        "p50": round(statistics.median(s), 4),
        "p95": round(s[int(len(s) * 0.95) - 1], 4),
        "p99": round(s[int(len(s) * 0.99) - 1], 4),
        "stddev": round(statistics.pstdev(s), 4),
    }


def _bench(fn) -> dict:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(RUNS):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return _percentiles(samples)


def _bipolar(n, dims, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((n, dims)) > 0.5, np.int8(1), np.int8(-1))


def run(sizes, dims):
    """Three paths, in-process: NumPy one-shot baseline; Rust zero-copy one-shot
    (slice-2 FFI, re-packs keys each call); resident RetrievalIndex (slice-2
    packed index, keys packed once then queried repeatedly — the realistic
    repeated-probe workload, build cost excluded as it amortizes)."""
    rows = []
    for n in sizes:
        keys = _bipolar(n, dims, seed=n)
        query = keys[0]
        numpy_stats = _bench(lambda: _numpy_cosine_topk(query, keys, TOP_K))
        row = {"n": n, "dims": dims, "numpy_ms": numpy_stats}
        if HAS_RUST:
            zc_stats = _bench(lambda: rust_cosine_topk(query, keys, TOP_K))
            idx = RetrievalIndex(keys)
            index_stats = _bench(lambda: idx.topk(query, TOP_K))
            row["rust_zerocopy_ms"] = zc_stats
            row["rust_index_ms"] = index_stats
            row["zerocopy_speedup_p50"] = round(numpy_stats["p50"] / zc_stats["p50"], 2)
            row["index_speedup_p50"] = round(numpy_stats["p50"] / index_stats["p50"], 2)
        rows.append(row)
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    sizes = [1000, 5000] if quick else [1000, 10000, 50000]
    dims = 10_000
    rows = run(sizes, dims)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_backends"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {"warmup": WARMUP, "runs": RUNS, "top_k": TOP_K, "dims": dims},
        "has_rust": HAS_RUST,
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"backend has_rust={HAS_RUST}  dims={dims}  warmup={WARMUP} runs={RUNS}")
    header = "n        numpy_p50  zerocopy_p50  zc_x    index_p50  index_x"
    print(header)
    print("-" * len(header))
    for r in rows:
        nps = r["numpy_ms"]
        line = f"{r['n']:<8} {nps['p50']:<10}"
        zc = r.get("rust_zerocopy_ms")
        idx = r.get("rust_index_ms")
        if zc and idx:
            line += (
                f" {zc['p50']:<13} {r['zerocopy_speedup_p50']:<7}x"
                f" {idx['p50']:<10} {r['index_speedup_p50']:<6}x"
            )
        print(line)
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
