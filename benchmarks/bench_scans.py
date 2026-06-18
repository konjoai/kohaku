#!/usr/bin/env python3
"""All-pairs scan benchmark for Track C1 slice 3.

Slice 3 ported the O(n²) similarity scans (uniqueness, duplicate, conflict)
from a Python double loop of ``HyperVector.cosine_similarity`` onto the resident
``RetrievalIndex`` (one packed index, one scored pass per pivot row). This
measures that swap in-process, both backends, per the repo benchmarking rules
(≥5 warmups, p50/p95/p99, results never overwritten).

Reference = the exact naive loop the slice replaced; "indexed" = the new path.
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

try:
    import kohaku  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku._accel import HAS_RUST  # noqa: E402
from kohaku._index import index_over  # noqa: E402
from kohaku._pure import EpisodicMemory, HyperVector  # noqa: E402

WARMUP = 5
RUNS = 15


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


def _memory(n, dims, seed):
    rng = np.random.default_rng(seed)
    mem = EpisodicMemory(capacity=n)
    for i in range(n):
        bits = np.where(rng.random(dims) > 0.5, np.int8(1), np.int8(-1))
        hv = HyperVector(bits)
        mem.store(hv, hv, f"e{i}")
    return mem


def _naive_uniqueness(entries):
    """The exact O(n²) Python loop slice 3 replaced (uniqueness signal)."""
    scores = {}
    for i, ei in enumerate(entries):
        best = -1.0
        for j, ej in enumerate(entries):
            if j == i:
                continue
            sim = float(ei.key.cosine_similarity(ej.key))
            if sim > best:
                best = sim
        scores[ei.id] = 1.0 - max(0.0, best)
    return scores


def _indexed_uniqueness(entries):
    """The slice-3 batched path."""
    idx = index_over(entries)
    scores = {}
    for i, ei in enumerate(entries):
        sims = idx.all_scores(ei.key.data)
        sims[i] = -1.0
        scores[ei.id] = 1.0 - max(0.0, float(sims.max()))
    return scores


def run(sizes, dims):
    rows = []
    for n in sizes:
        entries = _memory(n, dims, seed=n).entries()
        naive = _bench(lambda: _naive_uniqueness(entries))
        indexed = _bench(lambda: _indexed_uniqueness(entries))
        rows.append(
            {
                "n": n,
                "dims": dims,
                "naive_ms": naive,
                "indexed_ms": indexed,
                "speedup_p50": round(naive["p50"] / indexed["p50"], 2),
            }
        )
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    sizes = [200, 500] if quick else [200, 500, 1000]
    dims = 10_000
    rows = run(sizes, dims)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {"warmup": WARMUP, "runs": RUNS, "dims": dims, "scan": "uniqueness"},
        "has_rust": HAS_RUST,
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"all-pairs uniqueness scan  has_rust={HAS_RUST}  dims={dims}")
    header = "n        naive_p50   indexed_p50  speedup"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:<8} {r['naive_ms']['p50']:<11} "
            f"{r['indexed_ms']['p50']:<12} {r['speedup_p50']:<6}x"
        )
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
