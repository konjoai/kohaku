#!/usr/bin/env python3
"""ANN-narrowed re-rank benchmark (C1 follow-up: unify ANN + RetrievalIndex).

`EnrichedMemoryStore.query` now packs and scores *only* the ANN candidate rows
when `candidate_ids` is supplied, instead of scoring the whole store. This
measures that win in-process: full exact scan vs an ANN-narrowed query whose
candidate set is a small fraction of the store. Both return exact-cosine
rankings over their scored set — the index never invents a match.

Follows the repo benchmarking rules (≥5 warmups, p50/p95/p99, never overwrites).
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
from kohaku._pure import HyperVector  # noqa: E402
from kohaku.enriched import EnrichedMemoryStore  # noqa: E402

WARMUP = 5
RUNS = 20
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


def _store(n, dims, seed):
    rng = np.random.default_rng(seed)
    store = EnrichedMemoryStore()
    ids = []
    for i in range(n):
        bits = np.where(rng.random(dims) > 0.5, np.int8(1), np.int8(-1))
        hv = HyperVector(bits)
        ids.append(store.store(hv, hv, label=f"e{i}"))
    return store, ids


def run(sizes, dims, cand_frac):
    rows = []
    for n in sizes:
        store, ids = _store(n, dims, seed=n)
        probe = HyperVector(np.where(np.random.default_rng(n + 1).random(dims) > 0.5,
                                     np.int8(1), np.int8(-1)))
        k = max(1, int(n * cand_frac))
        cand = set(ids[:k])  # ANN would hand us a small candidate set
        full = _bench(lambda: store.query(probe, top_k=TOP_K, reinforce_hits=False))
        narrowed = _bench(
            lambda: store.query(probe, top_k=TOP_K, candidate_ids=cand, reinforce_hits=False)
        )
        rows.append({
            "n": n,
            "dims": dims,
            "candidates": k,
            "full_ms": full,
            "narrowed_ms": narrowed,
            "speedup_p50": round(full["p50"] / narrowed["p50"], 2),
        })
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    sizes = [1000, 5000] if quick else [1000, 5000, 20000]
    dims = 10_000
    cand_frac = 0.02  # ANN narrows to ~2% of the store
    rows = run(sizes, dims, cand_frac)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_ann_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {"warmup": WARMUP, "runs": RUNS, "top_k": TOP_K,
                   "dims": dims, "cand_frac": cand_frac},
        "has_rust": HAS_RUST,
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"facade query  has_rust={HAS_RUST}  dims={dims}  candidates≈{cand_frac:.0%}")
    header = "n        full_p50    narrowed_p50  speedup"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:<8} {r['full_ms']['p50']:<11} "
            f"{r['narrowed_ms']['p50']:<13} {r['speedup_p50']:<6}x"
        )
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
