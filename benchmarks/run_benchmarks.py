#!/usr/bin/env python3
"""Reproducible kohaku benchmarks (C3).

Measures the three things that matter for scaling — retrieval latency
(exact vs ANN), ANN recall@k, and on-disk size (.hkb vs JSON) — and prints a
table. The stable *invariants* derived from these live in
``python/tests/test_benchmarks.py`` and gate CI; this script is for humans
watching absolute numbers move.

    python benchmarks/run_benchmarks.py                 # default sweep
    python benchmarks/run_benchmarks.py --quick         # fast, for CI logs
    python benchmarks/run_benchmarks.py --json out.json # also write results
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku import Memory, save_binary, save_json  # noqa: E402
from kohaku._pure import DIMS, EpisodicMemory, HyperVector  # noqa: E402


def _time(fn, repeats: int) -> float:
    """Mean seconds per call over ``repeats`` runs."""
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def bench_retrieval(sizes, dims, queries):
    rows = []
    rng = np.random.default_rng(0)
    for n in sizes:
        exact = Memory(dims=dims, capacity=n + 1)
        approx = Memory(dims=dims, capacity=n + 1, ann=True)
        phrases = [
            f"concept {i} regarding subject {i % 13} and detail {i % 29}"
            for i in range(n)
        ]
        for p in phrases:
            exact.store(p)
            approx.store(p)
        probes = [phrases[int(rng.integers(0, n))] for _ in range(queries)]
        t_exact = (
            _time(lambda: [exact.query(p, reinforce=False) for p in probes], 1)
            / queries
        )
        t_ann = (
            _time(lambda: [approx.query(p, reinforce=False) for p in probes], 1)
            / queries
        )
        agree = sum(
            exact.query(p, top_k=1, reinforce=False)[0].text
            == approx.query(p, top_k=1, reinforce=False)[0].text
            for p in probes
        ) / len(probes)
        rows.append(
            {
                "n": n,
                "exact_ms": round(t_exact * 1e3, 3),
                "ann_ms": round(t_ann * 1e3, 3),
                "speedup": round(t_exact / t_ann, 2) if t_ann else float("inf"),
                "ann_top1_agreement": round(agree, 3),
            }
        )
    return rows


def bench_storage(sizes, dims):
    rows = []
    for n in sizes:
        mem = EpisodicMemory(capacity=n + 1)
        for i in range(n):
            hv = HyperVector.random(dims, seed=i)
            mem.store(hv, hv, f"memory {i}")
        with tempfile.TemporaryDirectory() as d:
            hkb = Path(d) / "m.hkb"
            js = Path(d) / "m.json"
            save_binary(mem, hkb)
            save_json(mem, js)
            rows.append(
                {
                    "n": n,
                    "hkb_kb": round(hkb.stat().st_size / 1024, 1),
                    "json_kb": round(js.stat().st_size / 1024, 1),
                    "ratio": round(js.stat().st_size / hkb.stat().st_size, 1),
                }
            )
    return rows


def _print_table(title, rows):
    print(f"\n{title}")
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="kohaku scaling benchmarks")
    parser.add_argument("--quick", action="store_true", help="small sizes for CI logs")
    parser.add_argument("--dims", type=int, default=DIMS)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument(
        "--json", type=str, default=None, help="write results to this path"
    )
    args = parser.parse_args()

    sizes = [100, 500, 1000] if args.quick else [500, 2000, 5000, 10000]
    dims = 2048 if args.quick else args.dims
    queries = 20 if args.quick else args.queries

    import kohaku

    print(f"kohaku {kohaku.__version__} — backend: {kohaku._BACKEND}")
    retrieval = bench_retrieval(sizes, dims, queries)
    storage = bench_storage(sizes, dims)
    _print_table(
        f"Retrieval latency (dims={dims}, {queries} queries) — exact vs ANN", retrieval
    )
    _print_table("On-disk size — .hkb (packed bits) vs JSON", storage)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"dims": dims, "retrieval": retrieval, "storage": storage}, indent=2
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
