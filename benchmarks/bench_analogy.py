#!/usr/bin/env python3
"""Analogical-memory capacity benchmark (Track D).

VSA superposition is lossy: a record holds only so many bound (attribute, value)
pairs before cleanup starts missing. This measures, honestly, how attribute-query
and analogical-transfer accuracy fall off as records get wider, and how both
scale with dimensionality — so users know the operating envelope.

Accuracy is exact (does cleanup return the planted answer?), averaged over many
randomly-generated vocabularies. No latency claims here; this is a correctness/
capacity curve, written under benchmarks/results/ and never overwritten.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import kohaku  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku.analogy import AnalogicalMemory  # noqa: E402

TRIALS = 200


def _trial(n_attrs: int, dims: int, trial: int) -> tuple[bool, bool, float, float]:
    """One random two-record world; returns (get_ok, analogy_ok, get_conf, ana_conf)."""
    attrs = [f"attr{i}" for i in range(n_attrs)]
    a_vals = [f"a{i}_{trial}" for i in range(n_attrs)]
    b_vals = [f"b{i}_{trial}" for i in range(n_attrs)]
    m = AnalogicalMemory(dims=dims)
    m.add_record("A", dict(zip(attrs, a_vals)))
    m.add_record("B", dict(zip(attrs, b_vals)))
    probe = trial % n_attrs

    g = m.get("A", attrs[probe])
    a = m.analogy("A", "B", a_vals[probe])
    return (
        g.value == a_vals[probe],
        a.value == b_vals[probe],
        g.confidence,
        a.confidence,
    )


def run(attr_counts, dims):
    rows = []
    for n in attr_counts:
        gets = anas = 0
        gconf = aconf = 0.0
        for t in range(TRIALS):
            g_ok, a_ok, gc, ac = _trial(n, dims, t)
            gets += g_ok
            anas += a_ok
            gconf += gc
            aconf += ac
        rows.append({
            "attrs_per_record": n,
            "dims": dims,
            "get_accuracy": round(gets / TRIALS, 3),
            "analogy_accuracy": round(anas / TRIALS, 3),
            "mean_get_confidence": round(gconf / TRIALS, 3),
            "mean_analogy_confidence": round(aconf / TRIALS, 3),
        })
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    attr_counts = [3, 6] if quick else [3, 6, 10, 16, 24, 40]
    dims = 10_000
    rows = run(attr_counts, dims)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_analogy"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {"trials": TRIALS, "dims": dims},
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"analogical memory capacity  dims={dims}  trials={TRIALS}")
    header = "attrs  get_acc  analogy_acc  get_conf  analogy_conf"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['attrs_per_record']:<6} {r['get_accuracy']:<8} "
            f"{r['analogy_accuracy']:<12} {r['mean_get_confidence']:<9} "
            f"{r['mean_analogy_confidence']}"
        )
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
