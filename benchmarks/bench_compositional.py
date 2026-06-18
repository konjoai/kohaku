#!/usr/bin/env python3
"""Robust-recall benchmark (Track D3): does Hopfield cleanup help under noise?

A cue is rarely a perfect copy of a stored memory. This measures hit@1 — does
the nearest stored key to the cue equal the intended target? — as the cue is
corrupted by bit flips, with and without Hopfield pattern-completion, in two
regimes:

* **orthogonal** — random, well-separated memories. Here cosine is already
  near-perfect even at heavy noise (10k-D gives a huge margin), so cleanup has
  nothing to add. Honest result: don't pay the O(N·D) cleanup cost here.
* **clustered** — memories that are noisy variants of a few centroids (highly
  correlated). This is where a corrupted cue can land on the wrong sibling and
  where cleanup's behaviour actually matters.

Vector-level (no encoder) so noise is controlled exactly. Averaged over many
random stores; written under benchmarks/results/, never overwritten.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import kohaku  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kohaku._index import RetrievalIndex  # noqa: E402
from kohaku._pure import HyperVector  # noqa: E402
from kohaku.compositional import complete_cue  # noqa: E402

TRIALS = 200
N_KEYS = 50
DIMS = 10_000


def _flip(data: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    out = data.copy()
    n = int(len(out) * frac)
    idx = rng.choice(len(out), size=n, replace=False)
    out[idx] *= -1
    return out


def _bipolar(rng: np.random.Generator) -> np.ndarray:
    return np.where(rng.random(DIMS) > 0.5, np.int8(1), np.int8(-1)).astype(np.int8)


def _keys(regime: str, rng: np.random.Generator):
    if regime == "orthogonal":
        return [HyperVector(_bipolar(rng)) for _ in range(N_KEYS)]
    # clustered: noisy variants of a handful of centroids (highly correlated)
    n_clusters = 5
    centroids = [_bipolar(rng) for _ in range(n_clusters)]
    keys = []
    for i in range(N_KEYS):
        base = centroids[i % n_clusters]
        keys.append(HyperVector(_flip(base, 0.12, rng)))
    return keys


def run(regime: str, noise_levels):
    rows = []
    for frac in noise_levels:
        base_hits = clean_hits = 0
        for t in range(TRIALS):
            rng = np.random.default_rng(t)
            keys = _keys(regime, rng)
            index = RetrievalIndex(np.stack([k.data for k in keys]))
            target = t % N_KEYS
            noisy = _flip(keys[target].data, frac, rng)

            base = index.topk(noisy, 1)[0][0]
            cleaned = complete_cue(HyperVector(noisy), keys)
            clean = index.topk(cleaned.data, 1)[0][0]
            base_hits += base == target
            clean_hits += clean == target
        rows.append({
            "regime": regime,
            "noise": frac,
            "baseline_hit@1": round(base_hits / TRIALS, 3),
            "cleanup_hit@1": round(clean_hits / TRIALS, 3),
        })
    return rows


def main() -> None:
    quick = "--quick" in sys.argv
    noise_levels = [0.2, 0.4] if quick else [0.1, 0.2, 0.3, 0.4, 0.45, 0.48]
    rows = run("orthogonal", noise_levels) + run("clustered", noise_levels)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{stamp}_compositional"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": stamp,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {"trials": TRIALS, "n_keys": N_KEYS, "dims": DIMS},
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"robust recall  dims={DIMS}  keys={N_KEYS}  trials={TRIALS}")
    header = "regime       noise   baseline_hit@1   cleanup_hit@1"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['regime']:<12} {r['noise']:<7} {r['baseline_hit@1']:<16} {r['cleanup_hit@1']}")
    print(f"\nwrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
