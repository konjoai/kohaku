#!/usr/bin/env python3
"""
Kohaku Python bridge.

Provides a KohakuMemory class that generates hypervectors client-side using
the same LCG formula as the Rust implementation, enabling interoperability
without requiring compiled bindings.

The Rust CLI binary is used for the demo subcommand. Quantized similarity
computations run in pure Python (lists of int, cosine via dot product).
"""

from __future__ import annotations

import subprocess

# ─── LCG constants (must match Rust implementation in src/hypervector.rs) ────
_LCG_MUL: int = 6_364_136_223_846_793_005
_LCG_ADD: int = 1_442_695_040_888_963_407
_MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
_SEED_XOR: int = 0xDEAD_BEEF_CAFE_BABE

DIMS: int = 10_000


def _lcg_next(state: int) -> int:
    """One step of the Knuth LCG; operates on 64-bit unsigned integers."""
    return (state * _LCG_MUL + _LCG_ADD) & _MASK64


def _random_hypervector(dims: int, seed: int) -> list[int]:
    """Generate a deterministic bipolar (+1/-1) hypervector from seed.

    Matches the Rust ``HyperVector::random`` implementation exactly.
    """
    state: int = (seed ^ _SEED_XOR) & _MASK64
    data: list[int] = []
    for _ in range(dims):
        state = _lcg_next(state)
        # High bit → +1/-1 (mirrors Rust: v >> 63 == 0 → +1, else -1)
        data.append(1 if (state >> 63) == 0 else -1)
    return data


def _cosine_similarity(a: list[int], b: list[int]) -> float:
    """Cosine similarity for bipolar ±1 vectors.

    For bipolar vectors |v|² = D, so cosine = dot(a, b) / D.
    """
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)


def _text_to_seed(text: str) -> int:
    """Map text to a deterministic 64-bit seed via Python's built-in hash.

    Note: Python's hash() is not stable across interpreter restarts (PYTHONHASHSEED).
    For reproducible seeds, use explicit integer seeds directly.
    """
    # Use a simple djb2-style hash for cross-run stability
    h: int = 5381
    for ch in text.encode("utf-8"):
        h = ((h << 5) + h + ch) & _MASK64
    return h


# ─── KohakuMemory ─────────────────────────────────────────────────────────────


class KohakuMemory:
    """Lightweight in-process episodic memory backed by HDC hypervectors.

    Stores experiences as bipolar ±1 hypervectors and retrieves them via
    cosine similarity without requiring the Rust binary.

    Example::

        mem = KohakuMemory(capacity=1000)
        mem.store("apple", "A crisp red fruit")
        mem.store("ocean", "Vast salt water body")
        results = mem.query("apple", top_k=3)
        for r in results:
            print(r["label"], r["similarity"])
    """

    def __init__(self, capacity: int = 1000, dims: int = DIMS) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = capacity
        self._dims = dims
        # Each entry: {"id": int, "label": str, "seed": int, "key": list[int], "value": list[int]}
        self._entries: list[dict] = []
        self._next_id: int = 1

    def store(self, label: str, text: str) -> int:
        """Store a labelled text experience as a hypervector.

        The key is derived from ``text``; the value encodes both label and text.
        Returns the assigned entry id.
        """
        key_seed = _text_to_seed(text)
        val_seed = _text_to_seed(label + "::" + text)

        if len(self._entries) == self._capacity:
            self._entries.pop(0)  # FIFO eviction

        entry_id = self._next_id
        self._next_id += 1

        self._entries.append(
            {
                "id": entry_id,
                "label": label,
                "text": text,
                "key": _random_hypervector(self._dims, key_seed),
                "value": _random_hypervector(self._dims, val_seed),
            }
        )
        return entry_id

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Retrieve the top_k most similar stored memories for the given text.

        Returns a list of dicts with keys: ``entry_id``, ``label``, ``similarity``.
        Results are sorted by descending similarity.
        """
        if top_k <= 0 or not self._entries:
            return []

        query_key = _random_hypervector(self._dims, _text_to_seed(text))

        scored = [
            {
                "entry_id": e["id"],
                "label": e["label"],
                "text": e["text"],
                "similarity": _cosine_similarity(query_key, e["key"]),
            }
            for e in self._entries
        ]
        scored.sort(key=lambda r: (-r["similarity"], r["entry_id"]))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"KohakuMemory(len={len(self)}, capacity={self._capacity}, dims={self._dims})"


# ─── Rust CLI bridge ──────────────────────────────────────────────────────────


def run_kohaku_cli(*args: str, timeout: int = 30) -> str:
    """Run the Kohaku Rust CLI binary and return stdout as a string.

    Looks for the binary in PATH, then in typical Cargo release/debug locations.

    Raises:
        FileNotFoundError: if the binary cannot be located.
        subprocess.CalledProcessError: if the binary exits with non-zero status.
        subprocess.TimeoutExpired: if the binary takes longer than ``timeout`` seconds.
    """
    import shutil
    import os

    binary_name = "kohaku"
    binary = shutil.which(binary_name)

    if binary is None:
        # Try Cargo output directories relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "target", "release", binary_name),
            os.path.join(script_dir, "..", "target", "debug", binary_name),
        ]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c):
                binary = c
                break

    if binary is None:
        raise FileNotFoundError(
            "kohaku binary not found. Run `cargo build --release` first."
        )

    result = subprocess.run(
        [binary] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def run_demo_via_cli() -> str:
    """Run ``kohaku demo`` via the CLI binary and return its output."""
    return run_kohaku_cli("demo")


# ─── __main__ demonstration ───────────────────────────────────────────────────


if __name__ == "__main__":
    print("Kohaku Python Bridge — KohakuMemory demonstration")
    print("=" * 52)
    print()

    mem = KohakuMemory(capacity=100, dims=DIMS)

    # Store 3 memories
    experiences = [
        ("apple", "A round red fruit with a crisp texture that grows on trees"),
        ("ocean", "A vast body of salt water covering most of the Earth surface"),
        (
            "library",
            "A building containing organized collections of books and knowledge",
        ),
    ]

    for label, text in experiences:
        eid = mem.store(label, text)
        print(f"  Stored [{eid:>2}]: {label!r}")

    print()
    print(f"  Memory: {mem}")
    print()

    # Query
    query_text = "A round red fruit with a crisp texture that grows on trees"
    print(f"  Query: {query_text!r}")
    print()
    results = mem.query(query_text, top_k=3)

    header = f"  {'ID':<4}  {'Label':<12}  {'Similarity':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['entry_id']:<4}  {r['label']:<12}  {r['similarity']:>12.6f}")

    print()
    if results:
        top = results[0]
        print(f"  → Best match: {top['label']!r} (sim={top['similarity']:.6f})")
    print()

    # Optionally invoke the Rust CLI demo
    print("  Attempting to run Rust CLI demo...")
    try:
        output = run_demo_via_cli()
        print(output)
    except FileNotFoundError as exc:
        print(f"  [Skipped — {exc}]")
    except subprocess.CalledProcessError as exc:
        print(f"  [CLI error: {exc}]")
