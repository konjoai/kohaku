# Changelog

All notable changes to Kohaku are documented here.

## [0.2.0] — 2026-04-28

### Added
- `python/kohaku/` — pip-installable Python package (`pyproject.toml`, hatchling build backend, `numpy>=1.24` runtime dependency)
- `python/kohaku/_pure.py` — pure-Python HDC implementation using numpy: `HyperVector` (random, bundle, bundle_all, bind, permute, cosine_similarity, hamming_distance) and `EpisodicMemory` (store, clear, FIFO eviction). LCG matches Rust exactly: seed XOR `0xDEAD_BEEF_CAFE_BABE`, multiplier `6364136223846793005`, addend `1442695040888963407`, sign-bit extraction identical to Rust
- `python/kohaku/_query.py` — `RetrievalResult` frozen dataclass; `query()` (top-k descending) and `query_threshold()` retrieval functions
- `python/kohaku/_async.py` — `AsyncEpisodicMemory` with `asyncio.to_thread`-backed async wrappers for all memory operations
- `python/kohaku/__init__.py` — auto-detect backend: imports compiled `_kohaku_rs` Rust extension if present, transparently falls back to pure-Python `_pure` otherwise; exports `_BACKEND` string
- `src/pybindings.rs` — PyO3 binding scaffold for `PyHyperVector` and `PyEpisodicMemory`; gated behind `#[cfg(feature = "python")]`; buildable with `maturin develop --features python`
- `Cargo.toml` — optional `pyo3 = "0.21"` dependency and `[features] python = ["pyo3"]` feature gate
- `python/tests/test_pure.py` — 15 tests covering shape, bipolarity, determinism, orthogonality, self-similarity, bundle, bind round-trip, permute invertibility, memory store/eviction/query/threshold, and Hamming distance
- `python/tests/test_async.py` — 8 async tests covering store, query, empty memory, roundtrip, threshold, clear, len, and 10-way concurrent stores
- All 23 Python tests pass: `python3 -m pytest python/tests/ -v`

## [0.1.0] — 2026-04-28

### Added
- Core HDC engine: random generation (LCG-seeded bipolar ±1), bundle (majority vote), bind (element-wise multiply), permute (circular shift)
- Cosine similarity and Hamming distance metrics with mathematical invariant tests
- `EpisodicMemory` struct with FIFO capacity management
- Associative retrieval: top-k (sorted descending) and threshold-based query
- CLI binary with `demo` and `bench` subcommands (ASCII table output, no external table crate)
- Python bridge script (`python/kohaku.py`) with `KohakuMemory` class (LCG-compatible vector generation, subprocess CLI bridge)
- Integration test suite: 8 core tests + bonus threshold test covering orthogonality, similarity, bundle, permute, store/retrieve, top-k ordering, capacity eviction, bind round-trip
