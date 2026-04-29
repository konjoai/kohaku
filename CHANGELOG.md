# Changelog

All notable changes to Kohaku are documented here.

## [0.3.0] — 2026-04-28

### Added
- `python/kohaku/context.py` — `ContextConfig` dataclass (max_tokens=4096, tokens_per_entry=50, top_k=5, similarity_threshold=0.1) and `ContextMemoryManager`: sliding-window episodic store sized to LLM context limit. Text→hypervector encoding is deterministic via LCG character hashing matching the core HDC engine. Provides `store(key, value, label)`, `retrieve(query_text, top_k)`, `build_context_block(query_text)`, `capacity()`, `utilization()`.
- `python/kohaku/attention.py` — `attention_weighted_encode(tokens, weights, dims)`: bundle token hypervectors weighted by normalized attention scores with binarized output. `encode_text(text, dims)`: uniform-weighted convenience wrapper for whitespace-split tokens. Both functions are deterministic via the same LCG path as `_pure.py`.
- `python/kohaku/hf_hooks.py` — `KohakuMemoryCallbackStub` (always importable, raises `ImportError` on instantiation when transformers is absent) and `KohakuMemoryCallback` (real `transformers.TrainerCallback` when transformers is installed): `on_step_end` stores mean attention or step counter; `on_log` stores training metrics. Module import never raises regardless of transformers availability.
- `python/kohaku/openai_compat.py` — `MemoryMiddleware`: `augment(messages)` finds the last user message, retrieves relevant memories from a `ContextMemoryManager`, and prepends a system message with `build_context_block` output. `learn_from_exchange(messages)` stores assistant responses keyed by the preceding user message. No external dependencies required.
- `python/kohaku/__init__.py` — exports `ContextConfig`, `ContextMemoryManager`, `attention_weighted_encode`, `encode_text`, `MemoryMiddleware`. Version bumped to `0.3.0`.
- `python/pyproject.toml` — version bumped to `0.3.0`.
- `python/tests/test_context.py` — 8 tests: store/retrieve, context block prefix, capacity ratio, utilization, semantic ranking, FIFO eviction, encoding determinism, config defaults.
- `python/tests/test_attention.py` — 4 tests: uniform==encode_text (cosine>0.99), high-weight token dominance, bipolar output contract, empty input raises ValueError.
- `python/tests/test_hf_hooks.py` — 4 tests: hf_hooks importable without transformers, stub raises ImportError on instantiation, openai_compat importable, MemoryMiddleware.augment returns list.
- Total test count: 39/39 passing (23 prior + 16 new).

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
