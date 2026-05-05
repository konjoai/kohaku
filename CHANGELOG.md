# Changelog

All notable changes to Kohaku are documented here.

## [0.5.0] — 2026-05-03

### Added
- `python/kohaku/learning.py` — online HDC item memory. `ItemMemory(dims)` maps `label → Prototype` with float32 accumulator + binarized `vector` on demand (`sign(accumulator)`, ties → +1). API: `add(label, vector, weight=1.0)`, `update(label, vector, sign=±1, weight=1.0)`, `train_from_feedback(label, vector, correct, weight=1.0)`, `predict(vector, top_k)`, `get(label)`, `labels()`, `clear()`, `__contains__`, `__len__`. Empirical: 12 noisy examples at 20% bit-flips recover the latent prototype with cosine > 0.95.
- `python/kohaku/hopfield.py` — modern continuous Hopfield retrieval (Ramsauer et al. 2020, *"Hopfield Networks is All You Need"*). `HopfieldAssociator(beta=0.05, binarize_each_step=True, dims=DIMS)`. Storage is `O(N·D)` matrix of stored patterns — at D=10 000 that's 40 KB per pattern instead of the 400 MB weight matrix the classical formulation requires. `recall(query, max_iters, eps)` runs `p* = softmax(β·X·q)·X` iteratively until normalized-state change ≤ eps; returns `HopfieldRecall(pattern, iterations, converged, weights, best_index, best_similarity)`. `complete()` for one-shot pattern completion. Empirical: 30%-flipped queries against 5 stored patterns recover the correct pattern with softmax weight > 0.9 in ≤ 3 iterations.
- `python/kohaku/memory_system.py` — combined episodic + semantic store, modeled on Tulving (1972). `MemorySystem(episodic_capacity, dims, decay_config)` holds an `EpisodicMemory` (decay-eligible) and an `ItemMemory` (semantic prototypes, no decay). `store_episode()` adds raw experiences; `reinforce_concept()` / `teach()` write directly to semantic memory. `consolidate_to_semantic(similarity_threshold)` runs cluster promotion: each episodic cluster's centroid is pushed to semantic memory weighted by cluster size — the "sleep consolidation" operation. `recall(query, top_k, use_decay, decay_config)` queries both stores, returning a merged ranked list of `CombinedRecall(source, label, similarity, entry_id, value)` with `source ∈ {"episodic", "semantic"}`.
- `python/kohaku/__init__.py` — exports `ItemMemory`, `Prototype`, `HopfieldAssociator`, `HopfieldRecall`, `MemorySystem`, `CombinedRecall`. Version bumped to `0.5.0`.
- `python/pyproject.toml` — version bumped to `0.5.0`.
- `python/tests/test_learning.py` (11 tests), `python/tests/test_hopfield.py` (13 tests), `python/tests/test_memory_system.py` (11 tests). Coverage: empty / boundary / dims-validation / sign / weight / convergence / cleanup / consolidation / decay-merge / sources.
- Total test count: **108 passed** (38 new + 70 prior; pre-existing skip resolved).

### Notes
- The Hopfield β default of 0.05 is calibrated for D=10 000 bipolar vectors so a single positive cosine of 0.5 dominates the softmax (β·D·cos = 250 → effectively a hard winner). Override `beta` for smaller dims.
- Online learning prototypes are exact bipolar after binarization — no precision drift after `update()`. Float32 accumulators support up to ~2²⁴ examples per label without overflow concerns.

## [0.4.0] — 2026-05-02

### Added
- `python/kohaku/persistence.py` — disk persistence in two formats. **JSON** (`save_json` / `load_json`): human-readable round-trip, stores ±1 components as ints. **Binary `.hkb`** (`save_binary` / `load_binary`): packed-bit format with `KHKU` magic, little-endian header (version, dims, capacity, next_id, timestamp, num_entries) and per-entry (id, timestamp, label_len, UTF-8 label, packed key bits, packed value bits) using `numpy.packbits` (1 bit per ±1 component, big-endian within byte, padded to multiple of 8). ~10x smaller than JSON in practice. `save()` / `load()` dispatch by file extension. Round-trip preserves IDs, timestamps, labels, capacity, and the memory's internal `_next_id`/`_timestamp` counters; recall behavior is bit-identical post-roundtrip.
- `python/kohaku/consolidation.py` — semantic clustering via bundle-of-bundles. Greedy single-pass `consolidate(memory, similarity_threshold=0.3)` returns `Cluster` records (`centroid_key`, `centroid_value`, `member_ids`, `label`, `size`); each entry joins the cluster with the highest centroid cosine similarity ≥ threshold (else seeds a new cluster). Centroids are recomputed by `bundle_all` (majority-vote) over all member keys/values on every merge — this is the maximum-likelihood prototype under independent symmetric bit-flip noise. `consolidate_to_memory()` returns a fresh `EpisodicMemory` of centroids labeled `"<seed_label> (n=<size>)"`.
- `python/kohaku/decay.py` — Ebbinghaus-style forgetting curves. `DecayConfig(half_life, floor=0.0)` validated at construction; `decay_weight(age, config)` = `max(0.5 ** (age / half_life), floor)`. `query_with_decay(memory, query_key, top_k, config)` is a drop-in replacement for `query()`: computes `decayed_sim = raw_sim * weight(age)` where `age = (memory._timestamp - 1) - entry.timestamp`. Preserves sign of similarity; sort is by decayed value descending.
- `python/kohaku/__init__.py` — exports `save`, `load`, `save_json`, `load_json`, `save_binary`, `load_binary`, `Cluster`, `consolidate`, `consolidate_to_memory`, `DecayConfig`, `decay_weight`, `query_with_decay`. Version bumped to `0.4.0`.
- `python/tests/test_persistence.py` — 12 tests: JSON round-trip, JSON readability, binary round-trip, magic header, binary-smaller-than-JSON, bad magic rejected, truncated file rejected, extension-dispatch, unknown extension raises, empty memory round-trip, Unicode labels, query-result preservation post-roundtrip.
- `python/tests/test_consolidation.py` — 7 tests: orthogonal entries stay separate, noisy variants merge, centroid concentration vs single members, capacity/labels of consolidated memory, empty memory, threshold validation, member ID tracking.
- `python/tests/test_decay.py` — 12 tests: weight at age 0 / half-life / 2x half-life, floor clamping, half-life validation, floor validation, negative age rejection, recent-beats-old recall, no-decay limit, empty memory, top_k=0, default config.
- Total test count: 69 passed, 1 skipped (pre-existing) — 38 prior + 31 new.

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
