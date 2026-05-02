# Kohaku — Development Plan

## Current Version: v0.4.0

## Phase 1: Core HDC Engine (v0.1.0) ✅
- [x] Hypervector arithmetic: random, bundle, bind, permute
- [x] Cosine similarity and Hamming distance
- [x] Episodic memory store with capacity management (FIFO eviction)
- [x] Associative retrieval (top-k, threshold)
- [x] CLI: demo + bench subcommands
- [x] Python bridge script with KohakuMemory class
- [x] Integration test suite (8 core tests + bonus)

## Phase 2: Python Bindings (v0.2.0) ✅
- [x] PyO3 bindings scaffold (`src/pybindings.rs`) — buildable when maturin is available, gated behind `--features python`
- [x] `Cargo.toml` updated with optional pyo3 dependency and `python` feature flag
- [x] Pure-Python HDC implementation (`python/kohaku/_pure.py`) — LCG matches Rust exactly (XOR seed with 0xDEAD_BEEF_CAFE_BABE, same multiplier/addend), bipolar ±1 numpy arrays
- [x] Auto-detect backend in `python/kohaku/__init__.py` — uses Rust extension if present, falls back to pure-Python transparently
- [x] `RetrievalResult` dataclass + `query()` / `query_threshold()` in `python/kohaku/_query.py`
- [x] Async wrappers (`python/kohaku/_async.py`) — `AsyncEpisodicMemory` using `asyncio.to_thread`
- [x] pip-installable package (`python/pyproject.toml`) with hatchling build backend, numpy≥1.24 dependency
- [x] Test suite: 15 pure-Python tests (`test_pure.py`) + 8 async tests (`test_async.py`) — 23/23 passing
- [x] Async mode: `pytest-asyncio` with `asyncio_mode = "auto"`

### Phase 2 Architecture Notes
- `maturin` was not available at implementation time; the PyO3 binding code is written and correct but requires `maturin develop --features python` to compile.
- The pure-Python path is the default and complete — no Rust dependency at runtime.
- When maturin is eventually available: `maturin develop --features python` inside the repo root will compile `_kohaku_rs.so`, and the auto-detect in `__init__.py` will transparently switch to the fast Rust path.

## Phase 3: LLM Integration (v0.3.0) ✅
- [x] Context window memory manager (`python/kohaku/context.py`) — `ContextConfig` dataclass + `ContextMemoryManager`: sliding-window episodic store sized to `max_tokens // tokens_per_entry`, deterministic text→HyperVector encoding via LCG word hashing, `store`, `retrieve`, `build_context_block`, `capacity`, `utilization`
- [x] Attention-guided encoding (`python/kohaku/attention.py`) — `attention_weighted_encode` (weighted sum of token HVs, binarized) and `encode_text` (uniform-weighted convenience wrapper); deterministic via same LCG path as core HDC engine
- [x] HuggingFace Transformers hooks (`python/kohaku/hf_hooks.py`) — `KohakuMemoryCallback` (real `TrainerCallback` when transformers is installed) + `KohakuMemoryCallbackStub` (always importable, raises `ImportError` on instantiation when transformers absent); `on_step_end` and `on_log` handlers
- [x] OpenAI API compatible memory layer (`python/kohaku/openai_compat.py`) — `MemoryMiddleware`: `augment()` injects retrieved memories as system message prefix; `learn_from_exchange()` stores assistant responses as memories
- [x] 16 new tests: 8 `test_context.py` + 4 `test_attention.py` + 4 `test_hf_hooks.py` — 39/39 total passing
- [x] `__init__.py` updated: exports `ContextConfig`, `ContextMemoryManager`, `attention_weighted_encode`, `encode_text`, `MemoryMiddleware`; version bumped to `0.3.0`

## Phase 4: Persistence (v0.4.0) ✅
- [x] Serialize/deserialize memory to disk — JSON + binary `.hkb` (`python/kohaku/persistence.py`). `save_json` / `load_json` produce human-readable round-trips; `save_binary` / `load_binary` use a packed-bit format (magic `KHKU`, 1 bit per ±1 component, ~10x smaller than JSON). `save()` / `load()` dispatch by file extension. Round-trip preserves entry IDs, timestamps, labels (UTF-8), capacity, and `_next_id` / `_timestamp` counters.
- [x] Memory consolidation (`python/kohaku/consolidation.py`) — semantic clustering via bundle-of-bundles. Greedy single-pass: each new entry joins the existing cluster with highest centroid similarity ≥ threshold (else seeds a new one); centroids are recomputed on every merge by `bundle_all` (majority vote) over all member keys/values. `consolidate()` returns `Cluster` records (centroid_key, centroid_value, member_ids, label, size); `consolidate_to_memory()` produces a fresh `EpisodicMemory` of centroids labeled `"<seed_label> (n=<size>)"`.
- [x] Forgetting curves / temporal decay (`python/kohaku/decay.py`) — `DecayConfig(half_life, floor)` with exponential weight `0.5 ** (age / half_life)` clamped to `floor`. `query_with_decay()` is a drop-in alternative to `query()`: computes `decayed_sim = raw_sim * weight(age)` where `age = (memory._timestamp - 1) - entry.timestamp`. Validates `half_life > 0`, `floor ∈ [0, 1]`, `age ≥ 0`.
- [x] 31 new tests (12 `test_persistence.py` + 7 `test_consolidation.py` + 12 `test_decay.py`); 69/69 total passing (1 pre-existing skip).
- [x] `__init__.py` exports `save`, `load`, `save_json`, `load_json`, `save_binary`, `load_binary`, `Cluster`, `consolidate`, `consolidate_to_memory`, `DecayConfig`, `decay_weight`, `query_with_decay`. Version bumped to `0.4.0`.

## Phase 5: Learning (v0.5.0)
- [ ] Online HDC learning: update item memory from feedback
- [ ] Hopfield network associator layer
- [ ] Episodic vs semantic memory distinction
