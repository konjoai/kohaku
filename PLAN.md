# Kohaku — Development Plan

## Current Version: v0.8.0

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

## Phase 5: Learning (v0.5.0) ✅
- [x] Online HDC learning (`python/kohaku/learning.py`) — `ItemMemory` maps `label → Prototype`. Float32 accumulator per label; binarized prototype on demand via `sign(accumulator)`. `add()` / `update(sign=±1, weight=…)` / `train_from_feedback(correct=…)` keep the prototype always in {+1, −1}^D. `predict(top_k)` returns ranked `RetrievalResult`. Verified: 12 noisy examples (20% bit-flips) → recovered prototype > 0.95 cosine to the latent.
- [x] Hopfield network associator (`python/kohaku/hopfield.py`) — modern continuous Hopfield (Ramsauer et al. 2020). Storage `O(N·D)` (not `O(D²)`). Recall: `p* = softmax(β·X·q) · X`, iterated until L2-normalized state changes by ≤ `eps`. `binarize_each_step` keeps dynamics inside the hypercube. `recall()` returns `HopfieldRecall(pattern, iterations, converged, weights, best_index, best_similarity)`. Verified: 30%-flipped queries recover correct stored pattern with softmax weight > 0.9 in ≤ 3 iterations.
- [x] Episodic vs semantic distinction (`python/kohaku/memory_system.py`) — `MemorySystem` wraps an `EpisodicMemory` (raw, decay-eligible, capacity-limited) and an `ItemMemory` (semantic prototypes, no decay). `consolidate_to_semantic()` runs cluster-promotion: each episodic cluster's centroid is pushed to semantic memory weighted by cluster size — modeling sleep consolidation. `recall()` queries both stores, merges by similarity, tags each result with `source="episodic" | "semantic"`. Optional `use_decay=True` applies `query_with_decay` to the episodic side.
- [x] 38 new tests (11 `test_learning.py` + 13 `test_hopfield.py` + 11 `test_memory_system.py` + 3 hopfield extras); 108/108 total passing.
- [x] `__init__.py` exports `ItemMemory`, `Prototype`, `HopfieldAssociator`, `HopfieldRecall`, `MemorySystem`, `CombinedRecall`. Version bumped to `0.5.0`.

## Phase 6: Production Hardening (v0.6.0) ✅
- [x] Real-time streaming consolidation (background thread/asyncio)
- [x] Memory compaction & deduplication
- [x] Multi-tenant isolation for serving multiple users from one engine

## Phase 7: Visualization + REST API (v0.7.0) ✅
- [x] `api/main.py` — unified FastAPI app exposing both surfaces in one process:
      **viz** — `GET /viz/graph` (nodes + edges + cosine k-means cluster labels + per-node Ebbinghaus decay weight),
      `GET /viz/decay` (per-concept forgetting curves), `POST /viz/probe` (ranked neighbours),
      `GET /viz/memory_map.html` (serves the viewer);
      **REST** — `POST /encode`, `POST /store`, `POST /query`, `POST /bundle`,
      `GET /stats`, `GET /health`.
      Two states on one app: read-only `VizState` over `demo/sample_memory.json`,
      write-able `RestState` (EpisodicMemory + ItemMemory, `threading.Lock`-guarded)
      for the REST surface. `/query` accepts `half_life` / `floor` and returns
      `decayed_similarity` via `query_with_decay`. Raw float-vector inputs are
      sign-binarized at the API boundary so HDC ops always see ±1.
- [x] `demo/memory_map.html` — interactive d3-force-directed viewer (d3 v7 via CDN). Node radius = decay weight, colour = k-means cluster, edges = cosine ≥ slider threshold. Probe input animates dashed edges from the strongest-match node to the rest of the activated set. Live sliders for threshold / half-life / k.
- [x] `demo/sample_memory.json` — 12 concepts across 3 ground-truth clusters (animals / programming / cities). Within-cluster cosine ≥ 0.7, between-cluster ≤ 0.4 by construction, so k-means recovers the labels deterministically.
- [x] `api/requirements.txt` — fastapi, uvicorn[standard], pydantic v2, numpy, httpx.
- [x] `api/Dockerfile` — python:3.11-slim, `PYTHONPATH=/app/python`, uvicorn on `$PORT`.
- [x] `render.yaml` — Render.com web service spec, Docker env, `/health` healthcheck.
- [x] `api/test_viz.py` — 6 tests: graph contract & node-field invariants, edge-threshold subset relation, k-means cluster recovery, decay-curve shape, decay weight matches `decay_weight(age, cfg)` exactly, probe ranks the target cluster at the top.
- [x] `api/test_api.py` — 18 integration tests via `TestClient` for the REST surface (no mocks).

## Phase 9: kyro bridge + cosmos UI (v0.8.0) ✅
- [x] `python/kohaku/kyro_bridge.py` — `HDCRetriever` exposes a kyro-compatible RAG surface. `ingest(docs)` accepts strings or `{"text", "id"?}` dicts; `retrieve(query, top_k, half_life?, floor?)` returns `RetrievedChunk(entry_id, doc_id, text, similarity, decayed_similarity, age)`. Owns its own `EpisodicMemory` and a parallel `entry_id → (doc_id, text)` map (HVs are not invertible). 15 unit tests in `python/tests/test_kyro_bridge.py`.
- [x] `api/main.py` — `POST /bridge/ingest` and `POST /bridge/retrieve` on the same unified app. App state holds a separate `HDCRetriever` so RAG chunks never pollute `/store` + `/query`. 7 new TestClient tests.
- [x] `demo/memory_map.html` — full cosmos visualization. Stars = memories (brightness = Ebbinghaus decay, size = access count, colour = cluster), gravity drift along high-similarity links, traveling light dots on connections, query shockwaves with line-arc to top-k, time-dial scrub, constellation/trails/pulse toggles, drag-to-orbit, particle-converge birth animation. Browser HDC engine (DIMS=1024) ports the kohaku LCG path bit-exactly.
- [x] `demo/index.html` — full rebuild: black-sky landing with floating glass search; query blooms a probe star, top-5 lines, and cluster-coloured chips.
- [x] 182 tests total (22 new + 160 prior).

## Phase 7b: Standalone REST Server Module ✅
- [x] `python/kohaku/server.py` — standalone FastAPI app (separate from `api/main.py`) for embedding in other processes. `create_app(capacity, dim)` factory + `serve(host, port, capacity, dim)`. Endpoints: `POST /memory/store`, `POST /memory/query`, `DELETE /memory/clear`, `GET /memory/stats`, `GET /health`. Env-var overrides: `KOHAKU_CAPACITY`, `KOHAKU_DIM`.
- [x] `python/kohaku/cli.py` — `kohaku serve` subcommand (--host, --port, --capacity, --dim).
- [x] `pyproject.toml` optional `[api]` group: fastapi>=0.100.0, uvicorn[standard]>=0.22.0. `[project.scripts]` entry for `kohaku` CLI.
- [x] `__init__.py` exports `create_app`, `serve` guarded with try/except ImportError.
- [x] 20 new tests in `python/tests/test_server.py` using `fastapi.testclient.TestClient`.
- [x] 202 tests total (20 new + 182 prior).
