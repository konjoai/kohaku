# Kohaku — Development Plan

## Current Version: v0.10.0

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

## Phase 10: Memory Graph Export (v0.9.0) ✅
- [x] `python/kohaku/graph_export.py` — `GraphExportConfig`, `MemoryNode`, `MemoryEdge`, `MemoryGraph`, `MemoryGraphExporter`. Exports episodic + semantic memory as a graph (nodes = memory entries, edges = cosine similarity >= threshold). Cosine computed in FP32 via `np.einsum`. Pairwise O(N²) with `max_nodes` safety cap. Atomic writes via `.tmp` + `os.replace`.
- [x] `MemoryGraph.to_json()` — JSON serialisation. `MemoryGraph.to_gexf()` — GEXF 1.3 XML with node attribute declarations (source, timestamp, decay_weight, cluster_id).
- [x] `MemoryGraphExporter.save_json()` / `save_gexf()` / `save()` — dispatch by file extension (.json, .gexf). Raises `ValueError` on unknown extension.
- [x] `api/main.py` — `GET /export/graph?threshold=0.3` (JSON) and `GET /export/graph/gexf?threshold=0.3` (application/xml) over the live REST state.
- [x] `python/kohaku/cli.py` — `kohaku export --format json|gexf --threshold 0.3 --out graph.json --from FILE` subcommand.
- [x] `__init__.py` exports `GraphExportConfig`, `MemoryGraphExporter`, `MemoryGraph`, `MemoryNode`, `MemoryEdge`. Version bumped to `0.9.0`.
- [x] 20 new tests in `python/tests/test_graph_export.py`; 197 tests total (python/tests/).

## Researched Feature Roadmap

Plan-of-record for the next set of capabilities, derived from a survey of
production HDC / agentic-memory systems. Items are ordered by their priority
band and dependency depth, not necessarily by implementation order.

### 🔴 P1 — Critical (this sprint)

- **Temporal validity intervals** — each memory item carries `valid_from: datetime` and `valid_until: Optional[datetime]`. Retrieval automatically filters items where `now > valid_until` (and skips items where `now < valid_from`, i.e. future-dated facts). Kills the stale-memory failure mode. Encoded into hypervector context bits in a later wave; for v0.10.0 the filter lives in the metadata layer.
  - `store(text, valid_until=datetime(2026, 1, 1))`
  - `query()` skips expired items by default; opt-in `include_expired=True`.
- **Salience scoring** — each memory carries a composite score: `salience = importance × recency_decay × (1 + reinforcement_count · k) × trust(source)`. `importance` set at store time (0.0-1.0). `reinforcement_count` increments on each retrieval hit. Retrieval can re-rank by salience instead of raw cosine. `GET /memories?sort=salience`.
- **Memory provenance + source tagging** — every stored memory has `source: str` (e.g. `user_input`, `web_search`, `tool_result`, `agent_inference`). `GET /memories?source=web_search` filters by provenance. Poisoning defense: `agent_inference` gets `trust=0.5` by default, dampening its salience. Trust weights are tunable via `SOURCE_TRUST_WEIGHTS`.
- **Sleep-phase consolidation daemon** — background thread runs every `consolidation_interval_minutes` (default 60). Finds episodic clusters with pairwise cosine ≥ 0.85, merges them into a semantic prototype via centroid bundling. Returns / logs a structured `SleepReport{episodes_consolidated, prototypes_created, memory_freed, run_seconds}`. `POST /consolidate` triggers a run on demand.

### 🟠 P2 — High Impact / Medium Complexity (next sprint)

- **Single-shot episodic binding (who / what / when / where)** — `store_episode(who=hv_agent, what=hv_action, when=hv_time, where=hv_context)` produces a composite HV via element-wise bind (XOR/multiply). Retrieval from any partial cue: `query_episode(what=hv_action)` returns the full bound episode. Uniquely native to HDC; out of reach for vector-DB-backed memory.
- **Multi-hop associative chaining** — `chain_query(start, hops=3)` iteratively retrieves the top match for `start`, then for that match, etc. Returns the hop chain with per-step similarity. Relational queries across the memory graph.
- **Write-time validation + poisoning defense** — before storing: (1) semantic-coherence check (cosine to existing similar items — flag contradictions), (2) per-source rate limit (e.g. ≤100 `agent_inference` stores/min), (3) novelty threshold (reject near-duplicate where cosine > 0.99). `POST /memories/validate` dry-run endpoint.
- **Memory graph export (Graphiti-compatible)** — already partially landed via Phase 10 (`MemoryGraphExporter`). Next: add Graphiti / Mem0-shaped node + edge JSON dialects so external graph tools can import without translation.

### 🟡 P3 — Strategic (later)

- **Neuromorphic spike encoding** — optional mode encoding HDC vectors as spike trains for neuromorphic hardware (Intel Loihi 2). Spike density = vector magnitude; refractory period encodes binarisation. Gated behind a `--neuromorphic` feature flag.
- **Cross-agent memory sharing** — shared memory pool with per-agent write namespaces and read-all semantics. Composes with the existing `TenantMemoryStore` (which is the dual: isolated read/write per tenant).
- **Forgetting-curve fine-tuning** — per-memory `forgetting_rate` override that bypasses the Ebbinghaus default. High-priority memories decay slower (`half_life ∝ importance`).

## Phase 11: Critical P1 Features (v0.10.0) ✅
- [x] `python/kohaku/enriched.py` — `MemoryMetadata` dataclass with `valid_from`, `valid_until`, `source`, `importance`, `reinforcement_count`. `SOURCE_TRUST_WEIGHTS` dict (user_input=1.0, tool_result=0.9, web_search=0.8, agent_inference=0.5). `EnrichedMemoryStore` wraps `EpisodicMemory` and a parallel `{entry_id: MemoryMetadata}` table. `store(..., source, importance, valid_from, valid_until)`, `query(..., sort='similarity'|'salience', source_filter, include_expired)`, `list_memories(sort, source_filter, limit)`, `reinforce(entry_id)`, `expire_old(now)`. Salience formula: `importance × decay_weight(age_days, half_life) × (1 + reinforcement_count · 0.1) × trust(source)`. Expired items skipped by default.
- [x] `python/kohaku/sleep.py` — `SleepConsolidator` background thread. `consolidation_interval_minutes`, `similarity_threshold=0.85`. Each run: find clusters via the existing `consolidation.consolidate`, merge episodic entries into semantic prototypes, log structured `SleepReport(episodes_consolidated, prototypes_created, memory_freed, run_seconds, started_at)`. Manual `run_once()` and context-manager lifecycle. Thread-safe via `threading.Lock`. Optional `on_report` callback for external observability.
- [x] `api/main.py` — `GET /memories?sort=salience&source=web_search&limit=10` lists enriched memories with full metadata. `POST /memories/store` accepts the enriched fields. `POST /memories/query` retrieves with `sort` + `source_filter` + `include_expired`. `POST /consolidate` triggers a one-shot sleep-phase consolidation run and returns the `SleepReport`. New `EnrichedRestState` on the unified app.
- [x] Tests: `python/tests/test_enriched.py` (validity filter, salience re-ranking, source filter, trust weights, reinforcement, expire_old), `python/tests/test_sleep.py` (manual run, threshold gating, structured report, callback, lifecycle), plus integration tests in `api/test_api.py` for the new endpoints.
- [x] `__init__.py` exports `MemoryMetadata`, `EnrichedMemoryStore`, `EnrichedRetrievalResult`, `SOURCE_TRUST_WEIGHTS`, `SleepConsolidator`, `SleepReport`. Version bumped to `0.10.0`.

## Phase 12: Provenance, Time-Range, Health (v0.10.x) ✅
Three P2 features that turn the kohaku store into an observable, debuggable production system.

- [x] `python/kohaku/provenance.py` — SQLite-backed DAG of memory lineage. `ProvenanceGraph.record(memory_id, parent_ids, source_type, metadata)` upserts a row; `get_ancestors` / `get_descendants` BFS with `max_depth`; `get_full_graph` returns a `ProvenanceGraphResult` with deduped edges + union nodes. Thread-safe (RLock). Persists across processes. `EnrichedMemoryStore.store(..., parent_ids=...)` auto-records when a `ProvenanceGraph` is attached (`EnrichedMemoryStore(provenance=pg)`), and `record_consolidation()` is the hook for sleep-phase merges.
- [x] `python/kohaku/time_filter.py` — `TimeFilter(valid_after, valid_before)` with interval-overlap semantics ("memory was *known* during ``[a, b]``"). `from_iso()` accepts ISO 8601 with `Z` suffix. `apply_time_filter(memories, tf)` filters dataclass/dict iterables. `bucket_timeline(memories, start, end, bucket="hour|day|week|month")` returns `TimelineBucket` rows including empty intervals between `start` and `end`. `filter_recent(memories, since_hours, limit)` sorts most-recent-first.
- [x] `python/kohaku/memory_health.py` — `MemoryHealthAnalyzer.compute()` returns `MemoryHealthReport(total_memories, stale_memories, expired_memories, orphaned_memories, duplicate_candidates: List[DuplicatePair], storage_bytes, avg_access_frequency, salience_buckets[5], health_score in [0,1], recommendations)`. Staleness = age (from `valid_from`) ≥ `stale_days` AND `reinforcement_count == 0`. Duplicates = O(n²) cosine ≥ 0.95 over bipolar key vectors. Orphans = entries present in the store but missing from the attached `ProvenanceGraph`. `list_stale()` / `delete_stale(dry_run=...)` for the cleanup workflow.
- [x] `api/main.py` — 7 new endpoints on the unified app:
  - `GET /memories/{id}/provenance?direction=ancestors|descendants|both&max_depth=5`
  - `GET /memories/search?q=&valid_after=&valid_before=&source=&sort=salience|recency|similarity&limit=`
  - `GET /memories/timeline?start=&end=&bucket=day&preview_per_bucket=5`
  - `GET /memories/recent?limit=20&since_hours=24`
  - `GET /memories/health?stale_days=30&duplicate_threshold=0.95`
  - `GET /memories/health/stale?days=30`
  - `DELETE /memories/stale?days=30&dry_run=true`
- [x] Tests: 45 new (`test_provenance.py` 14, `test_time_filter.py` 16, `test_memory_health.py` 15). Total **327 passed**.
- [x] `__init__.py` exports `ProvenanceGraph`, `ProvenanceNode`, `ProvenanceGraphResult`, `TimeFilter`, `TimelineBucket`, `apply_time_filter`, `bucket_timeline`, `filter_recent`, `MemoryHealthAnalyzer`, `MemoryHealthReport`, `DuplicatePair`, `StaleMemory`.

## Phase 13: P2 Features — Episodic Binding, Chaining, Validation (v0.11.0) ✅

- [x] `python/kohaku/episode.py` — `EpisodeStore` with role-binding. `store_episode(label, *, who, what, when, where)` binds provided role HVs into a composite via `bundle(bind(R_role, value_hv), ...)`. Fixed deterministic role HVs (`_ROLE_SEEDS`) so any two stores over the same dims share the same role space. `query_episode(*, who, what, when, where, top_k)` retrieves from any partial cue. `unbind_role(entry_id, role)` returns the original HV. 17 unit tests in `python/tests/test_episode.py`.
- [x] `python/kohaku/chaining.py` — `chain_query(memory, start_key, hops, min_similarity)` iteratively follows the highest-similarity unvisited entry's key HV. Returns `ChainResult(hops: List[HopResult], terminated_early)` with `.labels()` and `.similarities()` helpers. Terminates early on empty memory, no unvisited candidates, or `similarity < min_similarity`. 14 unit tests in `python/tests/test_chaining.py`.
- [x] `python/kohaku/validation.py` — `WriteValidator(memory, duplicate_threshold, rate_limits)` with two gates: (1) novelty — reject if nearest cosine >= threshold; (2) rate limit — per-source sliding-window deque. `validate(key_hv, source)` is read-only; `record(source)` commits the slot; `validate_and_store(...)` does both atomically. `RateLimit(max_stores, window_seconds)` validated at construction. 17 unit tests in `python/tests/test_validation.py`.
- [x] `api/main.py` — 4 new endpoints: `POST /episodes/store`, `POST /episodes/query`, `POST /chain`, `POST /memories/validate`. `RestState` gains `episodes: EpisodeStore` and `validator: WriteValidator` (pre-configured with `agent_inference` rate limit of 100/min).
- [x] `__init__.py` exports `EpisodeStore`, `EpisodeRoles`, `EpisodeResult`, `chain_query`, `ChainResult`, `HopResult`, `WriteValidator`, `RateLimit`, `ValidationResult`. Version bumped to `0.11.0`.
- [x] 46 new tests (17 episode + 14 chaining + 17 validation — 2 from chaining/validation consolidated = 46 net). Total **404 passed**.

## Phase 14: Tags, Conflicts, Portability (v0.11.x) ✅

Three orthogonal P2 features that complete the curatorial layer:

- [x] **Tagging** — `MemoryMetadata.tags: set[str]` with lowercase + 64-char normalisation; empty/whitespace tags dropped. `EnrichedMemoryStore.store(..., tags=[...])` accepts tags at write time. `add_tags()` / `remove_tags()` / `get_tags()` / `all_tags()` (returns count by tag). `query()` and `list_memories()` accept `tags_any` (any-match) and `tags_all` (all-match) filters. Tags surface in `list_memories()` dicts and `EnrichedRetrievalResult.tags`.
- [x] `python/kohaku/conflicts.py` — `detect_conflicts(store, similarity_threshold=0.40, contradiction_threshold=0.45, max_pairs=100)` scans all pairs for contradiction signals. Score in [0, 1] composed from four independent signals: shared topic (gate, weight 0.40), predicate divergence via Jaccard on same subject anchor (0.15), polarity flip via negation marker (0.30), numeric divergence (0.15). Returns `ConflictPair(a_id, b_id, similarity, contradiction_score, reasons)` sorted by score desc. `resolve_conflict(store, a_id, b_id, keep="a"|"b"|"both"|"dismiss")` applies the decision; drops the loser from episodic memory + metadata + provenance.
- [x] `python/kohaku/portability.py` — `export_memories(store, fmt="json"|"markdown"|"csv")` returns `ExportBundle(format, payload, memory_count, tag_count)`. JSON preserves full metadata + tags + provenance edges (when a graph is attached). Markdown is human-readable. CSV pipes tags with `|`. `import_memories(store, payload, dedup_threshold=0.99)` parses JSON, re-encodes labels via `encode_text`, and skips entries whose closest existing match is ≥ threshold. Returns `ImportReport(imported, skipped_duplicates, skipped_invalid, new_ids, duplicate_of)`.
- [x] `api/main.py` — 6 new endpoints:
  - `GET  /memories/tags` — tag → count index across all live memories.
  - `GET  /memories/{id}/tags`, `POST /memories/{id}/tags`, `DELETE /memories/{id}/tags?tag=a,b`.
  - `GET  /memories/conflicts?similarity_threshold=0.40&contradiction_threshold=0.45&max_pairs=100`.
  - `POST /memories/conflicts/resolve` with `{a_id, b_id, keep}`.
  - `GET  /memories/export?format=json|markdown|csv`.
  - `POST /memories/import` with `{payload: str, dedup_threshold: 0.99}`.
  - `GET /memories` now accepts `?tags=…&tags_all=…` comma-separated filters; `POST /memories/store` and `POST /memories/query` accept `tags` / `tags_any` / `tags_all`.
- [x] Tests: **37 new** (11 `test_tags.py` + 12 `test_conflicts.py` + 14 `test_portability.py`). Total **410 passed**.
- [x] `__init__.py` exports `ConflictPair`, `ConflictResolution`, `detect_conflicts`, `resolve_conflict`, `ExportBundle`, `ImportReport`, `export_memories`, `export_json`, `export_markdown`, `export_csv`, `import_memories`, `import_iter`.
