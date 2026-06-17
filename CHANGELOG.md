# Changelog

All notable changes to Kohaku are documented here.

## [0.18.0] — 2026-06-17

### Added — Track C1 (commit to Rust): accelerated cosine top-k

The first slice of the Rust commitment — a real, built, tested accelerator,
with pure-Python as the correctness baseline (per CLAUDE.md).

- **Rust kernel** (`src/accel.rs`) — `cosine_topk` over bipolar vectors via
  bit-packed XOR + popcount (`cosine = 1 − 2·hamming/D`), the genuine HDC win
  over float multiply-accumulate. Exposed through PyO3 as
  `kohaku._kohaku_rs.cosine_topk`. 6 Rust unit tests.
- **maturin build** — root `pyproject.toml` makes `pip install .` build the
  `kohaku._kohaku_rs` extension and bundle the Python package. `python/pyproject.toml`
  (hatchling, pure-Python) remains the baseline install. Fixed the PyO3 bindings
  that never actually compiled (`pybindings` was missing from `lib.rs`; `bundle`
  signature; modern `Bound` module API).
- **`kohaku._accel`** — `cosine_topk` / `cosine_all` dispatch to Rust when the
  extension is present, else a NumPy matmul. The canonical `kohaku.query` and
  `query_with_decay` now compute similarities in one batched pass through this
  shim instead of a per-entry Python loop.
- **`_BACKEND`** now reports `"rust-accel"` when the extension is loaded,
  `"python"` otherwise. The Rust extension *accelerates*; it no longer replaces
  the canonical pure-Python classes (which broke when the two APIs diverged).
- **CI** — new `rust-accel` job builds the wheel with maturin and runs the
  library suite against the Rust backend (`--import-mode=importlib`), proving
  parity with the pure-Python path.
- **Tests** — `python/tests/test_accel.py` (NumPy path always; Rust-vs-NumPy
  parity + backend flag when built). Full suite: **572 passed** (pure),
  **519 passed** (Rust backend).

### Changed
- `__init__.py` / `python/pyproject.toml` / root `pyproject.toml` — version `0.18.0`.

## [0.17.0] — 2026-06-17

### Added — Track C3: benchmarks-as-a-gate

- **`benchmarks/run_benchmarks.py`** — reproducible scaling bench: retrieval
  latency (exact vs ANN), ANN top-1 agreement, and on-disk size (`.hkb` vs
  JSON). `--quick` for CI logs, `--json` to persist results. Runs straight from
  a checkout.
- **`python/tests/test_benchmarks.py`** — six performance *invariants* that gate
  CI (stable, not wall-clock): ANN recall@10 ≥ 0.8, candidate-set pruning,
  `.hkb` ≥ 5× smaller than JSON, exact binary round-trip recall, facade ANN
  top-1 parity, and a bench-script smoke test.
- **CI** — the `python` job now runs the gate tests and prints the quick bench.

### Changed

- **ANN default params retuned for recall** (`kohaku.ann`) — `num_tables`
  8 → **16**, `hash_bits` 16 → **12**. Because every LSH candidate is re-ranked
  with exact cosine, extra candidates are cheap but a missed bucket is a lost
  result; the new defaults lift recall@10 from ~0.73 to ~0.9 at 5% query noise.
  (Surfaced by the new recall gate.)
- `__init__.py` / `pyproject.toml` — version `0.17.0`.

## [0.16.0] — 2026-06-17

### Added — Track B3: unified system snapshot

One directory, the whole system — closing the persistence fragmentation gap
(episodic `.hkb` here, three loose SQLite files there).

- **`kohaku.system`** (`python/kohaku/system.py`) — `save_system(store, dir, *,
  provenance=, versions=, relationships=)` writes the episodic store
  (`memory.hkb`), the per-memory metadata table (`metadata.json`), and any
  attached SQLite side stores (provenance / versions / relationships) into one
  directory with a `manifest.json`. SQLite stores are copied via the sqlite
  backup API, so even `:memory:` stores persist. Side stores default to those
  already attached to `store`.
- **`load_system(dir) -> SystemBundle`** — rebuilds a wired-up
  `EnrichedMemoryStore` (recall is exact — HVs come from the packed `.hkb`,
  metadata and reinforcement counts are restored as-is) plus the side stores,
  re-attached for future writes.
- **`EnrichedMemoryStore.from_state(memory, metadata, *, capacity, dims, ...)`**
  — reconstruct a store from a loaded `EpisodicMemory` + metadata table without
  re-storing (which would mint new ids and reset counts).

- **Tests** — 7 new (`python/tests/test_system.py`): metadata + recall exact
  round-trip, manifest contents, full provenance/versions/relationships
  round-trip, attached-default behaviour, missing-manifest error. Full suite:
  **562 passed**.

- `__init__.py` / `pyproject.toml` — exports `save_system`, `load_system`,
  `SystemBundle`; version `0.16.0`.

## [0.15.0] — 2026-06-17

### Added — Track B2: approximate nearest-neighbour retrieval

Lifts the O(N·D) brute-force retrieval ceiling without adding a heavy
dependency.

- **`kohaku.ann.LSHIndex`** (`python/kohaku/ann.py`) — random-hyperplane
  (SimHash) locality-sensitive hashing over bipolar hypervectors. Pure NumPy,
  no FAISS/hnswlib. `add` / `remove` / `clear` / `candidates` / `query`
  (candidate gather + **exact** cosine re-rank) / `from_memory`. Configurable
  `num_tables` (recall) and `hash_bits` (precision).

- **`Memory(ann=True)`** — the facade maintains an `LSHIndex` and narrows
  similarity queries to LSH candidates before exact ranking. Results are
  unchanged except for the rare LSH miss; non-similarity sorts and empty
  candidate sets fall back to a full exact scan. Index stays consistent across
  FIFO eviction (rebuild), `expire`, and `clear`. `Memory.load(..., ann=True)`
  rebuilds it.

- **`EnrichedMemoryStore.query(..., candidate_ids=...)`** — optional precomputed
  candidate subset; entries outside it are skipped while every other
  filter/ranking still applies. `None` preserves the exact full scan.

- **Tests** — 12 new (`python/tests/test_ann.py`): index ops, parameter
  validation, self-match, near-duplicate recall, `from_memory`, and facade
  parity / eviction / expire / clear. Full suite: **555 passed**.

- `__init__.py` / `pyproject.toml` — exports `LSHIndex`; version `0.15.0`.

## [0.14.0] — 2026-06-17

### Added — Track B1: semantic encoder

The biggest quality lever from `ROADMAP.md` Track B — meaning-based recall
instead of token overlap.

- **`kohaku.semantic`** (`python/kohaku/semantic.py`) — project dense
  embeddings into HDC space via SimHash (sign of a fixed Gaussian random
  projection), which approximately preserves cosine similarity.
  - `project_to_hypervector(embedding, dims, *, seed)` — the standalone
    projection, with a per-`(embedding_dim, dims, seed)` matrix cache.
  - `EmbeddingEncoder(*, embed_fn=None, model_name="all-MiniLM-L6-v2", dims, seed)`
    — callable `str -> HyperVector`. Accepts any `embed_fn` (sentence-
    transformers, OpenAI, custom) so there is **no hard dependency**; the
    sentence-transformers path is lazily imported and raises a clear
    `ImportError` if the optional package is missing.

- **`Memory(encoder=...)`** — the facade now accepts any `str -> HyperVector`
  encoder, defaulting to the lexical `encode_text`. `save()` records the
  encoder kind; `Memory.load(path, encoder=...)` re-attaches it and warns on a
  custom/none mismatch (since HVs are re-derived from labels on load).

- **`[semantic]` extra** — `pip install kohaku[semantic]` pulls
  `sentence-transformers`.

- **Tests** — 11 new (`python/tests/test_semantic.py`), all using an injected
  `embed_fn` so the suite needs no heavy dependency. Full suite: **543 passed**.

- `python/kohaku/__init__.py` / `python/pyproject.toml` — version bumped to
  `0.14.0`; exports `EmbeddingEncoder`, `project_to_hypervector`.

## [0.13.0] — 2026-06-17

### Added — Track A: the `Memory` facade

The first slice of the `ROADMAP.md` Track A (credibility & correctness).

- **`Memory` facade** (`python/kohaku/memory_facade.py`) — a string-in /
  string-out front door over `EnrichedMemoryStore`. `Memory().store("…text…")`
  encodes via `encode_text` and persists; `query("…text…")` returns ranked
  `MemoryHit` rows (`.text`, `.score`, `.similarity`, `.salience`, `.source`,
  `.tags`). Supports `source` / tag filters, salience/recency sorting,
  reinforcement, expiry, and `save()` / `load()` (labels + metadata only —
  hypervectors are re-derived deterministically, so the round-trip is exact).
  Exported as `kohaku.Memory` / `kohaku.MemoryHit`. This makes the README's
  headline example actually run for the first time.

- **`enriched_meta.py` split** — `MemoryMetadata`, `EnrichedRetrievalResult`,
  the source-trust table, and the datetime/tag helpers moved out of
  `enriched.py` (was 521 lines, over the 500-line quality cap) into
  `enriched_meta.py`. `enriched.py` (now 399 lines) re-exports them, so every
  existing `from kohaku.enriched import …` path is unchanged.

- **README** — the Quick Start now matches the real, working API and documents
  `MemoryHit` plus when to reach past the facade to `EnrichedMemoryStore`.

### Changed — CI actually guards Python now

- `.github/workflows/ci.yml` — split into `rust` and `python` jobs. The Rust
  job additionally runs `cargo build --features python` so the optional PyO3
  bindings can't silently rot. The new Python job installs the package and runs
  the real `python/tests` + `api` suites (the konjo-gate previously pointed at a
  non-existent `tests/` path, so the Python suite never ran in CI).

- **Tests** — 16 new (`python/tests/test_memory_facade.py`). Full suite green.

- `python/kohaku/__init__.py` / `python/pyproject.toml` — version bumped to
  `0.13.0`.

## [0.12.0] — 2026-05-24

### Added — Phase 15: Graphiti/Mem0 Dialects + Forgetting-Rate Overrides

Two orthogonal capabilities: Graphiti/Mem0 export compatibility and
per-memory decay fine-tuning, plus six pre-sprint lint fixes.

- **Graphiti export dialect** — `MemoryGraph.to_graphiti()` /
  `to_graphiti_json()` emit a Graphiti-compatible graph dict.
  Memory nodes map to Graphiti episodes (`uuid`, `name`, `content`,
  `source`, `valid_at`, `invalid_at`, `attributes`); memory edges map
  to relations with `name="similar_to"` and a `weight` equal to the
  cosine similarity. `entities` is empty — kohaku memories are episodic
  facts, not named entities. `MemoryGraphExporter.save_graphiti(graph, path)`
  writes atomically via `.tmp` + `os.replace`.

- **Mem0 export dialect** — `MemoryGraph.to_mem0()` / `to_mem0_json()`
  emit a Mem0-compatible memory list. Each node becomes a `memory` record
  with `id`, `memory` (label text), `hash` (16-char SHA-256 prefix for
  deduplication), `metadata` dict, `score` (decay_weight when available,
  else 1.0), `created_at`, and `updated_at`.
  `MemoryGraphExporter.save_mem0(graph, path)` writes atomically.

- **REST endpoints** — `GET /export/graph/graphiti?threshold=0.3` and
  `GET /export/graph/mem0?threshold=0.3` on the unified FastAPI app.
  The `threshold` query parameter is forwarded to `GraphExportConfig`.

- **Per-memory forgetting-rate override** (`python/kohaku/enriched.py`) —
  `MemoryMetadata.forgetting_rate: Optional[float]` (validated > 0 at
  construction). `salience()` computes `effective_half_life = half_life_days
  / forgetting_rate` when set, leaving the default Ebbinghaus path intact
  when the field is `None`. This lets callers slow decay for high-priority
  memories (`rate < 1`) or accelerate it for ephemeral facts (`rate > 1`).
  `EnrichedMemoryStore.store(..., forgetting_rate=...)` accepts the
  parameter at write time. `POST /memories/store` exposes it as a Pydantic
  field validated `gt=0`.

- **Lint cleanup** — 6 ruff violations fixed: unused imports
  (`field`, `HyperVector`, `Sequence`, `ConflictPair`, `pytest`) and
  ambiguous variable name `l` in `test_portability.py`. `PLAN.md` version
  header updated from stale `v0.10.0` to `v0.11.0`.

- **Tests** — 35 new: 16 `python/tests/test_graphiti_mem0.py` (Graphiti
  and Mem0 struct contracts, hash determinism, score fallback, file I/O),
  10 `python/tests/test_forgetting_rate.py` (metadata validation, salience
  ordering, store integration, ranking with aged memories), 9 new API tests
  in `api/test_api.py`. **445 tests total**.

- `python/kohaku/__init__.py` / `python/pyproject.toml` — version bumped
  to `0.12.0`.

## [0.10.0] — 2026-05-12

### Added — Phase 11: Critical P1 Features

The first wave of the Researched Feature Roadmap. Four production-grade
capabilities, all integrated with the unified FastAPI app and covered by 54
new tests.

- **Temporal validity intervals** — `python/kohaku/enriched.py` introduces
  `MemoryMetadata(valid_from, valid_until, ...)`. `EnrichedMemoryStore.query()`
  filters expired items by default (opt-in `include_expired=True`).
  `expire_old()` drops everything past its `valid_until`. Naive datetimes are
  promoted to UTC on write so cross-timezone comparisons never raise.
  Validation: `valid_until >= valid_from`, `importance ∈ [0, 1]`,
  `reinforcement_count >= 0`.

- **Salience scoring** — composite score per memory:
  `salience = importance · 0.5^(age_days / half_life) · (1 + count · k) · trust(source)`.
  `EnrichedMemoryStore.query(sort='salience')` re-ranks results by this
  composite instead of raw cosine. Every retrieval reinforces every hit
  by default (the engine of the feedback loop), with `reinforce_hits=False`
  for read-only probes.

- **Memory provenance + trust weights** — each memory carries a `source`
  string (`"user_input"`, `"tool_result"`, `"web_search"`, `"agent_inference"`).
  `SOURCE_TRUST_WEIGHTS = {user_input: 1.0, tool_result: 0.9, web_search: 0.8,
  agent_inference: 0.5}` modulates salience. Agent-generated memories don't
  fail closed — they just rank lower. Tunable per `EnrichedMemoryStore` instance.

- **Sleep-phase consolidation daemon** — `python/kohaku/sleep.py` adds
  `SleepConsolidator(memory, consolidation_interval_minutes=60.0,
  similarity_threshold=0.85, on_report)`. Time-scheduled background thread
  (distinct from the existing pressure-driven `StreamingConsolidator`). Each
  run emits a structured `SleepReport(started_at, run_seconds, episodes_before,
  episodes_after, episodes_consolidated, prototypes_created, memory_freed,
  similarity_threshold)`. Manual `run_once()` plus `start()` / `stop()` /
  context-manager lifecycle. Callback exceptions are logged and swallowed
  so a bad observer can't kill the daemon.

- **Unified FastAPI endpoints** — `api/main.py` gains a `RestState.enriched`
  store and `RestState.sleep` daemon:
  - `POST /memories/store` — encode + persist with metadata.
  - `POST /memories/query` — top-k retrieval with `sort`, `source_filter`,
    `include_expired`, `min_similarity`, `reinforce_hits`.
  - `GET /memories?sort=salience&source=X&limit=N` — inventory with metadata.
  - `POST /memories/expire` — drop expired entries.
  - `GET /memories/trust-weights` — inspect the live trust table.
  - `POST /consolidate` — one-shot sleep-phase consolidation; returns the
    `SleepReport`.

- **Tests**: `python/tests/test_enriched.py` (24 cases — metadata validation,
  validity windows, salience math, trust modulation, source filter,
  reinforcement, expire-old, capacity eviction, sort modes),
  `python/tests/test_sleep.py` (15 cases — interval/threshold validation,
  empty/singleton/orthogonal/cluster runs, run count + report history,
  callback + exception swallow, lifecycle, background interval firing,
  `to_dict` JSON-serialisability), plus 15 new integration tests in
  `api/test_api.py` for the new endpoints. **282 tests total** (54 new + 228 prior).

- `python/kohaku/__init__.py` — exports `EnrichedMemoryStore`,
  `EnrichedRetrievalResult`, `MemoryMetadata`, `SOURCE_TRUST_WEIGHTS`,
  `SleepConsolidator`, `SleepReport`. Version bumped to `0.10.0`.
- `python/pyproject.toml` — version bumped to `0.10.0`.
- `PLAN.md` — adds **"Researched Feature Roadmap"** section (P1/P2/P3),
  ticks the Phase 11 P1 deliverables.

### Notes
- The enriched store wraps `EpisodicMemory` rather than modifying it — the
  `MemoryEntry` shape and `.hkb` binary format are unchanged, preserving
  Rust/Python LCG parity.
- "Encode validity into hypervector context bits" is intentionally deferred:
  the metadata-layer filter is correct today; the HV-bit encoding is a
  follow-up for a future wave once the consolidation pipeline needs it.

## [0.8.0] — 2026-05-10

### Added — Phase 9: kyro bridge + cosmos UI

- `python/kohaku/kyro_bridge.py` — `HDCRetriever(capacity, dims)` exposes
  the kyro-compatible RAG surface over kohaku's HDC engine. `ingest(docs)`
  accepts strings or `{"text", "id"?}` dicts, encodes each via
  `encode_text`, stores into a private `EpisodicMemory`, and keeps a
  parallel `entry_id → (doc_id, text)` map (HVs are not invertible).
  `retrieve(query, top_k, half_life?, floor?)` returns `RetrievedChunk`
  rows with raw cosine similarity, optional Ebbinghaus
  `decayed_similarity`, and `age` (ticks since ingest). 15 unit tests
  in `python/tests/test_kyro_bridge.py`.
- `api/main.py` — two bridge endpoints with their own dedicated
  `HDCRetriever` instance (kept isolated from `/store` + `/query` so the
  RAG corpus can't pollute the general-purpose memory):
  - `POST /bridge/ingest` — `{documents: [str | {text, id?}]}` →
    `{entry_ids, total_chunks}`.
  - `POST /bridge/retrieve` — `{query, top_k, half_life?, floor?}` →
    `{results: [{entry_id, doc_id, text, similarity, decayed_similarity, age}], decay_applied, total_chunks}`.
  7 new TestClient integration tests in `api/test_api.py`.
- `demo/memory_map.html` — full-screen cosmos visualization. Pure-black
  background. Each stored memory becomes a star whose **brightness**
  follows the Ebbinghaus decay curve, **size** grows with access
  frequency (recall-induced reconsolidation), and **colour** marks one
  of three signature clusters (amber · rose · teal). Connections form
  between near-neighbours (cosine > 0.32) and a traveling light dot
  animates along each link. Queries materialize a probe star with an
  expanding shockwave ring; lines arc to the top-k matches and the hit
  stars flare. A **time dial** scrubs the "now" tick forward (memories
  dim) or backward (memories brighten — the act of remembering).
  Toggles: **constellation mode** draws faint cluster connectors;
  **trails** toggle link visibility; **pulse** intensifies twinkle.
  Pointer-drag a star to nudge it. New memory births spawn 14
  particles that converge from the screen edges to the star's
  position. Browser HDC engine (DIMS=1024) ports the kohaku LCG path
  exactly: bipolar ±1, sign(LCG state >> 63), seed XOR with
  0xDEAD_BEEF_CAFE_BABE.
- `demo/index.html` — full rebuild as the cosmos landing page. Black
  sky with drifting dust, a single floating glass search input, and a
  curated 16-seed cosmos ready for query. Submitting a query spawns a
  probe star above the input, fires lines into the sky to the top-5
  matches, and blooms a row of cluster-coloured chips with similarity
  scores below. Idle placeholder rotates through evocative prompts.

### Notes
- `kyro_bridge` ships in this repo so the kyro RAG pipeline pulls it via
  `pip install kohaku` — no reverse dependency.
- Both demo pages are single-file, no build step, and run standalone
  (the in-browser HDC engine matches kohaku's pure-Python path bit-for-bit).
- Total test count: **182 passed** (15 bridge + 7 API + 159 prior).

## [0.7.0] — 2026-05-09

### Added — Phase 7: Visualization + REST API
- `api/main.py` — FastAPI visualization service backed by the live `kohaku` library. `VizState` loads `demo/sample_memory.json` into an `EpisodicMemory`, runs cosine k-means on the bipolar hypervectors (centroid re-binarised by majority vote each iteration; deterministic seeding), and exposes:
  - `GET /viz/graph?threshold&k&half_life` — `{nodes, edges, dims, threshold, num_clusters, half_life, current_clock}`. Each node carries `id, entry_id, label, cluster, cluster_label, color, last_accessed, age, decay_weight`. Edges include only pairs with `cosine ≥ threshold`. Decay weight is computed by the real `kohaku.decay.decay_weight` from the entry's age in memory ticks — proven by `test_decay_weights_match_ages_in_graph`.
  - `GET /viz/decay?half_life&horizon&steps` — per-concept Ebbinghaus curves: `[{age, weight}]` over `[0, horizon]` plus each concept's `current_age` / `current_weight` marker.
  - `POST /viz/probe` — encodes a query phrase via `kohaku.encode_text` and returns top-k cosine matches across the live memory.
  - `GET /viz/memory_map.html` — serves the d3-force viewer.
- `demo/memory_map.html` — d3-force-directed viewer (d3 v7 via CDN). Node radius scaled by Ebbinghaus decay weight, colour by k-means cluster, edges drawn for cosine ≥ slider threshold. Probe input animates dashed edges from the strongest-match node to the rest of the activated set. Sliders re-issue the API call on change for half-life / threshold / k. Decay panel renders all 12 forgetting curves with the current-age marker.
- `demo/sample_memory.json` — 12 hand-authored concepts across three ground-truth clusters (animals, programming, cities). Phrases share heavy anchor vocabulary inside each cluster so within-cluster cosine reliably exceeds 0.7 and between-cluster cosine stays under 0.4 — verified by `test_kmeans_recovers_ground_truth_clusters`.
- `api/test_viz.py` — 6 tests: graph contract & node-field invariants, edge-threshold subset relation across thresholds, k-means recovery of all three ground-truth clusters, decay-curve shape (anchored at 1.0, monotonically non-increasing, half-life crossing at 0.5), graph decay weight ↔ `decay_weight(age, cfg)` agreement, probe ranks programming concepts at the top with empty-query rejection.
- `python/kohaku/__init__.py` — version bumped to `0.7.0`.
- `python/pyproject.toml` — version bumped to `0.7.0`.
- `PLAN.md` — Phase 7 added and ticked.
- Total test count: **147 passed** (6 new + 141 prior; pre-existing skip resolved).

### Notes
- The visualization layer is read-only — it observes a kohaku memory built from the seed file but does not mutate `EpisodicMemory` semantics. The same `VizState` can be wrapped around any external `EpisodicMemory` by passing `memory=...` to `create_app(state=VizState(memory=mem, concepts=...))`.
- K-means uses farthest-point-free deterministic seeding (first `k` entries) to keep the graph layout stable across reloads. Centroids are re-binarised to ±1 each iteration so they remain valid bipolar hypervectors.
- The viz layer is read-only — it observes a kohaku memory built from the seed file but does not mutate `EpisodicMemory` semantics.
- Sign-binarization invariant enforced at the API boundary: raw float vectors collapse to ±1 before entering any HDC op.

### Added — Phase 7: REST API (same release)

Adds a write-able REST surface alongside the viz endpoints, on the same FastAPI app:

- `POST /encode` / `POST /store` / `POST /query` / `POST /bundle` /
  `GET /stats` / `GET /health`. `RestState` holds an `EpisodicMemory` +
  `ItemMemory` guarded by `threading.Lock`. `/query` accepts `half_life`
  and `floor` to attach Ebbinghaus `decayed_similarity` from
  `query_with_decay`. Probing by `label` uses the learned semantic
  prototype.
- `api/requirements.txt` — fastapi, uvicorn[standard], pydantic v2,
  numpy, httpx (for `TestClient`).
- `api/Dockerfile` — python:3.11-slim, `PYTHONPATH=/app/python`,
  uvicorn on `$PORT` (default 8000).
- `render.yaml` — Render.com web service spec, Docker env, `/health`
  healthcheck, `KOHAKU_CAPACITY` env var.
- `api/test_api.py` — 18 integration tests via `TestClient` over the
  real FastAPI app. No mocks — `encode_text`, `HyperVector.bundle_all`,
  `EpisodicMemory.store`, `query`, `query_with_decay` all called directly.

## [0.6.0] — 2026-05-07

### Added
- `python/kohaku/streaming.py` — `StreamingConsolidator`: background daemon thread that polls memory utilization every `poll_interval_s` seconds and auto-runs semantic consolidation when `len(memory) / capacity >= trigger_ratio`. Thread-safe via `threading.Lock`. API: `store(key, value, label)`, `retrieve(query_key, top_k)`, `start()`, `stop(timeout)`, `is_running`, `consolidation_count`. Context-manager protocol (`__enter__` / `__exit__`) for clean lifecycle management. Consolidated memory is rebuilt from cluster centroids in-place; each run increments `consolidation_count`.
- `python/kohaku/compaction.py` — `find_duplicates(memory, similarity_threshold)`: O(n²) cosine scan over `MemoryEntry.key` to identify near-duplicate groups (returns `List[Set[int]]` of entry IDs). `deduplicate(memory, similarity_threshold)`: removes all but the oldest (lowest-ID) entry in each group in-place; returns count removed. `compact(memory, target_utilization)`: deduplicates first, then FIFO-evicts entries until `len(memory) <= capacity * target_utilization`; returns total entries removed. Raises `ValueError` for `target_utilization` outside `(0, 1]`.
- `python/kohaku/tenant.py` — `TenantMemoryStore(dimension, capacity)`: registry of per-tenant `EpisodicMemory` instances. Unknown tenants auto-provisioned on first `store()` or `retrieve()` call. Empty tenant IDs raise `ValueError`. API: `store(tenant_id, key, value, label)`, `retrieve(tenant_id, query_key, top_k)`, `size(tenant_id)`, `drop_tenant(tenant_id)`, `tenant_ids`, `tenants_count()`. Tenants are fully isolated — cross-tenant retrieval is impossible by construction.
- `python/kohaku/__init__.py` — exports `StreamingConsolidator`, `find_duplicates`, `deduplicate`, `compact`, `TenantMemoryStore`. Version bumped to `0.6.0`.
- `python/pyproject.toml` — version bumped to `0.6.0`.
- `PLAN.md` — Phase 6 checkboxes ticked, current version updated to `v0.6.0`.
- `python/tests/test_streaming_consolidation.py` — 10 tests: init defaults, bad trigger ratio, bad poll interval, start/stop lifecycle, double-start no-op, context manager, thread-safe concurrent stores, consolidation fires above trigger, consolidation count increments, no consolidation below trigger.
- `python/tests/test_compaction.py` — 10 tests: identical keys have similarity 1.0, orthogonal keys not duplicates, find_duplicates returns correct ID groups, find_duplicates empty memory, deduplicate removes near-duplicates, deduplicate keeps oldest entry, deduplicate no-op when clean, compact bad utilization raises, compact reduces to target, compact deduplicates before eviction.
- `python/tests/test_tenant.py` — 11 tests: init defaults, bad dimension raises, bad capacity raises, empty tenant ID raises on store/retrieve, unknown tenant auto-provisioned, unknown tenant size returns zero without provisioning, isolation between tenants, size per tenant, drop removes all data, drop nonexistent returns false, ten independent tenants.
- Total test count: **139 passed** (31 new + 108 prior).

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
