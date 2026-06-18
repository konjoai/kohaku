# Kohaku — Forward Roadmap

> State assessment and prioritized plan as of 2026-06-17 (Python `v0.12.0`, Rust `v0.1.0`).
> `PLAN.md` is the *backward-looking* phase log; this is the *forward-looking* plan.

## 1. Where Kohaku is today

Kohaku is a **mature, Python-first HDC episodic-memory engine** with a large,
well-tested feature surface. The suite is green: **441 passing + 1 skipped**
(`python/tests/`), **55 passing** (`api/`), **33 passing** (Rust). `cargo clippy
-- -D warnings` is clean.

### Memory layers — all three are present

| Layer | Where | Notes |
|-------|-------|-------|
| **Episodic** | `_pure.py` (`EpisodicMemory`), `enriched.py` (`EnrichedMemoryStore`), `episode.py` (role-bound who/what/when/where) | Capacity-managed FIFO, temporal decay, validity intervals, salience, provenance. |
| **Associative** | `retrieval.rs` / `_query.py`, `hopfield.py`, `chaining.py` | Top-k & threshold cosine retrieval, modern continuous Hopfield clean-up, multi-hop chaining. |
| **Semantic** | `consolidation.py`, `learning.py` (`ItemMemory`), `memory_system.py`, `sleep.py` | Cluster→centroid consolidation, online prototypes, episodic↔semantic split, sleep-phase daemon. |

So: **yes** — episodic, associative, and semantic layers are all implemented,
and the episodic↔semantic *consolidation* path (the interesting part) exists.

### Other features shipped
Temporal decay (Ebbinghaus) · per-memory forgetting-rate override · salience
scoring · source/trust weighting · provenance DAG (SQLite) · memory versioning
(SQLite) · typed relationships (SQLite) · auto-importance scoring · conflict
detection + resolution · tags · time-range filtering & timelines · memory-health
analyzer · bulk ops · streaming + sleep consolidation daemons · multi-tenant
isolation · compaction/dedup · persistence (`.hkb` packed-bit + JSON) · graph
export (native, GEXF, **Graphiti**, **Mem0**) · portable import/export
(JSON/MD/CSV) · OpenAI-compatible middleware · HF Trainer hooks · kyro RAG bridge
· unified FastAPI app (REST + viz + bridge) · standalone server module + `kohaku`
CLI · cosmos visualizations.

## 2. The gaps that matter (verified, not speculative)

1. **The Rust core is effectively abandoned at v0.1.0.** The runtime backend is
   pure-Python (`_BACKEND == "python"`); `maturin` is not in the build or CI, so
   `_kohaku_rs.so` is never produced. The README's headline "🦀 Rust core —
   high-performance HDC engine" is currently aspirational. Rust covers only
   HV/memory/retrieval; **none** of the ~30 Python feature modules have a Rust
   path. PyO3 bindings (`pybindings.rs`) only wrap `PyHyperVector` /
   `PyEpisodicMemory` and are never compiled in CI.

2. **The README's first code example does not run.** It advertises
   `from kohaku import Memory` with `mem.store("User prefers Italian wine")` /
   `mem.query("...")`. But `Memory` is **not exported** (the class is
   `EpisodicMemory`), and `store(key, value, label)` requires *hypervectors*,
   not raw strings. The advertised top-level ergonomic API does not exist.

3. **The CI quality gate is currently red.** `enriched.py` is **521 lines** and
   the konjo-gate enforces a hard **≤ 500-line** cap on every `.rs`/`.py`. Any
   run of `konjo-gate.yml` fails on this file today.

4. **No high-level facade.** There are 30+ modules but no single object that
   wires enriched store + decay + consolidation + provenance + relationships
   behind a string-in/string-out API — which is exactly what the README implies
   and what a new user reaches for first.

5. **Fragmented persistence.** Episodic memory saves to `.hkb`/JSON, but
   metadata, provenance, versions, and relationships live in three separate
   SQLite files. There is no "save/load the whole system" round-trip.

6. **O(n²) everywhere, no index.** Retrieval, dedup, conflict scan, importance
   uniqueness, and graph export all do pairwise cosine with `max_*` safety caps.
   There is no ANN index — a hard scaling ceiling past ~10⁴ memories.

7. **Encoding is lexical, not semantic.** `encode_text` bundles per-*token* LCG
   hypervectors, so cosine similarity is essentially bag-of-words overlap. "User
   likes wine" and "the customer enjoys merlot" are near-orthogonal. The README
   positions Kohaku against RAG/embeddings, but offers no semantic encoder.

8. **Demo sprawl.** Seven+ standalone HTML demos
   (`memory_map`, `_cosmic`, `_cosmos`, `_dashboard`, `kohaku-live`, `dashboard`,
   `index`) with the last several commits all being UI re-iterations. The
   product surface is fragmenting.

## 3. Plan — three tracks, prioritized

### 🔴 Track A — Credibility & correctness (do first, small, high-leverage) — ✅ DONE (v0.13.0)

- [x] **A1. Fix the README** — Quick Start now matches the real API and the
  headline example runs (via the A2 facade).
- [x] **A2. Ship a `Memory` facade** (`kohaku/memory_facade.py`): `Memory()` with
  `store(text, **meta) -> id`, `query(text, top_k=...) -> [MemoryHit]`,
  `save(path)` / `load(path)`. Wraps `EnrichedMemoryStore` + `encode_text`.
  Exported as `kohaku.Memory`.
- [x] **A3. Split `enriched.py`** (521 → 399): `MemoryMetadata` + salience math
  extracted into `enriched_meta.py`; `enriched.py` re-exports for compatibility.
- [x] **A4. Wire the `python` feature into CI** — `ci.yml` now builds
  `cargo build --features python` and runs the real `python/tests` + `api`
  suites in a dedicated Python job. Full maturin wheel publishing remains part
  of C1 (the Rust-story decision).

### 🟠 Track B — Scale & semantics (the next real capability jump)

- [x] **B1. Semantic encoder (opt-in).** ✅ (v0.14.0) `kohaku.semantic` —
  `EmbeddingEncoder` projects a dense embedding into HDC space via SimHash
  (sign of a fixed random projection), gated behind the `[semantic]` extra so
  the zero-dependency lexical path stays the default. `Memory(encoder=...)`
  wires it in. Accepts any `embed_fn`, so it composes with OpenAI embeddings
  too. This is the single biggest quality lever for real LLM-memory use.
- [x] **B2. ANN index for retrieval.** ✅ (v0.15.0) `kohaku.ann.LSHIndex` —
  native bipolar-LSH (random-hyperplane SimHash), pure NumPy, no FAISS/hnswlib.
  `Memory(ann=True)` narrows similarity queries to LSH candidates then re-ranks
  with exact cosine (brute force stays the correctness baseline via
  `candidate_ids=None`). Lifts the ~10⁴ ceiling.
- [x] **B3. Unified persistence.** ✅ (v0.16.0) `kohaku.system.save_system` /
  `load_system` snapshot episodic `.hkb` + metadata + provenance + versions +
  relationships into one directory with a `manifest.json` (SQLite stores copied
  via the backup API, so `:memory:` persists too). `SystemBundle` carries the
  rebuilt store + side stores; recall is exact after the round-trip.

### 🟡 Track C — Strategic positioning (decide, then commit)

- **C1. Resolve the Rust story.** Either (a) **re-commit**: port the hot O(n²)
  loops (retrieval, consolidation, conflict/importance scans) to Rust behind the
  existing feature flag and publish wheels via maturin — making the "Rust core"
  claim true; or (b) **reposition**: relabel as a Python-first engine with an
  optional Rust kernel, and stop advertising performance the runtime doesn't
  deliver. Pick one; the current limbo is the worst option.
- [x] **C2. Consolidate the demos.** ✅ (v0.21.0) Maintained demo is
  `kohaku-live.html`; four redundant memory-map iterations moved to
  `demo/archive/`. `memory_map.html` (viz API) and `index.html`
  (`demo/server.py`) are retained because the servers serve them — full
  single-page collapse would require rewiring those endpoints (deferred).
  `demo/README.md` documents the canonical surface.
- [x] **C3. Benchmarks as a gate.** ✅ (v0.17.0) `benchmarks/run_benchmarks.py`
  (latency exact vs ANN, ANN agreement, `.hkb` vs JSON size) + six invariant
  gates in `test_benchmarks.py` run in CI. Surfaced + fixed an ANN default that
  favoured precision over recall (recall@10 ~0.73 → ~0.9).

- [~] **C1. Resolve the Rust story → decision: commit to Rust.** *In progress.*
  - [x] Slice 1 (v0.18.0): Rust `cosine_topk` (bit-packed popcount) behind the
    `python` feature flag, exposed via PyO3; maturin build (`pip install .`);
    `kohaku._accel` dispatch with NumPy baseline; canonical `query` /
    `query_with_decay` batched through it; CI `rust-accel` job proving parity.
  - **Finding (`benchmarks/bench_backends.py`):** the bit-packed kernel is
    correct but the current PyO3 interface marshals Python list-of-lists per
    call (`keys.tolist()` ≈ 10M ints), so Rust is **~3× slower than NumPy**
    (0.33× at N=1k–5k, dims=10k). NumPy's `asarray`+BLAS wins because the win
    is destroyed by marshaling, not the math.
  - [x] Slice 2 (v0.19.0): **zero-copy FFI** via the `numpy`/`rust-numpy` crate
    (`cosine_topk` takes `PyReadonlyArray1/2<i8>`) **+ resident packed index**
    (`PackedIndex`, exposed as `kohaku.RetrievalIndex`); `query` /
    `query_with_decay` route through a per-memory cached index (invalidated by a
    monotonic `_generation` counter).
  - **Result (`benchmarks/bench_backends.py`):** zero-copy one-shot is now
    ~parity with NumPy (0.7–1.0×, vs 0.25× in slice 1) — re-packing every call
    is no clear win over BLAS, so NumPy stays the one-shot default. The resident
    index, which packs keys once, is **~160–230× faster than NumPy** on the
    repeated-probe workload. That amortization is the real win.
  - [x] Slice 3 (v0.20.0): ported the O(n²) similarity scans onto
    `RetrievalIndex` — uniqueness (importance), duplicate detection
    (compaction + memory-health), conflict detection, consolidation's
    entry-vs-centroid scan, and the enriched/facade query path (cached index).
    New `index_over` helper. **~15–22× faster** than the naive Python loop on
    the all-pairs uniqueness scan (`benchmarks/bench_scans.py`). Also fixed a
    slice-2 latent bug: direct `_entries` deletes now bump `_generation` so the
    index cache can't go stale.
  - [x] Follow-up (v0.21.0): **unified ANN + RetrievalIndex.** The facade query
    now packs and scores only the ANN candidate rows when `candidate_ids` is
    supplied (`index_over`), so LSH narrowing actually saves work — **~72×**
    faster facade query at N=5000 / 2% candidates (`benchmarks/bench_ann_rerank.py`).
  - **Publishing wheels remains deferred** (out of scope — local/CI builds only).

## 4. Suggested first sprint

Track A in full (A1–A4) — it is all low-risk, makes the front door honest,
turns CI green, and unblocks the facade everything else benefits from — plus
**B1 (semantic encoder)** as the headline feature. Defer B2/B3 and the Rust
decision (C1) to a dedicated follow-up once A is merged.

**Update:** Tracks A and B are now complete (v0.13.0 → v0.16.0). The remaining
work is all Track C — the strategic positioning decisions (C1 Rust story, C2
demo consolidation, C3 benchmarks-as-a-gate).
