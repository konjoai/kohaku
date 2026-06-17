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
- **B2. ANN index for retrieval.** Replace the O(n²) scan with an optional
  index (FAISS/hnswlib, or a native bipolar-LSH bucketed by sign-projection).
  Keep the brute-force path as the correctness baseline. Lifts the ~10⁴ ceiling.
- **B3. Unified persistence.** One `save_system(dir)` / `load_system(dir)` that
  snapshots episodic `.hkb` + metadata + provenance + versions + relationships
  together, with a manifest and a round-trip test.

### 🟡 Track C — Strategic positioning (decide, then commit)

- **C1. Resolve the Rust story.** Either (a) **re-commit**: port the hot O(n²)
  loops (retrieval, consolidation, conflict/importance scans) to Rust behind the
  existing feature flag and publish wheels via maturin — making the "Rust core"
  claim true; or (b) **reposition**: relabel as a Python-first engine with an
  optional Rust kernel, and stop advertising performance the runtime doesn't
  deliver. Pick one; the current limbo is the worst option.
- **C2. Consolidate the demos** into one maintained `kohaku-live.html` and move
  the rest to an `demo/archive/` so the product surface stops fragmenting.
- **C3. Benchmarks as a gate.** A reproducible bench (recall@k vs memory size,
  latency, `.hkb` size) checked into CI, so scaling regressions are visible.

## 4. Suggested first sprint

Track A in full (A1–A4) — it is all low-risk, makes the front door honest,
turns CI green, and unblocks the facade everything else benefits from — plus
**B1 (semantic encoder)** as the headline feature. Defer B2/B3 and the Rust
decision (C1) to a dedicated follow-up once A is merged.
