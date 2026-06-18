# kohaku

Episodic memory engine for LLMs — persistent, associative, and beyond context windows. Hyperdimensional Computing (HDC) in Rust with Python bindings, temporal decay, semantic consolidation, and OpenAI-compatible memory middleware.

**v0.12.0** (Python) / **v0.1.0** (Rust) — 445 tests passing.

## Stack
Rust 2021 · rand · serde · anyhow · clap · PyO3 (optional, `--features python`) · Python 3.9+ · NumPy · asyncio · hatchling

## Commands
```bash
cargo build                                      # build Rust core
cargo test                                       # run Rust unit tests
cargo clippy -- -D warnings                      # lint
cargo run -- demo                                # run HDC demo
cargo run -- bench                               # run Rust benchmarks
maturin develop --features python                # compile PyO3 .so (requires maturin)
python -m pytest tests/ -x                       # Python test suite
```

## Critical Constraints
- No `unwrap()`/`expect()` outside tests — use `anyhow::Result` and `?`
- No silent failures — log a warning when a fallback path swallows an error
- PyO3 bindings are **optional** (`--features python`) — pure-Python path is always the correctness baseline
- LCG seed must be identical between Rust and Python: XOR with `0xDEAD_BEEF_CAFE_BABE`, same multiplier/addend
- Hypervectors must be bipolar ±1 float32 — assert at all HDC operation boundaries
- `AsyncEpisodicMemory` uses `asyncio.to_thread` — never call blocking Rust from the event loop directly
- `.hkb` binary format: magic `KHKU`, 1 bit per ±1 component — preserve round-trip fidelity in all serialization tests
- `decay_weight` must clamp to `floor ∈ [0, 1]` — assert `half_life > 0` and `age ≥ 0` at every call site
- Version bumps touch `python/pyproject.toml` + `python/kohaku/__init__.py`

## Crate / Module Map
| Component | Role |
|-----------|------|
| `src/hypervector.rs` | Random HV generation, bundle, bind, permute ops |
| `src/memory.rs` | `EpisodicMemory` — capacity-managed FIFO store |
| `src/retrieval.rs` | Top-k and threshold associative retrieval |
| `src/pybindings.rs` | PyO3 bindings (`--features python`) |
| `python/kohaku/_pure.py` | Pure-Python HDC — identical semantics to Rust LCG path |
| `python/kohaku/__init__.py` | Auto-detects Rust extension, falls back to pure-Python |
| `python/kohaku/context.py` | `ContextMemoryManager` — sliding-window store sized to token budget |
| `python/kohaku/attention.py` | `attention_weighted_encode` — HV encoding weighted by attention scores |
| `python/kohaku/openai_compat.py` | `MemoryMiddleware` — augment/learn from OpenAI-compatible exchanges |
| `python/kohaku/persistence.py` | JSON + `.hkb` binary save/load |
| `python/kohaku/consolidation.py` | Greedy semantic clustering into centroid memories |
| `python/kohaku/decay.py` | `DecayConfig` + `query_with_decay` — exponential temporal decay |
| `python/kohaku/extraction.py` | Heuristic free-text → `(subject, attribute, value)` triples; feeds `AnalogicalMemory.learn` / `Memory.learn` |

## Planning Docs
- `PLAN.md` — current phase state and version history
- `CHANGELOG.md` — all notable changes

## Konjo Quality Framework

Three walls against AI slop — all enforced by CI.

**Wall 1 — Pre-commit** (`bash .konjo/scripts/install-hooks.sh`):
cargo check, clippy, ruff lint, ruff format, DRY check, TODO scan. Blocks the commit.

**Wall 2 — CI gate** (`.github/workflows/konjo-gate.yml`):
Coverage ≥ 80% · mutation survival ≤ 10% · complexity ≤ 15 · file ≤ 500L · zero DRY violations. Blocks the merge.

**Wall 3 — Adversarial review** (local only — disabled in CI):
`git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py`

See `KONJO_QUALITY_FRAMEWORK.md` for the full specification.

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
