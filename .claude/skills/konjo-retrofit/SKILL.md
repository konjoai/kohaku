---
name: konjo-retrofit
description: Retrofit the Konjo Quality Framework onto an existing repo that predates it. Use when asked to add konjo quality gates, improve code quality, audit an existing codebase, or run a quality sprint on any repo.
user-invocable: true
---

# Konjo Retrofit — Existing Repo Quality Migration

## Protocol

1. **Baseline Audit** — measure everything, fix nothing yet
2. **Triage** — P0 (security/correctness), P1 (coverage/unwrap), P2 (style/length)
3. **Install Framework** — set gates at current baseline minus 2%
4. **Coverage Ratchet** — +5% per sprint until 80% hard floor
5. **Complexity Ratchet** — one function at a time with characterization tests
6. **DRY Cleanup** — highest similarity violations first
7. **Activate Wall 3** — soft-fail first week, blocking second week

## Rust + Python Hybrid Checklist (kohaku, toki, drex, vectro)
- [ ] `cargo audit` clean
- [ ] `cargo deny check` configured and clean
- [ ] `cargo llvm-cov` gives ≥ 80% coverage
- [ ] `clippy -D unwrap_used` passes
- [ ] `ruff check` clean; `ruff format --check` clean
- [ ] `mypy --strict` clean
- [ ] `vulture` zero dead code
- [ ] `.konjo/deny.toml` committed
- [ ] `.github/workflows/konjo-gate.yml` active
- [ ] Python-only mode is always the correctness baseline
- [ ] PyO3 bindings tested from both sides
