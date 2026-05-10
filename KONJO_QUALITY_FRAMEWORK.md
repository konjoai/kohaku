# Konjo Code Quality Framework
## Three Walls Against AI Slop — Language-Agnostic, Gate-Enforced

**Version:** May 2026 · **Scope:** All KonjoAI repositories
**Reference implementation:** lopi (Rust) · **Applicable languages:** Rust, Python, Mojo, TypeScript

---

## The Three Walls

```
Wall 1: Pre-Commit Hook     ← local, fast (< 60s), blocks the commit
Wall 2: CI Gate             ← GitHub Actions, blocks the PR merge
Wall 3: Konjo Review Agent  ← Claude Opus in a separate session, blocks the merge
```

Every commit must pass Wall 1. Every PR must pass Wall 2. Every merge to main must pass Wall 3.
No bypass flags. No `--no-verify`. No `skip-review` comments.

---

## Quality Gate Reference Table

| Gate | Hard Block | Target | Tool (Rust) | Tool (Python) |
|------|-----------|--------|-------------|---------------|
| Line coverage | ≥ 80% | ≥ 95% | cargo-llvm-cov | pytest-cov |
| Mutation survival | ≤ 10% | 0% | cargo-mutants | mutmut |
| Cognitive complexity | ≤ 15 | ≤ 10 | clippy | radon cc |
| Lint violations | 0 | 0 | clippy | ruff check |
| Dead code | 0 | 0 | rustc | vulture |
| File length | ≤ 500 lines | ≤ 300 lines | loc | loc |
| DRY violations | 0 | 0 | dry_check.py | dry_check.py |
| unwrap() in non-test Rust | 0 | 0 | clippy::unwrap_used | — |
| Known CVEs | 0 | 0 | cargo-audit | safety |

---

## Install

```bash
bash .konjo/scripts/install-hooks.sh
```

See `KONJO_QUALITY_FRAMEWORK.md` in lopi for the full specification.
