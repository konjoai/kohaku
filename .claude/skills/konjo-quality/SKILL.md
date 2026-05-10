---
name: konjo-quality
description: Konjo Code Quality Framework — all gate definitions, thresholds, tools, and enforcement points. Auto-load when writing tests, reviewing code quality, refactoring, or when quality gate failures are mentioned. Applies the Three-Wall framework to prevent AI slop.
user-invocable: true
---

# Konjo Quality Framework — Agent Reference

## The Three Walls

| Wall | When | What | Blocks |
|------|------|------|--------|
| **Wall 1** | Pre-commit hook | Format, lint, unwrap scan, DRY (staged only), TODO scan | The commit |
| **Wall 2** | CI / GitHub Actions | Coverage, mutation, complexity, size, docs, audit, review | The merge |
| **Wall 3** | CI (PRs only) | Claude Opus adversarial review against 10 mandatory questions | The merge |

## Hard Quality Thresholds

| Metric | Hard Block | Target | Tool |
|--------|-----------|--------|------|
| Line coverage | ≥ 80% | ≥ 95% | cargo-llvm-cov / pytest-cov |
| Mutation survival | ≤ 10% | 0% | cargo-mutants / mutmut |
| Cognitive complexity | ≤ 15 | ≤ 10 | clippy / radon |
| Lint violations | 0 | 0 | clippy / ruff |
| Dead code | 0 | 0 | rustc / vulture |
| File length | ≤ 500 lines | ≤ 300 lines | wc |
| DRY violations | 0 | 0 | dry_check.py |
| unwrap() in non-test Rust | 0 | 0 | clippy::unwrap_used |
| Known CVEs | 0 | 0 | cargo-audit / safety |

## Zero-Tolerance Rules

- Dead code, commented-out code, TODO/FIXME in production
- Undocumented public APIs
- Silent error swallowing
- Duplicate code blocks (>10 lines, >85% similar)
- Rust: `unwrap()`, `expect()`, `panic!()` outside test code
- Python: bare `except:` without log and re-raise

## Running the Gates Locally

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings -D clippy::unwrap_used
cargo nextest run --workspace --lib
cargo llvm-cov nextest --workspace --fail-under-lines 80
python3 .konjo/scripts/dry_check.py --staged-only
git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py
```
