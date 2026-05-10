---
name: konjo-boot
description: Boot a Konjo session for kohaku — kohaku — episodic memory engine (HDC, Rust kernels, PyO3, temporal decay, semantic consolidation, OpenAI-compat middleware). Produces a Session Brief, runs Discovery, identifies the next sprint. Use at the start of any work session or when invoked with /konjo.
user-invocable: true
---

# Konjo Boot — kohaku

## Step 1 — Orient
Read CLAUDE.md, README.md, CHANGELOG.md, PLAN.md in order.

## Step 2 — Session Brief
Output a brief covering:
- Current version and test count
- Last shipped (from CHANGELOG.md)
- Active blockers
- Health: Green / Yellow / Red

## Step 3 — Discovery
- `cargo test` — are all Rust tests green?
- `python -m pytest tests/ -x` — are all Python tests green?
- `cargo clippy -- -D warnings` — any lint violations?
- `ruff check .` — any Python lint violations?

## Step 4 — Plan
Identify the next sprint from PLAN.md and propose the first concrete task.
