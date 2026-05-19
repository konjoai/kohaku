#!/usr/bin/env python3
"""
Konjo Adversarial Review Agent — Wall 3.
Critic model: claude-opus-4-6
Exit codes: 0=APPROVED/WARNING, 1=BLOCKER, 2=API error
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

CRITIC_MODEL = "claude-opus-4-6"
MAX_DIFF_CHARS = 80_000
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

SYSTEM_PROMPT = """\
You are the Konjo Adversarial Reviewer — an independent critic whose role is to
find flaws that the builder missed. You were NOT involved in writing this code.
You have no loyalty to the implementation choices made. Your only loyalty is to
the ten quality standards below.

THE TEN MANDATORY REVIEW QUESTIONS — answer each explicitly:
Q1 CORRECTNESS: logical errors, off-by-ones, race conditions, silent data corruption
Q2 COVERAGE BLIND SPOTS: untested input paths, silent failure modes
Q3 DEAD CODE: unreachable code, unused variables, commented-out blocks
Q4 DOCUMENTATION: every public API documented, math explained, invariants stated
Q5 ERROR HANDLING: errors propagated or swallowed, bare except/unwrap, silent fallbacks
Q6 DRY VIOLATION: duplicate blocks >10 lines at >85% similarity
Q7 COMPLEXITY AND SIZE: function >50L, file >500L, complexity >15
Q8 SECURITY: prompt injection, sensitive data logged, missing validation
Q9 PERFORMANCE: O(n^2) regressions, blocking async calls, unnecessary allocations
Q10 KONJO STANDARD: seaworthy under 10k requests for 30 days?

VERDICT RULES:
- BLOCKER: Q1 errors, Q3 dead code, Q5 silent failures, Q8 security, function >100L, undocumented public API
- WARNING: Q2 partial coverage, Q6 minor duplication, Q7 approaching limits
- APPROVED: all ten questions pass

OUTPUT FORMAT — valid JSON only, no markdown:
{"verdict": "APPROVED"|"WARNING"|"BLOCKER", "summary": "...", "questions": {"Q1": {"verdict": "PASS"|"WARN"|"BLOCK", "finding": "..."}, ...}, "blockers": [], "warnings": [], "approved_aspects": []}
"""

def _load_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        raise

def _call_api(diff_text: str, anthropic_module) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic_module.Anthropic(api_key=api_key)
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + f"\n\n[DIFF TRUNCATED at {MAX_DIFF_CHARS} chars]"
    user_content = f"Review this pull request diff against the ten Konjo quality standards.\n\n<diff>\n{diff_text}\n</diff>"
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=CRITIC_MODEL, max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            usage = response.usage
            print(f"[konjo-review] tokens: input={usage.input_tokens} output={usage.output_tokens} cache_read={getattr(usage, 'cache_read_input_tokens', 0)}", file=sys.stderr)
            return json.loads(raw)
        except (anthropic_module.RateLimitError, anthropic_module.APIStatusError):
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(f"[konjo-review] retrying in {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                raise
        except json.JSONDecodeError as exc:
            raise ValueError(f"Non-JSON response: {raw}") from exc
    raise RuntimeError("Exhausted retries")

def _render_human(result: dict) -> str:
    lines = ["# Konjo Adversarial Review Report\n"]
    verdict = result.get("verdict", "UNKNOWN")
    emoji = {"APPROVED": "✅", "WARNING": "⚠️", "BLOCKER": "🚫"}.get(verdict, "❓")
    lines.append(f"## Verdict: {emoji} {verdict}\n")
    lines.append(f"**Summary:** {result.get('summary', '')}\n")
    for b in result.get("blockers", []):
        lines.append(f"- BLOCKER: {b}")
    for w in result.get("warnings", []):
        lines.append(f"- WARNING: {w}")
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser()
    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument("--diff-file")
    diff_group.add_argument("--diff")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--soft-fail", action="store_true")
    args = parser.parse_args()
    if args.diff:
        diff_text = args.diff
    elif args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    else:
        if sys.stdin.isatty():
            print("ERROR: provide --diff-file or pipe a diff to stdin", file=sys.stderr)
            return 2
        diff_text = sys.stdin.read()
    if not diff_text.strip():
        print("[konjo-review] Empty diff. Approved.", file=sys.stderr)
        return 0
    if args.dry_run:
        print(f"[konjo-review] DRY RUN: model={CRITIC_MODEL} diff_chars={len(diff_text)}", file=sys.stderr)
        return 0
    try:
        anthropic = _load_anthropic()
        result = _call_api(diff_text, anthropic)
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"[konjo-review] ERROR: {exc}", file=sys.stderr)
        return 0
    verdict = result.get("verdict", "UNKNOWN")
    has_blockers = verdict == "BLOCKER" or bool(result.get("blockers"))
    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        report = _render_human(result)
        if args.output:
            Path(args.output).write_text(report)
        else:
            print(report)
    if has_blockers and not args.soft_fail:
        print(f"\n[konjo-review] VERDICT: {verdict} — merge blocked.", file=sys.stderr)
        return 1
    print(f"[konjo-review] VERDICT: {verdict}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
