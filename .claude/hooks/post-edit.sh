#!/bin/bash
FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("file_path",""))' 2>/dev/null)
[[ -z "$FILE" ]] && exit 0
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || exit 0
case "$FILE" in
  *.rs) echo "→ cargo check (triggered by edit to $FILE)"; cargo check --quiet 2>&1 | head -30 ;;
  *.py) if command -v ruff &>/dev/null; then echo "→ ruff check"; ruff check "$FILE" --quiet 2>&1 | head -20; fi ;;
  *.ts|*.tsx) echo "→ tsc"; npx tsc --noEmit 2>&1 | head -20 ;;
esac
