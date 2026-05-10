#!/usr/bin/env bash
# Konjo Quality Framework — Hook Installer
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
GRN='\033[0;32m'; YEL='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'; BOLD='\033[1m'
ok()   { echo -e "${GRN}  ✓${RST} $1"; }
warn() { echo -e "${YEL}  ⚠${RST} $1"; }
err()  { echo -e "${RED}  ✗${RST} $1"; }
echo -e "${BOLD}Konjo Quality Framework — Install${RST}"
echo ""
HOOK_SRC="$REPO_ROOT/.konjo/hooks/pre-commit"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"
if [[ ! -f "$HOOK_SRC" ]]; then err ".konjo/hooks/pre-commit not found"; exit 1; fi
chmod +x "$HOOK_SRC"
[[ -L "$HOOK_DST" ]] && rm "$HOOK_DST"
ln -sf "../../.konjo/hooks/pre-commit" "$HOOK_DST"
ok "Installed .git/hooks/pre-commit → .konjo/hooks/pre-commit"
HAS_RUST=false; HAS_PYTHON=false
[[ -f "$REPO_ROOT/Cargo.toml" ]] && HAS_RUST=true
{ [[ -f "$REPO_ROOT/pyproject.toml" ]] || [[ -f "$REPO_ROOT/requirements.txt" ]]; } && HAS_PYTHON=true
echo ""; echo -e "${BOLD}Repo type:${RST}"
$HAS_RUST && ok "Rust" || true
$HAS_PYTHON && ok "Python" || true
echo ""; echo -e "${BOLD}Tool availability:${RST}"
ALL_PRESENT=true
check_tool() { local cmd="$1" hint="$2"; if command -v "$cmd" &>/dev/null; then ok "$cmd"; else err "$cmd not found — $hint"; ALL_PRESENT=false; fi; }
check_tool "python3" "install Python 3.10+"
check_tool "git" "install git"
$HAS_RUST && check_tool "cargo" "install Rust" || true
$HAS_PYTHON && check_tool "ruff" "pip install ruff" || true
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then ok "ANTHROPIC_API_KEY"; else warn "ANTHROPIC_API_KEY not set"; fi
echo ""
$ALL_PRESENT && echo -e "${GRN}${BOLD}All required tools present. Framework installed.${RST}" || echo -e "${YEL}${BOLD}Some tools missing.${RST}"
echo ""
echo "Next steps:"
echo "  1. Add ANTHROPIC_API_KEY to GitHub Actions secrets"
echo "  2. Add .github/workflows/konjo-gate.yml to enable Wall 2"
echo "  3. Run: git commit --allow-empty -m 'test: verify konjo hooks'"
