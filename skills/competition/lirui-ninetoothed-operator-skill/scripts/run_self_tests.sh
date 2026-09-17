#!/usr/bin/env bash
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SKILL_DIR="$ROOT/skills/competition/lirui-ninetoothed-operator-skill"

echo "NineToothed skill self-test checklist"
echo "Repository root: $ROOT"
echo "Skill directory: $SKILL_DIR"
echo
echo "Recommended non-destructive checks:"
echo "1. ruff format --check"
echo "2. ruff check"
echo "3. python scripts/check_contributing_style.py"
echo "4. pytest"
echo
echo "Run commands manually when the local environment is ready:"
echo "ruff format --check"
echo "ruff check"
echo "python scripts/check_contributing_style.py"
echo "pytest"
echo
echo "For focused operator work, replace pytest with a narrower command, for example:"
echo "pytest tests/test_matmul.py -q"
echo
echo "After running tests or benchmarks, collect logs with:"
echo "python skills/competition/lirui-ninetoothed-operator-skill/scripts/collect_logs.py --input <log-file>"

