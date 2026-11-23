#!/usr/bin/env bash
# Run the full NineToothed local CI check sequence before committing.
# Mirrors the sequence documented in the repo CONTRIBUTING.md so an agent can
# self-verify style + tests in one call. Run from the repository root.
#
# Usage:
#   bash skills/competition/ninetoothed-operator-skill/scripts/run_ci_checks.sh [pytest-args...]
#
# Any extra arguments are forwarded to the final pytest invocation, e.g.
#   ... run_ci_checks.sh tests/test_add.py -q
#
# Exit code is non-zero if any step fails. Never bypass a failure by editing this
# script; fix the underlying issue (see references/failure-recovery.md).

set -u

fail=0

run_step() {
    local label="$1"
    shift
    echo "=========================================================="
    echo ">>> ${label}"
    echo ">>> \$ $*"
    echo "=========================================================="
    if "$@"; then
        echo "[OK] ${label}"
    else
        echo "[FAIL] ${label}"
        fail=1
    fi
    echo ""
}

if [ ! -d "src/ninetoothed" ]; then
    echo "ERROR: run this from the NineToothed repository root (src/ninetoothed not found)." >&2
    exit 2
fi

# 1. Apply mechanical blank-line fixes first, then re-check.
run_step "check_contributing_style --fix" python scripts/check_contributing_style.py --fix
# 2. Format and lint.
run_step "ruff format" ruff format
run_step "ruff check" ruff check
# 3. Project-specific style checker (no --fix, must pass clean).
run_step "check_contributing_style" python scripts/check_contributing_style.py
# 4. Tests. Extra args (paths, -q, -k ...) are forwarded here.
run_step "pytest" pytest "$@"

echo "=========================================================="
if [ "${fail}" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
else
    echo "ONE OR MORE CHECKS FAILED — see output above. Do not fake a pass."
fi
echo "=========================================================="
exit "${fail}"
