#!/usr/bin/env bash
# run_self_tests.sh — regression check for the 4 shipped worked examples.
#
# NOT a with-skill/no-skill A/B: both "primary" and "repeat" runs execute the
# SAME pre-written examples/**/test_correctness.py (the shipped reference
# solution), so there is nothing here for a skill to help or hurt. Its only
# job is to catch regressions in the examples themselves (e.g. a NineToothed
# API change breaking a shipped kernel) and flag CUDA nondeterminism (a
# passing example should pass identically both times).
#
# The REAL with/without-skill comparison — a fresh agent solving each task
# once with the skill loaded and once without — lives in the companion
# evaluation harness repo (`run_comparison.sh`, blocks 1+2; see this skill's
# README's "What's inside" section for the link). That is the only comparison
# that counts for judging the skill; see that repo's RUNBOOK.md.
#
# Usage:
#   bash run_self_tests.sh             # run primary + repeat
#   bash run_self_tests.sh --ab-only   # correctness only, no benchmark tests
#   bash run_self_tests.sh --b-only    # skip the repeat run (faster)
#
# Exit code: 0 if all primary-run tests pass; 1 otherwise.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
PYTEST="$PYTHON -m pytest"

if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

SKIP_A=false
AB_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--b-only"  ]] && SKIP_A=true
  [[ "$arg" == "--ab-only" ]] && AB_ONLY=true
done

LOG_DIR="$SCRIPT_DIR/test_logs"
mkdir -p "$LOG_DIR"

declare -A EX_NAME=(
  [01]="Elementwise/mask"
  [02]="Reduction/mean"
  [03]="Layout/pixel_unshuffle"
  [04]="Perf/diag/softmax"
)
declare -A EX_PATH=(
  [01]="examples/01_elementwise_where_mask"
  [02]="examples/02_reduction_parameterized"
  [03]="examples/03_layout_pixel_unshuffle"
  [04]="examples/04_perf_diag_generated_source"
)

run_pytest() {
  local label="$1" test_path="$2" log="$3"
  shift 3
  $PYTEST "$test_path" -v --tb=short --no-header "$@" > "$log" 2>&1
}

status_of() {
  local log="$1"
  [[ ! -f "$log" ]] && echo "NOT_RUN" && return
  if grep -q "passed" "$log" && ! grep -qE "failed|error" "$log"; then
    echo "PASS"
  elif grep -qE "no tests ran|skipped" "$log" && ! grep -qE "failed|error" "$log"; then
    echo "SKIP"
  else
    echo "FAIL"
  fi
}

passed_count() {
  grep -oP '\d+ passed' "$1" 2>/dev/null | grep -oP '\d+' | head -1 || echo 0
}

echo "======================================================================"
echo "  ninetoothed-operator-dev  self-test runner"
echo "======================================================================"
echo ""

declare -A B_STATUS A_STATUS

echo "[ primary run ]"
for key in 01 02 03 04; do
  path="${EX_PATH[$key]}"
  log="$LOG_DIR/B_${key}.log"
  printf "  Example %s %-26s … " "$key" "${EX_NAME[$key]}"
  if run_pytest "B-${key}" "$SCRIPT_DIR/$path/test_correctness.py" "$log"; then
    B_STATUS[$key]="PASS"; echo "PASS  ($(passed_count "$log") passed)"
  else
    B_STATUS[$key]=$(status_of "$log"); echo "${B_STATUS[$key]}"
  fi
done
echo ""

if [[ "$SKIP_A" == "false" ]]; then
  echo "[ repeat run (nondeterminism check) ]"
  for key in 01 02 03 04; do
    path="${EX_PATH[$key]}"
    log="$LOG_DIR/A_${key}.log"
    printf "  Example %s %-26s … " "$key" "${EX_NAME[$key]}"
    if run_pytest "A-${key}" "$SCRIPT_DIR/$path/test_correctness.py" "$log"; then
      A_STATUS[$key]="PASS"; echo "PASS  ($(passed_count "$log") passed)"
    else
      A_STATUS[$key]=$(status_of "$log"); echo "${A_STATUS[$key]}"
    fi
  done
  echo ""
fi

echo "[ Scripts self-test (no CUDA required) ]"
SCRIPTS_LOG="$LOG_DIR/scripts.log"
$PYTHON -m py_compile \
  "$SCRIPT_DIR/scripts/inspect_generated_source.py" \
  "$SCRIPT_DIR/scripts/bench_compare.py" \
  "$SCRIPT_DIR/scripts/gen_pytorch_oracle.py" \
  "$SCRIPT_DIR/scripts/run_correctness_matrix.py" \
  >> "$SCRIPTS_LOG" 2>&1 \
  && echo "  py_compile all 4 scripts: PASS" \
  || echo "  py_compile: FAIL"

$PYTHON "$SCRIPT_DIR/scripts/bench_compare.py" >> "$SCRIPTS_LOG" 2>&1 \
  && echo "  bench_compare self-test:  PASS" \
  || echo "  bench_compare self-test:  FAIL"

bash -n "$SCRIPT_DIR/scripts/aot_build_smoke.sh" \
  && echo "  aot_build_smoke.sh syntax: PASS" \
  || echo "  aot_build_smoke.sh syntax: FAIL"

echo ""
echo "======================================================================"
echo "  Summary"
echo "======================================================================"
printf "  %-32s  %-8s  %-8s\n" "Example" "primary" "repeat"
echo "  ------------------------------------------------------------"
for key in 01 02 03 04; do
  bs="${B_STATUS[$key]:-NOT_RUN}"
  as="${A_STATUS[$key]:-NOT_RUN}"
  printf "  %-32s  %-8s  %-8s\n" "${key} ${EX_NAME[$key]}" "$bs" "$as"
done
echo ""

all_b_pass=true
for key in 01 02 03 04; do
  [[ "${B_STATUS[$key]:-FAIL}" != "PASS" ]] && all_b_pass=false
done

if $all_b_pass; then
  echo "  ✓ All shipped examples PASSED (primary run).  Logs: $LOG_DIR/"
  exit 0
else
  echo "  ✗ Some shipped examples FAILED (primary run).  See: $LOG_DIR/"
  exit 1
fi
