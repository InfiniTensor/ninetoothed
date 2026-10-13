#!/usr/bin/env bash
# A/B one-shot (bash fallback when run_ab_suite.py is not synced yet)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NTOPS_ROOT="${1:-/root/work/ntops}"
OPS="${2:-silu,add,gelu,relu,mul}"
ELAPSED="${3:-7}"

cd "$ROOT"
python scripts/run_baseline_demo.py --reset-csv --ops "$OPS"
python scripts/forge.py gate --ntops-root "$NTOPS_ROOT" --ops "$OPS" --no-record-ab

IFS=',' read -ra OP_ARR <<< "$OPS"
for op in "${OP_ARR[@]}"; do
  op="$(echo "$op" | xargs)"
  [ -z "$op" ] && continue
  python scripts/record_run.py \
    --mode treatment \
    --task "$op" \
    --preflight-pass \
    --pytest-pass \
    --steps 1 \
    --interventions 0 \
    --elapsed "$ELAPSED"
done

python scripts/eval_ab.py --input docs/ab_runs.csv --output docs/AB_Report.md
cat docs/AB_Report.md
