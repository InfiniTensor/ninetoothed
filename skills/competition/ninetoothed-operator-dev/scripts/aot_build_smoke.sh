#!/usr/bin/env bash
# Smoke-test a NineToothed AOT build output directory.
# Checks that a .py launcher and a .h header exist and are non-empty.
# Usage: bash aot_build_smoke.sh <output_dir>
set -euo pipefail

DIR="${1:-build}"

if [ ! -d "$DIR" ]; then
  echo "FAIL: output dir '$DIR' does not exist" >&2
  exit 1
fi

py_count=$(find "$DIR" -maxdepth 1 -name '*.py' -size +0c | wc -l | tr -d ' ')
h_count=$(find "$DIR" -maxdepth 1 -name '*.h' -size +0c | wc -l | tr -d ' ')

echo "AOT output in '$DIR': ${py_count} non-empty .py, ${h_count} non-empty .h"

if [ "$py_count" -ge 1 ] && [ "$h_count" -ge 1 ]; then
  echo "PASS: AOT build produced launcher + header"
  exit 0
fi

echo "FAIL: expected >=1 non-empty .py and >=1 non-empty .h" >&2
echo "hint: pass num_warps/num_stages explicitly to make(..., caller='cuda') if JIT worked but AOT failed" >&2
exit 1
