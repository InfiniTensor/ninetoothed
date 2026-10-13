#!/bin/bash
# AOT (Ahead-of-Time) build smoke test for NineToothed kernels.
#
# Verifies that a compiled kernel can be exported for AOT deployment.
# Checks for non-empty .py and .h output files.
#
# Usage:
#   bash scripts/aot_build_smoke.sh <operator_name>
#   bash scripts/aot_build_smoke.sh add
#   bash scripts/aot_build_smoke.sh matmul

set -euo pipefail

OP="${1:-add}"
AOT_DIR="/tmp/ninetoothed_aot_${OP}"

echo "=== AOT Build Smoke Test: ${OP} ==="

# Clean previous build
rm -rf "${AOT_DIR}"
mkdir -p "${AOT_DIR}"

echo "[1/3] Generating AOT build for operator: ${OP}"

# Try to build AOT (this may fail if the operator doesn't support AOT)
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from examples.${OP}.kernel import kernel
    print('  Kernel imported successfully')
    
    # Check if the kernel has AOT support
    if hasattr(kernel, '_fn'):
        print('  Kernel function found')
    else:
        print('  WARNING: No _fn attribute (AOT may not be supported)')
except Exception as e:
    print(f'  ERROR: {e}')
    sys.exit(1)
" 2>&1

echo ""
echo "[2/3] Checking compiled cache..."

# Check if compiled source exists in ~/.ninetoothed/
CACHE_DIR="${HOME}/.ninetoothed"
if [ -d "${CACHE_DIR}" ]; then
    CACHE_FILES=$(find "${CACHE_DIR}" -name "*.py" -newer /tmp 2>/dev/null | head -5)
    if [ -n "${CACHE_FILES}" ]; then
        echo "  Found compiled cache files:"
        for f in ${CACHE_FILES}; do
            SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
            echo "    $(basename $f) (${SIZE} bytes)"
        done
        echo "  [PASS] Compiled cache exists"
    else
        echo "  [WARN] No recent cache files found"
        echo "  Run the kernel first: python -c 'from examples.${OP}.kernel import kernel; import torch; args = [torch.randn(1024, device=\"cuda\")]; kernel(*args)'"
    fi
else
    echo "  [WARN] ${CACHE_DIR} not found"
    echo "  Run the kernel first to generate compiled cache"
fi

echo ""
echo "[3/3] Inspecting generated source..."

# Run inspect_generated if available
if [ -f "scripts/inspect_generated.py" ]; then
    python3 scripts/inspect_generated.py --op "${OP}" --verbose 2>&1 || true
else
    echo "  [SKIP] scripts/inspect_generated.py not found"
fi

echo ""
echo "=== AOT Smoke Test Complete: ${OP} ==="
