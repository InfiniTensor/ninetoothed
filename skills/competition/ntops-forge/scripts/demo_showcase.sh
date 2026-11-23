#!/bin/bash
# ntops-forge 答辩演示：依次跑功能点，输出保存到 /tmp/showcase/
# 用法：source /root/miniconda3/bin/activate base && cd /root/work/skill && bash scripts/demo_showcase.sh

set -e
NTOPS="${NTOPS_ROOT:-/root/work/ntops}"
OUT="/tmp/showcase"
mkdir -p "$OUT"

run() {
  local name="$1"
  shift
  echo ""
  echo "========== $name =========="
  "$@" 2>&1 | tee "$OUT/${name}.log"
}

echo "Showcase logs -> $OUT"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo N/A)"
echo "Python: $(which python) $(python --version 2>&1)"

run "01-doctor"           python scripts/doctor.py
run "02-forge-list"       python scripts/forge.py list
run "03-forge-run-silu"   python scripts/forge.py run silu --ntops-root "$NTOPS"
run "04-forge-gate"       python scripts/forge.py gate --ntops-root "$NTOPS"
run "05-ab-report"        python scripts/eval_ab.py --input docs/ab_runs.csv --output docs/AB_Report.md
run "06-ab-cat"           cat docs/AB_Report.md
run "07-baseline-triton"  bash -c 'cat > /tmp/baseline_triton.py << EOF
import triton
import triton.language as tl
@triton.jit
def bad_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pass
EOF
python scripts/preflight.py /tmp/baseline_triton.py --kernel --strict; echo exit=$?'
run "08-compare-ref"      python scripts/compare_ref.py /tmp/silu_forge_kernel.py --ref "$NTOPS/src/ntops/kernels/silu.py"
run "09-forge-diagnose"   python scripts/forge_diagnose.py --text "No module named pytest"
run "10-copilot-finish"   python scripts/run_task.py --task silu --ntops-root "$NTOPS" --finish
run "11-forge-spec"       python scripts/forge_spec.py "relu unary max zero" --out /tmp/custom_relu.yaml
run "12-audit-jsonl"      bash -c 'tail -3 docs/forge_runs.jsonl 2>/dev/null || echo "(no jsonl yet)"'

echo ""
echo "=== DONE ==="
echo "截图对照表见 docs/DemoShowcase.md"
echo "日志目录: $OUT"
ls -la "$OUT"
