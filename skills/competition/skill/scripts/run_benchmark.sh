#!/bin/bash
# run_benchmark.sh — 运行 benchmark 套件
#
# 用法:
#   bash scripts/run_benchmark.sh                          # 运行所有 benchmark
#   bash scripts/run_benchmark.sh softmax                  # 运行特定 benchmark
#   bash scripts/run_benchmark.sh --output results.md      # 输出到文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$SKILL_DIR/examples"

# 解析参数
OUTPUT_FILE=""
VERBOSE=""
TARGET=""

for arg in "$@"; do
    case $arg in
        --output)
            shift
            OUTPUT_FILE="$1"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            TARGET="$arg"
            break
            ;;
    esac
done

echo "=========================================="
echo "  Ninetoothed Benchmark Runner"
echo "=========================================="
echo "Output: ${OUTPUT_FILE:-stdout}"
echo ""

benchmark_results=""

run_bench() {
    local label="$1"
    local script="$2"

    echo ""
    echo "--- Benchmark: $label ---"

    if [ -f "$script" ]; then
        local output
        output=$(python "$script" 2>&1)
        echo "$output"

        if [ -n "$OUTPUT_FILE" ]; then
            benchmark_results+="
## $label

\`\`\`
$output
\`\`\`
"
        fi
    else
        echo "⚠️  跳过: $script 不存在"
    fi
}

if [ -n "$TARGET" ]; then
    # 运行特定案例
    case "$TARGET" in
        elementwise|broadcast_add|add)
            run_bench "Element-wise Broadcast Add" "$EXAMPLES_DIR/elementwise_broadcast_add/benchmark.py"
            ;;
        softmax|reduction)
            run_bench "Reduction Softmax" "$EXAMPLES_DIR/reduction_softmax/benchmark.py"
            ;;
        non_contiguous)
            run_bench "Non-contiguous Stride Case" "$EXAMPLES_DIR/non_contiguous_stride_case/benchmark.py"
            ;;
        regression)
            run_bench "Performance Regression Case" "$EXAMPLES_DIR/performance_regression_case/benchmark.py"
            ;;
        *)
            echo "错误: 未知 target '$TARGET'"
            echo "可用: elementwise, softmax, non_contiguous, regression"
            exit 1
            ;;
    esac
else
    # 运行所有
    run_bench "Element-wise Broadcast Add" "$EXAMPLES_DIR/elementwise_broadcast_add/benchmark.py"
    run_bench "Reduction Softmax" "$EXAMPLES_DIR/reduction_softmax/benchmark.py"
    run_bench "Non-contiguous Stride Case" "$EXAMPLES_DIR/non_contiguous_stride_case/benchmark.py"
    run_bench "Performance Regression Case" "$EXAMPLES_DIR/performance_regression_case/benchmark.py"
fi

# 写入输出文件
if [ -n "$OUTPUT_FILE" ]; then
    {
        echo "# Benchmark Results"
        echo ""
        echo "Date: $(date)"
        echo "Host: $(hostname 2>/dev/null || echo 'unknown')"
        echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")' 2>/dev/null || echo 'unknown')"
        echo ""
        echo "$benchmark_results"
    } > "$OUTPUT_FILE"

    echo ""
    echo "结果已保存到: $OUTPUT_FILE"
fi

echo ""
echo "=========================================="
echo "  Benchmark Done."
echo "=========================================="
