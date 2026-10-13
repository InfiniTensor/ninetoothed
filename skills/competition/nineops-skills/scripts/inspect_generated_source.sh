#!/bin/bash
# inspect_generated_source.sh — 查看 ninetoothed 生成的 Triton source
#
# 用法:
#   bash scripts/inspect_generated_source.sh              # 交互选择
#   bash scripts/inspect_generated_source.sh broadcast_add # 直接指定
#   bash scripts/inspect_generated_source.sh --save output.txt  # 保存到文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLES_DIR="$SKILL_DIR/examples"

SAVE_FILE=""
TARGET=""

for arg in "$@"; do
    case $arg in
        --save)
            shift
            SAVE_FILE="$1"
            shift
            ;;
        *)
            TARGET="$arg"
            break
            ;;
    esac
done

echo "=========================================="
echo "  Generated Source Inspector"
echo "=========================================="
echo ""

inspect_source() {
    local label="$1"
    local script="$2"

    echo "--- $label ---"

    if [ ! -f "$script" ]; then
        echo "⚠️  跳过: $script 不存在"
        return
    fi

    # 运行脚本并添加环境变量来启用 debug 输出
    local output
    output=$(NINETOOTHED_DEBUG=1 python "$script" 2>&1 || true)

    # 尝试提取 generated source 部分
    local source
    source=$(echo "$output" | awk '/Generated Source:/,/^$/')

    if [ -z "$source" ]; then
        source=$(echo "$output" | awk '/TRITON_KERNEL/,/^END_KERNEL/')
    fi

    if [ -z "$source" ]; then
        source="$output"  # fallback: 显示全部
    fi

    echo "$source"
    echo ""

    if [ -n "$SAVE_FILE" ]; then
        {
            echo "=== $label ==="
            echo "$source"
            echo ""
        } >> "$SAVE_FILE"
    fi
}

if [ -n "$TARGET" ]; then
    case "$TARGET" in
        broadcast_add|add|elementwise)
            inspect_source "Broadcast Add" "$EXAMPLES_DIR/elementwise_broadcast_add/run.py"
            ;;
        softmax)
            inspect_source "Softmax" "$EXAMPLES_DIR/reduction_softmax/run.py"
            ;;
        non_contiguous)
            inspect_source "Non-contiguous Add" "$EXAMPLES_DIR/non_contiguous_stride_case/run.py"
            ;;
        regression)
            inspect_source "Performance Regression" "$EXAMPLES_DIR/performance_regression_case/run.py"
            ;;
        *)
            echo "错误: 未知 target '$TARGET'"
            echo "可用: broadcast_add, softmax, non_contiguous, regression"
            exit 1
            ;;
    esac
else
    inspect_source "Broadcast Add" "$EXAMPLES_DIR/elementwise_broadcast_add/run.py"
    inspect_source "Softmax" "$EXAMPLES_DIR/reduction_softmax/run.py"
    inspect_source "Non-contiguous Add" "$EXAMPLES_DIR/non_contiguous_stride_case/run.py"
    inspect_source "Performance Regression" "$EXAMPLES_DIR/performance_regression_case/run.py"
fi

echo "---"
if [ -n "$SAVE_FILE" ]; then
    echo "已保存到: $SAVE_FILE"
fi
echo "Inspect Done."
