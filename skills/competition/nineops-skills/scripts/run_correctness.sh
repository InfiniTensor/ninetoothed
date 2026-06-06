#!/bin/bash
# run_correctness.sh — 运行 correctness 测试套件
#
# 用法:
#   bash scripts/run_correctness.sh                    # 运行所有 correctness 测试
#   bash scripts/run_correctness.sh test_broadcast_add # 运行特定测试
#   bash scripts/run_correctness.sh --verbose          # 详细输出

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$SKILL_DIR/tests"

# 解析参数
VERBOSE=""
TARGET=""

for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE="-v"
            shift
            ;;
        *)
            TARGET="$arg"
            break
            ;;
    esac
done

echo "=========================================="
echo "  Ninetoothed Correctness Test Runner"
echo "=========================================="
echo "SKILL_DIR:  $SKILL_DIR"
echo "TARGET:     ${TARGET:-all}"
echo ""

# 如果指定了具体测试，尝试多种模式
if [ -n "$TARGET" ]; then
    if [ -f "$TESTS_DIR/${TARGET}.py" ]; then
        echo "运行 ${TARGET}.py..."
        python "$TESTS_DIR/${TARGET}.py"
    elif [ -f "$TESTS_DIR/test_${TARGET}.py" ]; then
        echo "运行 test_${TARGET}.py..."
        python "$TESTS_DIR/test_${TARGET}.py"
    elif [ -f "${TARGET}" ]; then
        echo "运行 ${TARGET}..."
        python "${TARGET}"
    else
        echo "错误: 找不到测试 '$TARGET'"
        echo ""
        echo "可用测试:"
        for f in "$TESTS_DIR"/*.py; do
            echo "  $(basename "$f" .py)"
        done
        exit 1
    fi
else
    # 运行所有以 test_ 开头的文件
    found=0
    for test_file in "$TESTS_DIR"/test_*.py; do
        if [ -f "$test_file" ]; then
            echo "运行 $(basename "$test_file")..."
            python $VERBOSE "$test_file"
            echo ""
            found=$((found + 1))
        fi
    done

    if [ $found -eq 0 ]; then
        echo "⚠️  未找到任何 test_*.py 文件。请先在 tests/ 下创建测试。"
        echo ""
        echo "运行 self-test tasks（描述性检查）:"
        python -c "
import os
tasks_path = os.path.join('$TESTS_DIR', 'selftest_tasks.md')
if os.path.exists(tasks_path):
    with open(tasks_path, 'r') as f:
        print(f.read())
else:
    print('selftest_tasks.md 不存在')
"
    else
        echo "测试完成: $found 个文件执行。"
    fi
fi

echo ""
echo "=========================================="
echo "  Done."
echo "=========================================="
