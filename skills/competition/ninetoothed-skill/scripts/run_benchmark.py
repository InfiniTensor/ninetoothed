"""
性能基准测试脚本（ntops vs PyTorch）。

用法：
    cd $PROJECT_ROOT
    python ninetoothed-skill/scripts/run_benchmark.py

说明：
    测试 7 个算子的性能对比，包括 element-wise、binary、reduction、matmul 类型。
    同时分析非连续输入的开销。
"""
import os
import sys

# 将测试目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

from test_benchmark import run_benchmarks


if __name__ == "__main__":
    results = run_benchmarks()
