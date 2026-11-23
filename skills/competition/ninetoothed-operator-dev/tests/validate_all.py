#!/usr/bin/env python3
"""
九齿 Skill 完整验证脚本。

一键运行：correctness 测试 + benchmark + 项目结构检查。

用法：
    # 完整验证（correctness + benchmark，快速模式）
    python tests/validate_all.py --quick

    # 仅 correctness
    python tests/validate_all.py --correctness-only

    # 仅 benchmark
    python tests/validate_all.py --benchmark-only

    # 仅结构检查
    python tests/validate_all.py --check-only
"""
import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXPECTED_FILES = [
    "SKILL.md",
    "README.md",
    "operators/__init__.py",
    "operators/add.py",
    "operators/add_2d.py",
    "operators/relu.py",
    "operators/sigmoid.py",
    "operators/gelu.py",
    "operators/softmax.py",
    "operators/sum.py",
    "operators/strided_add.py",
    "operators/rms_norm.py",
    "run_all_operator_tests.py",
    "benchmarks/bench.py",
    "benchmarks/run_benchmarks.py",
    "examples/usage_example.py",
    "assets/operator_template.py",
    "assets/test_template.py",
    "assets/benchmark_template.py",
    "references/nine_toothed_api.md",
    "references/operator_patterns.md",
    "references/migration_guide.md",
    "tests/README.md",
    "tests/self_eval_1_elementwise_gelu.md",
    "tests/self_eval_2_reduction_softmax.md",
    "tests/self_eval_3_noncontiguous_strided_add.md",
    "tests/self_eval_4_performance_analysis.md",
    "tests/self_eval_5_triton_migrate_add.md",
    "tests/bench_light.py",
    "tests/validate_all.py",
    "tests/skill_eval/test_prompts.md",
    "tests/skill_eval/leaky_relu_reference.py",
    "tests/skill_eval/hardswish_2d_reference.py",
    "tests/skill_eval/buggy_softmax.py",
    "proposal.md",
    "REFERENCE.md",
]


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def yellow(s):
    return f"\033[33m{s}\033[0m"


def bold(s):
    return f"\033[1m{s}\033[0m"


# ============================================================
# 1. 项目结构检查
# ============================================================
def check_structure():
    print(bold("\n" + "=" * 60))
    print(bold("1. 项目结构检查"))
    print("=" * 60)

    missing = []
    present = []

    for f in EXPECTED_FILES:
        path = os.path.join(ROOT, f)
        if os.path.exists(path):
            present.append(f)
            print(f"  {green('✓')} {f}")
        else:
            missing.append(f)
            print(f"  {red('✗')} {f}  -- 缺失")

    # 检查不应存在的文件
    forbidden = ["ntd_code"]
    for d in forbidden:
        path = os.path.join(ROOT, d)
        if os.path.exists(path):
            print(f"  {yellow('⚠')} {d}/ 应删除（官方代码副本，非 skill 组成部分）")

    # 统计
    print(f"\n  结果: {green(len(present))}/{len(EXPECTED_FILES)} 文件存在")
    if missing:
        print(f"  {red(f'缺失 {len(missing)} 个文件')}")
        for m in missing:
            print(f"    - {m}")
    return len(missing) == 0


# ============================================================
# 2. 语法检查
# ============================================================
def check_syntax():
    print(bold("\n" + "=" * 60))
    print(bold("2. Python 语法检查"))
    print("=" * 60)

    py_files = [
        "operators/__init__.py",
        "operators/add.py", "operators/add_2d.py",
        "operators/relu.py", "operators/sigmoid.py", "operators/gelu.py",
        "operators/softmax.py", "operators/sum.py",
        "operators/strided_add.py", "operators/rms_norm.py",
        "run_all_operator_tests.py",
        "benchmarks/bench.py", "benchmarks/run_benchmarks.py",
        "examples/usage_example.py",
        "assets/operator_template.py", "assets/test_template.py",
        "tests/bench_light.py", "tests/validate_all.py",
        "tests/skill_eval/leaky_relu_reference.py",
        "tests/skill_eval/hardswish_2d_reference.py",
        "tests/skill_eval/buggy_softmax.py",
    ]
    import ast

    failed = 0
    for f in py_files:
        path = os.path.join(ROOT, f)
        try:
            with open(path) as fh:
                ast.parse(fh.read())
            print(f"  {green('✓')} {f}")
        except SyntaxError as e:
            print(f"  {red('✗')} {f}: {e}")
            failed += 1

    print(f"\n  结果: {green(len(py_files) - failed)}/{len(py_files)} 通过")
    return failed == 0


# ============================================================
# 3. 导入检查
# ============================================================
def check_imports():
    print(bold("\n" + "=" * 60))
    print(bold("3. 模块导入检查"))
    print("=" * 60)

    try:
        from operators import (
            create_add_kernel, create_2d_add_kernel,
            create_relu_kernel, create_sigmoid_kernel, create_gelu_kernel,
            create_softmax_kernel, create_sum_kernel,
            create_strided_add_kernel, create_2d_strided_add_kernel,
            create_rms_norm_kernel,
        )
        kernels = {
            "add_1d": create_add_kernel,
            "add_2d": create_2d_add_kernel,
            "relu": create_relu_kernel,
            "sigmoid": create_sigmoid_kernel,
            "gelu": create_gelu_kernel,
            "softmax": create_softmax_kernel,
            "sum": create_sum_kernel,
            "strided_add": create_strided_add_kernel,
            "strided_add_2d": create_2d_strided_add_kernel,
            "rms_norm": create_rms_norm_kernel,
        }
        for name, fn in kernels.items():
            print(f"  {green('✓')} {name}")
        print(f"\n  结果: {green('10/10')} 算子工厂导入成功")
        return True
    except Exception as e:
        print(f"  {red('✗')} 导入失败: {e}")
        return False


# ============================================================
# 4. Correctness 测试
# ============================================================
def run_correctness():
    print(bold("\n" + "=" * 60))
    print(bold("4. Correctness 测试"))
    print("=" * 60)

    script = os.path.join(ROOT, "run_all_operator_tests.py")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        cwd=ROOT,
    )
    elapsed = time.time() - t0

    ok = result.returncode == 0
    print(f"\n  耗时: {elapsed:.1f}s")
    print(f"  结果: {green('全部通过') if ok else red('有失败')}")
    return ok


# ============================================================
# 5. Benchmark（轻量，避免 triton.testing.Benchmark 的 matplotlib 挂起）
# ============================================================
def run_benchmark(quick=True):
    print(bold("\n" + "=" * 60))
    print(bold("5. Benchmark (Stride vs Contiguous)"))
    print("=" * 60)

    script = os.path.join(os.path.dirname(__file__), "bench_light.py")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        cwd=ROOT,
    )
    elapsed = time.time() - t0

    ok = result.returncode == 0
    print(f"\n  耗时: {elapsed:.1f}s")
    print(f"  结果: {green('完成') if ok else red('有失败')}")
    return ok


# ============================================================
# 6. 一致性检查
# ============================================================
def check_consistency():
    print(bold("\n" + "=" * 60))
    print(bold("6. 代码规范一致性"))
    print("=" * 60)

    import glob as g

    issues = []
    op_files = g.glob(os.path.join(ROOT, "operators", "*.py"))
    op_files = [f for f in op_files if "__init__" not in f]

    # 检查 # noqa: F841
    for fpath in sorted(op_files):
        with open(fpath) as fh:
            content = fh.read()
        if "application" in content or "def " in content:
            # 检查 application 函数中是否有 output = 的行
            has_assignment = any(
                line.strip().startswith("output =")
                or line.strip().startswith("z =")
                or line.strip().startswith("y =")
                for line in content.split("\n")
            )
            has_noqa = "# noqa: F841" in content
            if has_assignment and not has_noqa:
                name = os.path.basename(fpath)
                issues.append(f"{name}: 缺 # noqa: F841")
                print(f"  {red('✗')} {name} 缺 # noqa: F841")
            else:
                print(f"  {green('✓')} {os.path.basename(fpath)}")

    if not issues:
        print(f"\n  结果: {green('全部一致')}")
    else:
        print(f"\n  结果: {red(f'{len(issues)} 个问题')}")
    return len(issues) == 0


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="九齿 Skill 验证")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（benchmark 用 quick sweep）")
    parser.add_argument("--correctness-only", action="store_true",
                        help="仅运行 correctness 测试")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="仅运行 benchmark")
    parser.add_argument("--check-only", action="store_true",
                        help="仅结构/语法/导入检查（不需要 CUDA）")
    args = parser.parse_args()

    t_start = time.time()

    print(bold("=" * 60))
    print(bold("九齿 Skill 验证套件"))
    print(bold("=" * 60))
    print(f"  项目: {ROOT}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 始终运行基础检查
    results["structure"] = check_structure()
    results["syntax"] = check_syntax()
    results["imports"] = check_imports()
    results["consistency"] = check_consistency()

    if not args.check_only:
        if not args.benchmark_only:
            results["correctness"] = run_correctness()
        if not args.correctness_only:
            results["benchmark"] = run_benchmark(quick=args.quick)

    # --- 汇总 ---
    elapsed = time.time() - t_start
    print(bold("\n" + "=" * 60))
    print(bold("验证汇总"))
    print("=" * 60)

    all_ok = True
    for name, ok in results.items():
        label = {
            "structure": "项目结构",
            "syntax": "Python 语法",
            "imports": "模块导入",
            "consistency": "代码规范",
            "correctness": "Correctness",
            "benchmark": "Benchmark",
        }.get(name, name)
        status = green("✓ PASS") if ok else red("✗ FAIL")
        print(f"  [{status}] {label}")
        if not ok:
            all_ok = False

    print(f"\n  总耗时: {elapsed:.1f}s")
    if all_ok:
        print(f"  {green(bold('全部通过!'))}")
    else:
        print(f"  {red(bold('存在问题，请检查以上 ✗ 项'))}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
