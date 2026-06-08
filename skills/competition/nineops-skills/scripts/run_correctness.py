#!/usr/bin/env python3
"""
run_correctness.py — 跨平台正确性测试运行器（Python 实现）

替代 run_correctness.sh，在 Windows/Linux/Mac 上均可运行。

用法:
    python scripts/run_correctness.py                    # 运行所有测试
    python scripts/run_correctness.py softmax            # 运行 examples 中指定算子的测试
    python scripts/run_correctness.py --verbose          # 详细输出
    python scripts/run_correctness.py --file path/to/test.py  # 运行指定文件
"""

import argparse
import os
import subprocess
import sys
import time


def get_skill_dir():
    """获取 skill 包根目录。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def discover_example_tests(skill_dir):
    """扫描 examples/ 下所有 run.py 和 benchmark.py。"""
    examples_dir = os.path.join(skill_dir, "examples")
    tests = []
    if os.path.isdir(examples_dir):
        for name in os.listdir(examples_dir):
            example_dir = os.path.join(examples_dir, name)
            if os.path.isdir(example_dir):
                for fname in ("run.py", "benchmark.py"):
                    fpath = os.path.join(example_dir, fname)
                    if os.path.isfile(fpath):
                        tests.append((f"{name}/{fname}", fpath))
    return tests


def discover_tests_dir_tests(skill_dir):
    """扫描 tests/ 目录下的 .py 文件。"""
    tests_dir = os.path.join(skill_dir, "tests")
    tests = []
    if os.path.isdir(tests_dir):
        for fname in sorted(os.listdir(tests_dir)):
            if fname.endswith(".py"):
                fpath = os.path.join(tests_dir, fname)
                tests.append((f"tests/{fname}", fpath))
    return tests


def run_test(label, fpath, verbose=False):
    """运行单个测试文件并返回结果。"""
    if not os.path.isfile(fpath):
        return {"label": label, "success": False, "reason": "文件不存在"}

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, fpath],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - start
        success = result.returncode == 0

        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr

        return {
            "label": label,
            "fpath": fpath,
            "success": success,
            "returncode": result.returncode,
            "output": output,
            "elapsed": elapsed,
            "reason": None,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "label": label,
            "fpath": fpath,
            "success": False,
            "returncode": -1,
            "output": "",
            "elapsed": elapsed,
            "reason": "TIMEOUT (>300s)",
        }
    except Exception as e:
        return {
            "label": label,
            "fpath": fpath,
            "success": False,
            "returncode": -1,
            "output": str(e),
            "elapsed": 0,
            "reason": f"EXCEPTION: {e}",
        }


def print_result(result, verbose=False):
    """格式化输出测试结果。"""
    status = "✅" if result["success"] else "❌"
    elapsed_s = f"({result['elapsed']:.1f}s)" if result["elapsed"] else ""
    print(f"  {status} {result['label']} {elapsed_s}")

    if not result["success"]:
        if result["reason"]:
            print(f"     原因: {result['reason']}")
        # 提取关键错误信息
        output = result.get("output", "")
        # 只显示最后几行错误
        lines = output.splitlines()
        error_lines = [l for l in lines if any(
            kw in l.lower() for kw in ["error", "traceback", "fail", "assert"])
        ]
        if error_lines:
            for l in error_lines[-5:]:
                print(f"     {l}")
        elif verbose and lines:
            for l in lines[-10:]:
                print(f"     {l}")

    if verbose and result["success"]:
        lines = result.get("output", "").splitlines()
        for l in lines[-5:]:
            if l.strip():
                print(f"     {l}")


def main():
    parser = argparse.ArgumentParser(
        description="Ninetoothed Correctness Test Runner (Cross-platform)",
    )
    parser.add_argument("target", nargs="?", default=None,
                        help="要运行的测试名（如 softmax, elementwise_broadcast_add）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="直接运行指定文件")
    args = parser.parse_args()

    skill_dir = get_skill_dir()

    # 收集测试列表
    all_tests = []
    all_tests.extend(discover_example_tests(skill_dir))
    all_tests.extend(discover_tests_dir_tests(skill_dir))

    if not all_tests:
        print("⚠️  未找到任何测试文件。")
        sys.exit(0)

    # 如果指定了 --file，只运行该文件
    if args.file:
        test_path = args.file
        if not os.path.isabs(test_path):
            test_path = os.path.join(skill_dir, test_path)
        label = os.path.relpath(test_path, skill_dir)
        all_tests = [(label, test_path)]

    # 如果指定了 target 且不是 --file，匹配名称
    if args.target and not args.file:
        target_lower = args.target.lower()
        matched = []
        for label, fpath in all_tests:
            if target_lower in label.lower():
                matched.append((label, fpath))
        if not matched:
            print(f"⚠️  未找到匹配 '{args.target}' 的测试。可用测试:")
            for label, _ in all_tests:
                print(f"    - {label}")
            sys.exit(1)
        all_tests = matched

    # 运行测试
    print("=" * 56)
    print("  Ninetoothed Correctness Test Runner")
    print("=" * 56)
    print(f"  Skill 目录: {skill_dir}")
    print(f"  测试数量:   {len(all_tests)}")
    if args.verbose:
        for label, fpath in all_tests:
            print(f"    - {label} ({fpath})")
    print()

    results = []
    for label, fpath in all_tests:
        print(f"  ▶ 正在运行: {label}")
        result = run_test(label, fpath, verbose=args.verbose)
        print_result(result, verbose=args.verbose)
        results.append(result)
        print()

    # 汇总
    passed = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    total = len(results)

    print("=" * 56)
    print(f"  结果: ✅ {passed} 通过 | ❌ {failed} 失败 | 共 {total}")
    print("=" * 56)

    if failed > 0:
        print("\n失败项:")
        for r in results:
            if not r["success"]:
                print(f"  ❌ {r['label']}")
        print("\n💡 提示: 打开 references/failure_diagnosis.md 查找对应错误的修复方法。")
        sys.exit(1)
    else:
        print("\n🎉 所有测试全部通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
