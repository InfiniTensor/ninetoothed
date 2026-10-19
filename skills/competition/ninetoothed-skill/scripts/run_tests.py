"""
运行所有自测算子验证测试。

用法：
    cd $PROJECT_ROOT
    python ninetoothed-skill/scripts/run_tests.py

或单独运行某个测试：
    python -m pytest ninetoothed-skill/tests/test_leaky_relu.py -v
"""
import os
import subprocess
import sys


TEST_FILES = [
    ("test_leaky_relu.py", "任务 1：Element-wise 算子（leaky_relu）"),
    ("test_log_softmax.py", "任务 2：Reduction 算子（log_softmax）"),
    ("test_non_contiguous.py", "任务 3：非连续输入/步长场景"),
    ("test_meshgrid.py", "任务 5：广播算子（meshgrid）"),
]

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "tests")


def run_all_tests():
    print("=" * 60)
    print("NineToothed Skill — 自测算子验证")
    print("=" * 60)

    all_passed = True
    for filename, description in TEST_FILES:
        print(f"\n>>> {description}")
        print(f"    文件: {filename}")
        filepath = os.path.join(TESTS_DIR, filename)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", filepath, "-v"],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            all_passed = False
            print(f"    结果: FAILED")
        else:
            print(f"    结果: PASSED")

    # Benchmark
    print(f"\n>>> 任务 4：性能对比")
    benchmark_path = os.path.join(TESTS_DIR, "test_benchmark.py")
    result = subprocess.run(
        [sys.executable, benchmark_path],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)

    print("\n" + "=" * 60)
    if all_passed:
        print("所有自测任务通过！")
    else:
        print("部分测试失败，请检查输出。")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
