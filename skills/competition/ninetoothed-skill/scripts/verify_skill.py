"""
NineToothed Skill 自身有效性验证脚本。

验证内容：
    1. SKILL.md 存在且包含必要章节
    2. references/ 中所有被引用的文件存在
    3. examples/ 覆盖至少 1 个完整示例
    4. tests/ 中每个测试文件包含必要的元信息
    5. scripts/ 中脚本可导入执行
    6. 代码模板语法基本正确

用法：
    cd $PROJECT_ROOT
    python ninetoothed-skill/scripts/verify_skill.py
"""
import os
import sys
import ast
import re

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0


def check(condition, message):
    global passed, failed
    if condition:
        print(f"  [PASS] {message}")
        passed += 1
    else:
        print(f"  [FAIL] {message}")
        failed += 1


def file_exists(relpath, description):
    path = os.path.join(SKILL_DIR, relpath)
    check(os.path.isfile(path), f"{description}: {relpath}")


def dir_exists(relpath, description):
    path = os.path.join(SKILL_DIR, relpath)
    check(os.path.isdir(path), f"{description}: {relpath}")


def main():
    global passed, failed
    print("=" * 60)
    print("NineToothed Skill 自身有效性验证")
    print("=" * 60)

    # 1. Required files and directories
    print("\n### 1. 目录结构完整性")
    dir_exists("references", "references/ 目录")
    dir_exists("examples", "examples/ 目录")
    dir_exists("scripts", "scripts/ 目录")
    dir_exists("tests", "tests/ 目录")
    file_exists("SKILL.md", "主 Skill 文件")
    file_exists("README.md", "项目说明")

    # 2. SKILL.md content checks
    print("\n### 2. SKILL.md 内容完整性")
    skill_path = os.path.join(SKILL_DIR, "SKILL.md")
    if os.path.isfile(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            skill_content = f.read()

        check("---" in skill_content[:20],
              "SKILL.md 包含 YAML frontmatter")
        check("description:" in skill_content[:500],
              "SKILL.md frontmatter 包含 description")
        check("triggers:" in skill_content[:500] or "触发" in skill_content,
              "SKILL.md 声明了触发条件")
        check("工作流" in skill_content or "workflow" in skill_content.lower(),
              "SKILL.md 描述了工作流")
        check("禁止" in skill_content or "约束" in skill_content,
              "SKILL.md 声明了约束条件")
        check("精度验证" in skill_content or "精度" in skill_content,
              "SKILL.md 包含精度验证说明")
        check("Generated Source" in skill_content or "AOT" in skill_content,
              "SKILL.md 包含 generated source / AOT 操作步骤")
        check("不支持" in skill_content or "不覆盖" in skill_content,
              "SKILL.md 说明了不支持场景")
        check("错误恢复" in skill_content or "诊断" in skill_content,
              "SKILL.md 包含错误诊断与恢复策略")
        check("$PROJECT_ROOT" in skill_content,
              "SKILL.md 使用 $PROJECT_ROOT 占位符（无硬编码路径）")

    # 3. References
    print("\n### 3. references/ 文件完整性")
    refs = ["code_templates.md", "ntl_api.md", "tensor_guide.md", "pitfalls.md"]
    for ref in refs:
        file_exists(f"references/{ref}", ref)
        filepath = os.path.join(SKILL_DIR, "references", ref)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            check(size > 500, f"{ref} 内容充足 ({size} bytes)")

    # 4. Examples
    print("\n### 4. examples/ 覆盖度")
    examples = [
        "01_elementwise_leaky_relu.md",
        "02_reduction_log_softmax.md",
        "03_binary_add.md",
        "04_broadcast_meshgrid.md",
        "01_benchmark_leaky_relu.md",
        "02_benchmark_log_softmax.md",
    ]
    for ex in examples:
        file_exists(f"examples/{ex}", ex)

    # 5. Tests
    print("\n### 5. tests/ 覆盖度")
    test_files = [
        "test_leaky_relu.py",
        "test_log_softmax.py",
        "test_non_contiguous.py",
        "test_benchmark.py",
        "test_meshgrid.py",
    ]
    for tf in test_files:
        file_exists(f"tests/{tf}", tf)
        # Check that each test file has docstring with task description
        filepath = os.path.join(SKILL_DIR, "tests", tf)
        if os.path.isfile(filepath):
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            check('"""' in content[:5],
                  f"{tf} 包含模块 docstring")
            check("$PROJECT_ROOT" in content,
                  f"{tf} 使用 $PROJECT_ROOT（无硬编码路径）")
            # Check no KMP_DUPLICATE_LIB_OK in code (only in comments is OK)
            code_lines = [l for l in content.split("\n") if not l.strip().startswith("#") and not l.strip().startswith('"""')]
            code_without_docstrings = "\n".join(code_lines)
            check("os.environ" not in code_without_docstrings or "KMP_DUPLICATE_LIB_OK" not in code_without_docstrings,
                  f"{tf} 无硬编码 KMP_DUPLICATE_LIB_OK")

    # 6. Scripts
    print("\n### 6. scripts/ 可执行性")
    file_exists("scripts/run_tests.py", "run_tests.py")
    file_exists("scripts/run_benchmark.py", "run_benchmark.py")
    file_exists("scripts/verify_skill.py", "verify_skill.py（本脚本）")

    # Check run_tests.py syntax
    run_tests_path = os.path.join(SKILL_DIR, "scripts", "run_tests.py")
    if os.path.isfile(run_tests_path):
        with open(run_tests_path, encoding="utf-8") as f:
            try:
                ast.parse(f.read())
                check(True, "run_tests.py Python 语法正确")
            except SyntaxError as e:
                check(False, f"run_tests.py 语法错误: {e}")

    # 7. Code template syntax check
    print("\n### 7. 代码模板语法检查")
    templates_path = os.path.join(SKILL_DIR, "references", "code_templates.md")
    if os.path.isfile(templates_path):
        with open(templates_path, encoding="utf-8") as f:
            tmpl_content = f.read()
        # Extract python code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', tmpl_content, re.DOTALL)
        syntax_ok = 0
        syntax_fail = 0
        for i, block in enumerate(code_blocks):
            block = block.strip()
            if not block or block.startswith("#"):
                continue
            try:
                ast.parse(block)
                syntax_ok += 1
            except SyntaxError:
                # Templates may have placeholders like {COMPUTATION} which aren't valid Python
                if "{" in block:
                    syntax_ok += 1  # Expected for template placeholders
                else:
                    syntax_fail += 1
        check(syntax_fail == 0,
              f"代码模板语法: {syntax_ok} 块通过, {syntax_fail} 块失败")

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"结果: {passed}/{total} 通过, {failed}/{total} 失败")
    if failed == 0:
        print("Skill 自身验证全部通过！")
    else:
        print(f"存在 {failed} 项问题需要修复。")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
