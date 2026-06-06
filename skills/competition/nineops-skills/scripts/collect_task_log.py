#!/usr/bin/env python3
"""
collect_task_log.py — 收集测试和 benchmark 日志，统一归档。

用法:
    python collect_task_log.py --output ./diagnose/
    python collect_task_log.py --task broadcast_add --format json

功能:
    1. 运行 correctness test，收集输出
    2. 运行 benchmark，收集结果
    3. 检查 generated source 并保存
    4. 汇总成诊断报告
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def run_command(cmd: list, timeout: int = 120) -> dict:
    """运行命令并返回输出。"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1, "success": False}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "Command not found", "returncode": -1, "success": False}


def collect_task(task_name: str, script_path: str, output_dir: str):
    """收集单个任务的日志。"""
    print(f"收集: {task_name}")
    result = run_command(["python", script_path])

    log_dir = os.path.join(output_dir, task_name)
    os.makedirs(log_dir, exist_ok=True)

    # 保存 stdout
    with open(os.path.join(log_dir, "stdout.log"), "w", encoding="utf-8") as f:
        f.write(result["stdout"])

    # 保存 stderr
    with open(os.path.join(log_dir, "stderr.log"), "w", encoding="utf-8") as f:
        f.write(result["stderr"])

    # 保存 summary
    summary = {
        "task": task_name,
        "timestamp": datetime.now().isoformat(),
        "success": result["success"],
        "returncode": result["returncode"],
        "stdout_lines": len(result["stdout"].splitlines()),
        "stderr_lines": len(result["stderr"].splitlines()),
    }
    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if result["success"]:
        print(f"  ✅ 成功 ({summary['stdout_lines']} 行输出)")
    else:
        print(f"  ⚠️  失败 (exit={result['returncode']})")
        if result["stderr"][:500]:
            print(f"  stderr: {result['stderr'][:500]}")

    return summary


def generate_report(output_dir: str, summaries: list):
    """生成汇总诊断报告。"""
    lines = [
        "# 任务日志诊断报告",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        f"工作目录: {os.getcwd()}",
        "",
        "## 摘要",
        "",
        "| 任务 | 状态 | 输出行数 |",
        "|------|------|----------|",
    ]

    success_count = 0
    for s in summaries:
        status = "✅" if s["success"] else "❌"
        if s["success"]:
            success_count += 1
        lines.append(f"| {s['task']} | {status} | {s['stdout_lines']} |")

    lines.extend([
        "",
        f"总计: {len(summaries)} 个任务, {success_count} 个成功, {len(summaries) - success_count} 个失败",
        "",
    ])

    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n诊断报告: {os.path.join(output_dir, 'report.md')}")


def main():
    parser = argparse.ArgumentParser(description="收集并归档 task 日志")
    parser.add_argument(
        "--output",
        default="diagnose_log",
        help="输出目录",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="只收集特定 task (broadcast_add, softmax 等)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式",
    )
    args = parser.parse_args()

    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(skill_dir, "examples")
    output_dir = os.path.abspath(args.output)

    os.makedirs(output_dir, exist_ok=True)

    # 定义要收集的任务
    all_tasks = [
        ("broadcast_add", os.path.join(examples_dir, "elementwise_broadcast_add", "run.py")),
        ("softmax", os.path.join(examples_dir, "reduction_softmax", "run.py")),
        ("non_contiguous", os.path.join(examples_dir, "non_contiguous_stride_case", "run.py")),
        ("performance_regression", os.path.join(examples_dir, "performance_regression_case", "run.py")),
    ]

    summaries = []

    for task_name, script_path in all_tasks:
        if args.task and args.task not in task_name:
            continue
        if os.path.exists(script_path):
            summary = collect_task(task_name, script_path, output_dir)
            summaries.append(summary)
        else:
            print(f"跳过: {task_name} ({script_path} 不存在)")

    if summaries:
        generate_report(output_dir, summaries)

    if args.format == "json":
        json_path = os.path.join(output_dir, "summary_all.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"JSON 汇总: {json_path}")

    print(f"\n所有日志已保存到: {output_dir}")


if __name__ == "__main__":
    main()
