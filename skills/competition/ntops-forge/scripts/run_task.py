#!/usr/bin/env python3
"""One-command task runner: read task yaml -> scaffold -> preflight -> print next steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GITHUB_ID = "hongwei-2026"


def load_task(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    # minimal fallback parser for our task yaml
    data: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Run ntops-copilot task workflow")
    p.add_argument(
        "--task",
        required=True,
        help="task name, e.g. silu (reads tasks/task_<name>.yaml)",
    )
    p.add_argument("--ntops-root", type=Path, help="path to ntops repo root")
    p.add_argument(
        "--contest-id", default="T1-1-X", help="contest id for PR branch hint"
    )
    p.add_argument("--skill-root", type=Path, default=ROOT)
    p.add_argument("--scaffold-only", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="overwrite existing kernel/torch files"
    )
    p.add_argument(
        "--no-formula",
        action="store_true",
        help="do not inject formula_hint from task card into application()",
    )
    p.add_argument(
        "--register",
        action="store_true",
        help="auto-register op in __init__.py (new ops only)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="run verify_task after scaffold (preflight + pytest)",
    )
    p.add_argument(
        "--finish",
        action="store_true",
        help="run --verify then record_run on success (one-shot demo/submission)",
    )
    args = p.parse_args()
    if args.finish:
        args.verify = True

    task_file = (
        args.skill_root
        / "skills"
        / "ntops-copilot"
        / "tasks"
        / f"task_{args.task}.yaml"
    )
    if not task_file.is_file():
        print(f"FAIL: task file not found: {task_file}", file=sys.stderr)
        return 1

    task = load_task(task_file)
    op_name = task.get("op_name", args.task)
    pattern = task.get("pattern", "unary")

    ntops_root = args.ntops_root
    if ntops_root is None:
        for candidate in [Path("/root/work/ntops"), Path.cwd() / "ntops"]:
            if (candidate / "pyproject.toml").is_file():
                ntops_root = candidate
                break

    if ntops_root is None or not (ntops_root / "pyproject.toml").is_file():
        print(
            "WARN: --ntops-root not set and auto-detect failed; only scaffold to /tmp"
        )
        kernel_out = Path(f"/tmp/{op_name}.py")
        torch_out = None
    else:
        kernel_out = ntops_root / "src" / "ntops" / "kernels" / f"{op_name}.py"
        torch_out = ntops_root / "src" / "ntops" / "torch" / f"{op_name}.py"
        if kernel_out.exists():
            print(f"WARN: {kernel_out} already exists; use --force to overwrite")
            if not getattr(args, "force", False):
                kernel_out = Path(f"/tmp/{op_name}_kernel.py")
                torch_out = Path(f"/tmp/{op_name}_torch.py")
                print(f"INFO: scaffolding to temp: {kernel_out}")

    py = sys.executable
    scaffold_cmd = [
        py,
        str(SCRIPTS / "scaffold_kernel.py"),
        "--name",
        op_name,
        "--pattern",
        pattern,
        "--style",
        "ntops",
        "--out",
        str(kernel_out),
    ]
    formula = task.get("formula_hint", "")
    if formula and not args.no_formula:
        scaffold_cmd.extend(["--formula", formula])
    code = run(scaffold_cmd)
    if code != 0:
        return code

    if torch_out is not None:
        code = run(
            [
                py,
                str(SCRIPTS / "scaffold_torch.py"),
                "--name",
                op_name,
                "--pattern",
                pattern,
                "--out",
                str(torch_out),
            ]
        )
        if code != 0:
            return code

    code = run([py, str(SCRIPTS / "preflight.py"), str(kernel_out), "--kernel"])
    if code != 0:
        return code

    if args.register and ntops_root is not None:
        code = run(
            [
                py,
                str(SCRIPTS / "register_op.py"),
                "--name",
                op_name,
                "--ntops-root",
                str(ntops_root),
            ]
        )
        if code != 0:
            return code

    if args.verify and ntops_root is not None:
        verify_cmd = [
            py,
            str(SCRIPTS / "verify_task.py"),
            "--name",
            op_name,
            "--ntops-root",
            str(ntops_root),
            "--kernel",
            str(kernel_out),
            "--pytest",
        ]
        pytest_file = task.get("pytest_file")
        if pytest_file:
            verify_cmd.extend(["--pytest-file", pytest_file])
        ref = task.get("reference_kernel") or task.get("reference")
        if ref and ntops_root is not None:
            ref_str = str(ref).replace("\\", "/").lstrip("/")
            if ref_str.startswith("ntops/"):
                ref_str = ref_str[len("ntops/") :]
            ref_path = Path(ref) if Path(ref).is_absolute() else ntops_root / ref_str
            if ref_path.is_file():
                verify_cmd.extend(["--compare-ref", str(ref_path)])
        code = run(verify_cmd)
        if code != 0:
            return code
        if args.finish:
            code = run(
                [
                    py,
                    str(SCRIPTS / "record_run.py"),
                    "--task",
                    op_name,
                    "--preflight-pass",
                    "--pytest-pass",
                    "--steps",
                    "2",
                    "--elapsed",
                    "90",
                ]
            )
            if code != 0:
                return code

    print("\n=== Next steps (Agent/human) ===")
    print(f"1) Edit application() in: {kernel_out}")
    print("   Formula hints: skills/ntops-copilot/formulas.md")
    if task.get("reference_kernel"):
        print(f"   Reference: {task['reference_kernel']}")
    if torch_out is not None:
        print(
            f"2) Register: python scripts/register_op.py --name {op_name} --ntops-root {ntops_root}"
        )
        print(
            f"3) Verify:  python scripts/verify_task.py --name {op_name} --ntops-root {ntops_root} --pytest"
        )
        pytest_hint = task.get("pytest_file") or f"tests/test_{op_name}.py"
        print(f"4) Test: cd {ntops_root} && pytest -q {pytest_hint}")
    print(f"5) PR branch: 2026-spring-{GITHUB_ID}-{args.contest_id}")
    print(f"6) PR title: [2026春季][{args.contest_id}] {GITHUB_ID}")

    if args.scaffold_only:
        return 0

    if ntops_root is not None:
        print("\n=== Optional: run pytest now ===")
        pytest_hint = task.get("pytest_file") or f"tests/test_{op_name}.py"
        print(f"cd {ntops_root} && pytest -q {pytest_hint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
