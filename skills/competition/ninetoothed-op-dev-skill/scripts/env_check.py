#!/usr/bin/env python3
"""Offline environment check for a NineToothed repository + this skill package."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _paths import (
    add_repo_root_args,
    is_ninetoothed_repo,
    resolve_examples_root,
    resolve_logs_dir,
    resolve_repo_root,
    resolve_skill_root,
)
from _paths import (
    skill_root as detect_skill_root,
)


def run_cmd(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def module_status(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, "not installed"
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return False, f"import error: {exc}"
    version = getattr(mod, "__version__", "unknown")
    return True, str(version)


def check_torch_cuda() -> list[str]:
    lines: list[str] = []
    ok, version = module_status("torch")
    lines.append(f"- torch: {'OK' if ok else 'MISSING'} ({version})")
    if not ok:
        return lines
    import torch

    cuda_ok = torch.cuda.is_available()
    lines.append(f"- torch.cuda.is_available: {cuda_ok}")
    if cuda_ok:
        lines.append(f"- torch.cuda.device_count: {torch.cuda.device_count()}")
        lines.append(f"- torch.cuda.device_name(0): {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        lines.append(f"- torch.cuda.get_device_capability(0): {cap}")
    return lines


def check_paths(repo: Path, skill: Path, examples: Path | None) -> list[str]:
    lines: list[str] = []
    checks: list[tuple[str, Path]] = [
        ("repo src/ninetoothed", repo / "src" / "ninetoothed"),
        ("repo tests/", repo / "tests"),
        ("skill SKILL.md", skill / "SKILL.md"),
        ("skill scripts/", skill / "scripts"),
    ]
    if examples is not None:
        checks.append(("examples root", examples))
    for label, path in checks:
        status = "OK" if path.exists() else "MISSING"
        try:
            rel = path.relative_to(repo) if path.is_relative_to(repo) else path
        except (ValueError, AttributeError):
            rel = path
        lines.append(f"- [{status}] {label}: `{rel}`")
    if not is_ninetoothed_repo(repo):
        lines.append(
            "- [MISSING] repo does not look like NineToothed (need src/ninetoothed/)"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Environment check for NineToothed repo + ninetoothed-op-dev-skill."
    )
    add_repo_root_args(parser)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path (default: <repo-root>/logs/env_check.md)",
    )
    args = parser.parse_args()

    try:
        repo = resolve_repo_root(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        skill = resolve_skill_root(args) if args.skill_root else detect_skill_root()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    examples = resolve_examples_root(args)
    log_dir = resolve_logs_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (log_dir / "env_check.md")

    code, nvidia_out, nvidia_err = run_cmd(["nvidia-smi"])
    nvcc_code, nvcc_out, nvcc_err = run_cmd(["nvcc", "--version"])
    pytest_code, pytest_out, pytest_err = run_cmd(
        [sys.executable, "-m", "pytest", "--version"]
    )

    torch_ok, torch_version = module_status("torch")
    ninetoothed_ok, ninetoothed_version = module_status("ninetoothed")

    lines = [
        "# Environment Check Report",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- NineToothed repo root: `{repo}`",
        f"- Skill root: `{skill}`",
        f"- Examples root: `{examples}`"
        if examples
        else "- Examples root: (not found; optional)",
        "",
        "## Python",
        "",
        f"- executable: `{sys.executable}`",
        f"- version: `{sys.version.split()[0]}`",
        f"- platform: `{platform.platform()}`",
        "",
        "## GPU / CUDA",
        "",
        f"- nvidia-smi exit code: `{code}`",
    ]
    if nvidia_out:
        lines.append("```text")
        lines.append(nvidia_out)
        lines.append("```")
    if nvidia_err:
        lines.append(f"- nvidia-smi stderr: `{nvidia_err}`")

    lines.extend(["", f"- nvcc exit code: `{nvcc_code}`"])
    if nvcc_out:
        lines.append("```text")
        lines.append(nvcc_out)
        lines.append("```")
    elif nvcc_err:
        lines.append(f"- nvcc: `{nvcc_err}`")

    lines.extend(["", "## Python packages", ""])
    lines.extend(check_torch_cuda())
    lines.append(
        f"- ninetoothed: {'OK' if ninetoothed_ok else 'MISSING'} ({ninetoothed_version})"
    )

    if pytest_code == 0:
        lines.append(f"- pytest: OK ({pytest_out})")
    else:
        lines.append(f"- pytest: MISSING or error ({pytest_err or pytest_out})")

    lines.extend(["", "## Repository / skill paths", ""])
    lines.extend(check_paths(repo, skill, examples))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Pass `--repo-root .` when running inside a NineToothed fork/clone.",
            "- This script does not install packages; it only reports status.",
            "- Skill package must not require `third_party/` outside the target repo.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")

    # Soft exit: missing optional GPU tools is OK; missing repo/skill is not.
    if not is_ninetoothed_repo(repo):
        print(
            "FAIL: --repo-root is not a NineToothed repository (need src/ninetoothed/).",
            file=sys.stderr,
        )
        return 1
    if not (skill / "SKILL.md").is_file():
        print("FAIL: skill SKILL.md missing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
