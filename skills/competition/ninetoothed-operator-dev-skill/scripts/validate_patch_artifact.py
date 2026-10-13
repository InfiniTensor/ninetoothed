#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'])[a-z]:[\\/]")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)PRIVATE KEY-----")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*[^\s]+"  # credential keyword scan
)

FORBIDDEN_PATH_PARTS = {
    ".git",
    ".venv",
    "artifacts",
    "logs",
    "__pycache__",
    "node_modules",
}

SENSITIVE_PATH_WORDS = {
    "credential",
    "credentials",
    "secret",  # credential keyword scan
    "secrets",  # credential keyword scan
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(
    command: list[str], cwd: Path, display_command: str | None = None
) -> dict[str, object]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": display_command or " ".join(command),
        "cwd": str(cwd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def extract_changed_files(text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for raw in parts[2:4]:
                    if raw.startswith(("a/", "b/")):
                        path = raw[2:]
                        if path != "/dev/null" and path not in seen:
                            seen.add(path)
                            files.append(path)
        elif line.startswith(("+++ ", "--- ")):
            raw = line[4:].strip()
            if raw.startswith(("a/", "b/")):
                path = raw[2:]
                if path != "/dev/null" and path not in seen:
                    seen.add(path)
                    files.append(path)
    return files


def path_risks(paths: list[str]) -> list[str]:
    risks: list[str] = []
    for path in paths:
        parts = {part for part in Path(path).parts}
        lower = path.lower()
        hit_parts = sorted(parts & FORBIDDEN_PATH_PARTS)
        if hit_parts:
            risks.append(f"forbidden path part in {path}: {', '.join(hit_parts)}")
        for word in SENSITIVE_PATH_WORDS:
            if word in lower:
                risks.append(f"sensitive path word in {path}: {word}")
                break
        if path.startswith(("/", "\\")):
            risks.append(f"absolute-like patch path: {path}")
    return risks


def build_report(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    repo = Path(args.repo)
    patch = Path(args.patch)
    risks: list[str] = []
    failures: list[str] = []
    warnings: list[str] = []

    report: dict[str, object] = {
        "timestamp": timestamp(),
        "repo": str(repo),
        "patch": str(patch),
        "clean_check_dir": str(args.clean_check_dir) if args.clean_check_dir else None,
        "policy": "do not modify repo; do not auto-apply patch; do not hide failures",
    }

    if not repo.exists():
        failures.append("repo does not exist")
    if not patch.exists():
        failures.append("patch does not exist")

    text = ""
    if patch.exists():
        raw_bytes = patch.read_bytes()
        report["line_endings"] = {
            "crlf": raw_bytes.count(b"\r\n"),
            "lf": raw_bytes.count(b"\n"),
            "cr_only": raw_bytes.count(b"\r") - raw_bytes.count(b"\r\n"),
        }
        if raw_bytes.count(b"\r\n"):
            warnings.append(
                "patch contains CRLF line endings; verify evaluator-platform apply-check"
            )
        text = raw_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            failures.append("patch is empty")

    if text:
        if WINDOWS_ABSOLUTE_RE.search(text):
            failures.append("patch contains absolute Windows path text")
        if PRIVATE_KEY_RE.search(text):
            failures.append("patch contains private key block")  # warning context
        if SECRET_ASSIGNMENT_RE.search(text):  # credential keyword scan
            failures.append("patch contains obvious credential assignment")

    changed_files = extract_changed_files(text)
    report["changed_files"] = changed_files
    if text and not changed_files:
        failures.append("changed files list could not be extracted")

    risks.extend(path_risks(changed_files))
    report["risks"] = risks

    apply_check: dict[str, object] = {
        "status": "BLOCKED",
        "reason": "clean-check-dir not provided",
    }
    if args.clean_check_dir:
        clean_dir = Path(args.clean_check_dir)
        if not clean_dir.exists():
            failures.append("clean-check-dir does not exist")
            apply_check = {
                "status": "BLOCKED",
                "reason": "clean-check-dir does not exist",
            }
        elif not (clean_dir / ".git").exists():
            failures.append("clean-check-dir is not a git repo")
            apply_check = {
                "status": "BLOCKED",
                "reason": "clean-check-dir is not a git repo",
            }
        elif patch.exists():
            apply_check = run_command(
                ["git", "apply", "--check", str(patch.resolve())],
                clean_dir,
                "git apply --check <patch>",
            )
            apply_check["status"] = "PASS" if apply_check["returncode"] == 0 else "FAIL"
            if apply_check["returncode"] != 0:
                failures.append("git apply --check failed")
    else:
        warnings.append("apply-check blocked because clean-check-dir was not provided")

    report["apply_check"] = apply_check
    report["warnings"] = warnings
    report["failures"] = failures

    if failures:
        status = "FAIL"
    elif risks or warnings:
        status = "WARN"
    else:
        status = "PASS"
    report["status"] = status

    lines = [
        "# Patch Artifact Validation",
        "",
        f"- status: {status}",
        f"- repo: `{repo}`",
        f"- patch: `{patch}`",
        f"- clean-check-dir: `{args.clean_check_dir or 'not provided'}`",
        "",
        "## Changed Files",
        "",
    ]
    if changed_files:
        lines.extend(f"- `{path}`" for path in changed_files)
    else:
        lines.append("- [BLOCKED] No changed files extracted.")
    lines.extend(["", "## Risks", ""])
    lines.extend(f"- {item}" for item in risks) if risks else lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append(
        "- None."
    )
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {item}" for item in failures) if failures else lines.append(
        "- None."
    )
    lines.extend(["", "## Apply Check", ""])
    lines.append(f"- status: {apply_check.get('status')}")
    if "command" in apply_check:
        lines.append(f"- command: `{apply_check['command']}`")
        lines.append(f"- returncode: `{apply_check['returncode']}`")
        if apply_check.get("stdout"):
            lines.extend(
                [
                    "",
                    "### stdout",
                    "",
                    "```text",
                    str(apply_check["stdout"]).rstrip(),
                    "```",
                ]
            )
        if apply_check.get("stderr"):
            lines.extend(
                [
                    "",
                    "### stderr",
                    "",
                    "```text",
                    str(apply_check["stderr"]).rstrip(),
                    "```",
                ]
            )
    else:
        lines.append(f"- reason: {apply_check.get('reason')}")

    return report, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a patch artifact without applying it."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--clean-check-dir")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    report, markdown = build_report(args)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{report['status']} validate_patch_artifact.py")
    print(f"markdown={output_md}")
    print(f"json={output_json}")
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
