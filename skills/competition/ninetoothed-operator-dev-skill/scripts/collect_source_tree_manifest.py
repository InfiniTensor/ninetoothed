#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "artifacts",
    "logs",
    ".venv",
    ".venv-sys",
    "venv",
    "node_modules",
    ".pytest_cache",
}

SOURCE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".lua",
    ".txt",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path, max_files: int | None) -> dict[str, object]:
    root = root.resolve()
    items: list[dict[str, object]] = []
    warnings: list[str] = []
    if not (root / ".git").exists():
        warnings.append(
            "WARNING: .git metadata missing; manifest is weaker than commit evidence"
        )

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if should_skip(rel):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        stat = path.stat()
        items.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
        if max_files is not None and len(items) >= max_files:
            warnings.append(f"WARNING: max-files limit reached at {max_files}")
            break

    return {
        "timestamp": timestamp(),
        "root": str(root),
        "git_metadata": (root / ".git").exists(),
        "file_count": len(items),
        "warnings": warnings,
        "items": items,
    }


def write_outputs(
    report: dict[str, object], output_json: Path, output_md: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Source Tree Manifest",
        "",
        f"- root: `{report['root']}`",
        f"- timestamp: `{report['timestamp']}`",
        f"- git_metadata: `{report['git_metadata']}`",
        f"- file_count: `{report['file_count']}`",
        "",
        "## Warnings",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None.")
    lines.extend(["", "## Files"])
    for item in report.get("items", [])[:200]:
        lines.append(f"- `{item['path']}` size={item['size']} sha256={item['sha256']}")
    if int(report["file_count"]) > 200:
        lines.append("- Output truncated in markdown; see JSON for full manifest.")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a source tree manifest for workspaces without git metadata."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")
    report = collect(root, args.max_files)
    write_outputs(report, Path(args.output_json), Path(args.output_md))
    print("PASS collect_source_tree_manifest.py")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
