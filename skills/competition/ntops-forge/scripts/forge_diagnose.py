#!/usr/bin/env python3
"""Match failure text to fix cards for ntops-forge."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX_CARDS = ROOT / "skills" / "ntops-forge" / "fix_cards.md"

RULES: list[tuple[str, str, str]] = [
    (r"@triton\.jit|Triton @triton", "FC-001", "改用 premake/application，禁止 Triton"),
    (
        r"still contains TODO|scaffold placeholder",
        "FC-002",
        "填入 spec formula，preflight --strict",
    ),
    (
        r"SKIPPED: CUDA|CUDA not available",
        "FC-003",
        "换 GPU 机并 source conda activate",
    ),
    (r"can't open file.*scripts", "FC-004", "cd /root/work/skill 后再跑 forge"),
    (
        r"test_conv2d|CompilationError.*triton",
        "FC-005",
        "勿 pytest tests/ 全量，用 spec pytest 文件",
    ),
    (r"differs from reference", "FC-006", "对照 spec formula 与 reference 内核"),
    (r"not registered", "FC-007", "forge run --register 或 register_op.py"),
    (r"must use premake|element_wise", "FC-008", "scaffold --style ntops 重新 CODEGEN"),
    (
        r"unrecognized arguments: --finish",
        "FC-009",
        "同步最新 scripts 到 GPU（v0.5+）或改用 forge",
    ),
    (
        r"ModuleNotFoundError: No module named 'ntops'",
        "FC-010",
        "在 ntops 根目录 pip install -e .",
    ),
    (
        r"No module named pytest",
        "FC-012",
        "pip install pytest；或 cd ntops && pip install -e .",
    ),
]


def diagnose_text(text: str) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for pattern, card, hint in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append((card, hint))
    return hits


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose forge/copilot failures")
    p.add_argument("--text", help="failure message text")
    p.add_argument(
        "--log", type=Path, help="read last failed line from forge_runs.jsonl"
    )
    p.add_argument("--file", type=Path, help="read failure from file")
    args = p.parse_args()

    text = args.text or ""
    if args.file and args.file.is_file():
        text += "\n" + args.file.read_text(encoding="utf-8", errors="ignore")
    if args.log and args.log.is_file():
        lines = [
            ln for ln in args.log.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        for ln in reversed(lines):
            if '"ok": false' in ln or '"stage":' in ln and '"error"' in ln:
                text += "\n" + ln
                break

    if not text.strip():
        print("用法: forge diagnose --text 'FAIL: ...' 或 --log docs/forge_runs.jsonl")
        return 1

    hits = diagnose_text(text)
    if not hits:
        print("WARN: no fix card matched; read skills/ntops-forge/fix_cards.md")
        return 2

    print("=== Fix Cards ===")
    for card, hint in hits:
        print(f"{card}: {hint}")
    if FIX_CARDS.is_file():
        print(f"\n详情: {FIX_CARDS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
