#!/usr/bin/env python3
"""Export skill package for PR to InfiniTensor/ninetoothed (skills/competition/)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ninetoothed-pr-export"

COPY_SKILL = [
    ("skills/ntops-forge", "skills/competition/ntops-forge"),
    ("skills/ntops-copilot", "skills/competition/ntops-copilot"),
]
COPY_SCRIPTS = "skills/competition/ntops-forge/scripts"
COPY_DOCS = [
    "docs/GPU_Test_Report.md",
    "docs/AB_Report.md",
    "docs/SelfTestPlan.md",
    "docs/ScoringAlignment.md",
    "docs/selftests",
    "docs/st2_st3_gpu_results.md",
    "docs/bench_silu.json",
]
ROOT_FILES = ["HONOR_CODE.md", "REFERENCE.md", "PR_DESCRIPTION.md"]


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare ninetoothed upstream PR export")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out: Path = args.output

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for src_rel, dst_rel in COPY_SKILL:
        copy_tree(ROOT / src_rel, out / dst_rel)

    scripts_dst = out / COPY_SCRIPTS
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(f, scripts_dst / f.name)
    for f in (ROOT / "scripts").glob("*.sh"):
        shutil.copy2(f, scripts_dst / f.name)

    docs_dst = out / "skills/competition/ntops-forge/docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    for rel in COPY_DOCS:
        src = ROOT / rel
        dst = docs_dst / Path(rel).name
        if src.is_dir():
            copy_tree(src, dst)
        elif src.is_file():
            shutil.copy2(src, dst)

    for name in ROOT_FILES:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, out / name)

    readme = out / "skills/competition/README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        """# 2026 Spring · NineToothed .skill Competition Submissions

## ntops-forge（于鸿伟 / hongwei-2026 · T3-1-1）

- 主 skill：`ntops-forge/`
- 辅 skill：`ntops-copilot/`
- 独立仓库：https://github.com/hongwei-2026/qiyuanbisai
- Commit：https://github.com/hongwei-2026/qiyuanbisai/commit/e7b32bb

```bash
# 在 ntops 环境验收（脚本在 ntops-forge/scripts/）
python skills/competition/ntops-forge/scripts/forge.py gate --ntops-root /path/to/ntops
```
""",
        encoding="utf-8",
    )

    print(f"OK: export -> {out}")
    print("Next: see docs/UpstreamPRGuide.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
