#!/usr/bin/env python3
"""Idempotently register an operator in ntops kernels/torch __init__.py."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _insert_sorted_name(block: str, name: str) -> str:
    """Insert `name` into a comma-separated import/__all__ block, sorted."""
    if re.search(rf"(^|[\s,\"]){re.escape(name)}([\s,\"]|$)", block):
        return block
    items = [m.group(1) for m in re.finditer(r'"([a-zA-Z_][\w]*)"', block)]
    if not items:
        items = [
            m.group(1) for m in re.finditer(r",\s*([a-zA-Z_][\w]*)\s*,", f",{block},")
        ]
    items = sorted(set(items + [name]))
    return ",\n    ".join(f'"{x}"' for x in items)


def register_kernels_init(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if re.search(rf"^\s*{re.escape(name)}\s*,?\s*$", text, re.MULTILINE):
        return False

    import_match = re.search(
        r"from ntops\.kernels import \(\s*(.*?)\s*\)",
        text,
        re.DOTALL,
    )
    if not import_match:
        raise ValueError(f"cannot parse kernels import block: {path}")

    names = [
        n.strip().rstrip(",") for n in import_match.group(1).splitlines() if n.strip()
    ]
    if name in names:
        return False
    names.append(name)
    names.sort()
    new_import = "from ntops.kernels import (\n    " + ",\n    ".join(names) + ",\n)"
    text = text[: import_match.start()] + new_import + text[import_match.end() :]

    all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if all_match:
        new_all = (
            "__all__ = [\n    " + _insert_sorted_name(all_match.group(1), name) + ",\n]"
        )
        text = text[: all_match.start()] + new_all + text[all_match.end() :]

    path.write_text(text, encoding="utf-8")
    return True


def register_torch_init(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8")
    line = f"from ntops.torch.{name} import {name}"
    if line in text:
        return False

    lines = text.splitlines()
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.startswith("from ntops.torch."):
            insert_at = i + 1
            if ln > line:
                insert_at = i
                break
    lines.insert(insert_at, line)
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"

    all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if all_match:
        new_all = (
            "__all__ = [\n    " + _insert_sorted_name(all_match.group(1), name) + ",\n]"
        )
        text = text[: all_match.start()] + new_all + text[all_match.end() :]

    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="Register operator in ntops __init__.py files"
    )
    p.add_argument("--name", required=True, help="operator name, e.g. silu")
    p.add_argument("--ntops-root", type=Path, required=True, help="ntops repo root")
    p.add_argument("--dry-run", action="store_true", help="print actions only")
    args = p.parse_args()

    kernels_init = args.ntops_root / "src" / "ntops" / "kernels" / "__init__.py"
    torch_init = args.ntops_root / "src" / "ntops" / "torch" / "__init__.py"
    for path in (kernels_init, torch_init):
        if not path.is_file():
            print(f"FAIL: missing {path}", file=sys.stderr)
            return 1

    if args.dry_run:
        print(f"would register '{args.name}' in:")
        print(f"  {kernels_init}")
        print(f"  {torch_init}")
        return 0

    k_changed = register_kernels_init(kernels_init, args.name)
    t_changed = register_torch_init(torch_init, args.name)
    if k_changed or t_changed:
        print(f"OK: registered '{args.name}'")
    else:
        print(f"OK: '{args.name}' already registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
