#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify MANIFEST.sha256 in a skill package."
    )
    parser.add_argument("root", nargs="?", type=Path)
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        print("FAIL verify_manifest.py: MANIFEST.sha256 is missing")
        return 1

    failures: list[str] = []
    listed: set[str] = set()
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            expected, rel = raw.split("  ", 1)
        except ValueError:
            failures.append(f"invalid manifest line {number}")
            continue
        path = root / rel
        listed.add(rel)
        if not path.is_file():
            failures.append(f"missing file: {rel}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {rel}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    for rel in sorted(actual - listed):
        failures.append(f"unlisted file: {rel}")
    for rel in sorted(listed - actual):
        failures.append(f"listed file is absent: {rel}")

    if failures:
        print("FAIL verify_manifest.py")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS verify_manifest.py")
    print(f"files={len(listed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
