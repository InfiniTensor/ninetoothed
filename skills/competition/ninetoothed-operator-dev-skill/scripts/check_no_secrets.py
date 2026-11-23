#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".csv", ".patch", ".yaml", ".yml"}
FORBIDDEN_FILENAMES = {".env", "server.txt", "id_rsa", "id_ed25519"}
PATTERNS = {
    "private key": re.compile(
        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)PRIVATE KEY-----"
    ),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a skill package for credential material."
    )
    parser.add_argument("root", nargs="?", type=Path)
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    failures: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in FORBIDDEN_FILENAMES:
            failures.append(f"forbidden credential filename: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} pattern: {rel}")

    if failures:
        print("FAIL check_no_secrets.py")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS check_no_secrets.py")
    print(f"root={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
