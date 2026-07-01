#!/usr/bin/env python3
"""Check contribution metadata rules from ``CONTRIBUTING.md``."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

BRANCH_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
TRAILING_PUNCTUATION = ".,;:!?"
PAST_TENSE_PATTERN = re.compile(
    r"\b(added|fixed|changed|modified|removed|updated|implemented|created|improved)\b",
    re.IGNORECASE,
)
PYTEST_OUTPUT_PATTERN = re.compile(
    r"`?pytest`?\s+output\s*:\s*```[a-zA-Z0-9_-]*\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Metadata:
    title: str | None = None
    branch: str | None = None
    body: str | None = None


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args)
    diagnostics = check_metadata(
        metadata,
        check_title=not args.skip_title,
        check_branch=not args.skip_branch,
        check_pytest_output=not args.skip_pytest_output,
    )

    for diagnostic in diagnostics:
        print(diagnostic)

    return 1 if diagnostics else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check contribution metadata rules from CONTRIBUTING.md."
    )
    parser.add_argument("--event", type=Path, help="GitHub event JSON file.")
    parser.add_argument("--title", help="Commit message subject or PR title.")
    parser.add_argument("--branch", help="Branch name to validate.")
    parser.add_argument("--body", help="PR body text.")
    parser.add_argument("--body-file", type=Path, help="File containing the PR body.")
    parser.add_argument(
        "--commit-message-file",
        type=Path,
        help="Commit message file whose first line should be checked as a title.",
    )
    parser.add_argument("--skip-title", action="store_true", help="Skip title checks.")
    parser.add_argument(
        "--skip-branch", action="store_true", help="Skip branch checks."
    )
    parser.add_argument(
        "--skip-pytest-output",
        action="store_true",
        help="Skip PR body pytest output checks.",
    )

    return parser.parse_args()


def load_metadata(args: argparse.Namespace) -> Metadata:
    metadata = Metadata()

    if args.event is not None:
        metadata = metadata_from_event(args.event)

    title = args.title if args.title is not None else metadata.title
    branch = args.branch if args.branch is not None else metadata.branch
    body = args.body if args.body is not None else metadata.body

    if args.body_file is not None:
        body = args.body_file.read_text(encoding="utf-8")

    if args.commit_message_file is not None:
        title = first_line(args.commit_message_file.read_text(encoding="utf-8"))

    return Metadata(title=title, branch=branch, body=body)


def metadata_from_event(path: Path) -> Metadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request") or {}
    head = pull_request.get("head") or {}

    return Metadata(
        title=pull_request.get("title"),
        branch=head.get("ref"),
        body=pull_request.get("body") or "",
    )


def first_line(text: str) -> str:
    return text.splitlines()[0] if text.splitlines() else ""


def check_metadata(
    metadata: Metadata,
    *,
    check_title: bool,
    check_branch: bool,
    check_pytest_output: bool,
) -> list[str]:
    diagnostics: list[str] = []

    if check_title:
        diagnostics.extend(check_title_text(metadata.title, label="title"))

    if check_branch:
        diagnostics.extend(check_branch_name(metadata.branch))

    if check_pytest_output:
        diagnostics.extend(check_pytest_output_block(metadata.body))

    return diagnostics


def check_title_text(title: str | None, *, label: str) -> list[str]:
    diagnostics: list[str] = []
    normalized = (title or "").strip()

    if not normalized:
        return [f"METADATA001: The {label} cannot be empty."]

    if not normalized[0].isupper():
        diagnostics.append(
            f"METADATA002: The {label} must start with an uppercase letter."
        )

    if normalized[-1] in TRAILING_PUNCTUATION:
        diagnostics.append(f"METADATA003: The {label} must not end with punctuation.")

    if PAST_TENSE_PATTERN.search(normalized):
        diagnostics.append(f"METADATA004: The {label} must use imperative mood.")

    return diagnostics


def check_branch_name(branch: str | None) -> list[str]:
    normalized = (branch or "").strip()

    if not normalized:
        return ["METADATA005: The branch name cannot be empty."]

    diagnostics: list[str] = []

    if not BRANCH_PATTERN.fullmatch(normalized):
        diagnostics.append(
            "METADATA006: The branch name must use kebab-case with lowercase letters, "
            "numbers, and hyphens."
        )

    if len(normalized) > 50:
        diagnostics.append(
            "METADATA007: The branch name must be 50 characters or shorter."
        )

    return diagnostics


def check_pytest_output_block(body: str | None) -> list[str]:
    normalized = body or ""
    match = PYTEST_OUTPUT_PATTERN.search(normalized)

    if match is None:
        return [
            "METADATA008: The PR description must include a `pytest` output code block."
        ]

    if not match.group(1).strip():
        return ["METADATA009: The `pytest` output code block cannot be empty."]

    return []


if __name__ == "__main__":
    raise SystemExit(main())
