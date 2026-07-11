"""Shared path helpers for skill scripts.

Target layout: a NineToothed repository root that contains ``src/ninetoothed/``.
Legacy competition layout ``third_party/ninetoothed`` remains a fallback.

All path-aware CLIs should call :func:`add_repo_root_args` and
:func:`resolve_repo_root`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_ENV_REPO = "NINETOOTHED_REPO_ROOT"
_ENV_SKILL = "NINETOOTHED_SKILL_ROOT"
_ENV_EXAMPLES = "NINETOOTHED_EXAMPLES_ROOT"


def skill_root(explicit: Path | str | None = None) -> Path:
    """Return the skill package root (directory that contains SKILL.md)."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not (root / "SKILL.md").is_file():
            raise FileNotFoundError(f"SKILL.md not found under --skill-root: {root}")
        return root
    env = os.environ.get(_ENV_SKILL)
    if env:
        return skill_root(Path(env))
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "SKILL.md").is_file() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(
        "Skill root not found (expected a directory containing SKILL.md). "
        "Pass --skill-root or set NINETOOTHED_SKILL_ROOT."
    )


def is_ninetoothed_repo(path: Path) -> bool:
    """Return True if *path* looks like a NineToothed repository root."""
    p = path.resolve()
    if (p / "src" / "ninetoothed").is_dir():
        return True
    # Editable install / nested package without src/ layout
    if (p / "ninetoothed" / "__init__.py").is_file() and (p / "tests").is_dir():
        return True
    return False


def ninetoothed_repo_root(explicit: Path | str | None = None) -> Path:
    """Resolve the target NineToothed repository root.

    Recognition order for auto-detect:
    1. ``--repo-root`` / ``NINETOOTHED_REPO_ROOT``
    2. Walk upward for ``src/ninetoothed/``
    3. Legacy: ``third_party/ninetoothed`` (competition workspace)
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if is_ninetoothed_repo(root):
            return root
        nested = root / "third_party" / "ninetoothed"
        if is_ninetoothed_repo(nested):
            return nested
        raise FileNotFoundError(
            f"Not a NineToothed repo root (missing src/ninetoothed/): {root}"
        )

    env = os.environ.get(_ENV_REPO)
    if env:
        return ninetoothed_repo_root(Path(env))

    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if is_ninetoothed_repo(candidate):
            return candidate
        legacy = candidate / "third_party" / "ninetoothed"
        if is_ninetoothed_repo(legacy):
            return legacy

    # Also search from skill location (skill may live under skills/... inside the fork)
    try:
        sk = skill_root()
    except FileNotFoundError:
        sk = Path(__file__).resolve()
    for candidate in (sk, *sk.parents):
        if is_ninetoothed_repo(candidate):
            return candidate
        legacy = candidate / "third_party" / "ninetoothed"
        if is_ninetoothed_repo(legacy):
            return legacy

    raise FileNotFoundError(
        "NineToothed repository root not found. "
        "Pass --repo-root pointing at a clone that contains src/ninetoothed/, "
        "or set NINETOOTHED_REPO_ROOT."
    )


def examples_repo_root(explicit: Path | str | None = None) -> Path | None:
    """Return optional ninetoothed-examples root (may be absent)."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_dir() else None
    env = os.environ.get(_ENV_EXAMPLES)
    if env:
        return examples_repo_root(Path(env))
    try:
        repo = ninetoothed_repo_root()
    except FileNotFoundError:
        return None
    # Sibling clone
    sibling = repo.parent / "ninetoothed-examples"
    if sibling.is_dir():
        return sibling
    # Competition layout
    for candidate in (repo, *repo.parents):
        legacy = candidate / "third_party" / "ninetoothed-examples"
        if legacy.is_dir():
            return legacy
    return None


def workspace_root(explicit: Path | str | None = None) -> Path:
    """Workspace used for logs/reports.

    Prefer the NineToothed repo root. Falls back to competition workspace
    (directory that contains ``third_party/ninetoothed``) when present.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    try:
        return ninetoothed_repo_root()
    except FileNotFoundError:
        pass
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        if (candidate / "third_party" / "ninetoothed").is_dir():
            return candidate
    # Last resort: skill's parent chain (fork with skills/ under repo)
    try:
        sk = skill_root()
        for candidate in sk.parents:
            if (candidate / "src" / "ninetoothed").is_dir() or (
                candidate / "skills"
            ).is_dir():
                return candidate
        return sk.parents[2] if len(sk.parents) >= 3 else sk.parent
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Workspace root not found. Pass --repo-root to a NineToothed clone."
        ) from exc


def logs_dir(repo: Path | None = None) -> Path:
    """Default log directory: ``<repo-root>/logs``."""
    base = repo if repo is not None else workspace_root()
    return Path(base) / "logs"


def add_skill_root_arg(parser: argparse.ArgumentParser) -> None:
    """Attach ``--skill-root`` for tools that only need the skill package path."""
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=None,
        help=f"Skill package root (directory containing SKILL.md). Env: {_ENV_SKILL}",
    )


def add_repo_root_args(parser: argparse.ArgumentParser) -> None:
    """Attach path flags for scripts that access a NineToothed repository."""
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="NineToothed repository root (directory containing src/ninetoothed/). "
        f"Env: {_ENV_REPO}",
    )
    add_skill_root_arg(parser)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=None,
        help=f"Optional ninetoothed-examples root (--examples-root). Env: {_ENV_EXAMPLES}",
    )


def resolve_repo_root(args: argparse.Namespace) -> Path:
    return ninetoothed_repo_root(getattr(args, "repo_root", None))


def resolve_skill_root(args: argparse.Namespace) -> Path:
    return skill_root(getattr(args, "skill_root", None))


def resolve_examples_root(args: argparse.Namespace) -> Path | None:
    return examples_repo_root(getattr(args, "examples_root", None))


def resolve_logs_dir(args: argparse.Namespace) -> Path:
    repo = resolve_repo_root(args)
    return logs_dir(repo)
