from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".txt",
    ".json",
    ".csv",
    ".patch",
    ".yaml",
    ".yml",
    ".log",
}


class ContentSafetyTest(unittest.TestCase):
    def iter_text(self):
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path, path.read_text(encoding="utf-8", errors="strict")

    def test_no_generated_cache_or_repository_metadata(self) -> None:
        forbidden_names = {
            "__pycache__",
            ".pytest_cache",
            ".git",
            ".venv",
            "node_modules",
        }
        hits = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.name in forbidden_names or path.suffix in {".pyc", ".pyo"}
        ]
        self.assertEqual([], hits)

    def test_no_private_absolute_paths(self) -> None:
        windows = re.compile(r"(?i)(?:^|[\s\"'])[a-z]:[\\/]")
        unix_private = re.compile(
            r"(?:/" + "home" + r"/[^/\s]+|/" + "root" + r"/|~" + r"/)"
        )
        hits = []
        for path, text in self.iter_text():
            if path.name == "test_content.py":
                continue
            if windows.search(text) or unix_private.search(text):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], hits)

    def test_no_external_workspace_dependencies(self) -> None:
        needles = [
            "work" + "logs/",
            "sub" + "missions/",
            "repos/" + "ntops",
            "repos/" + "InfiniCore",
        ]
        hits = []
        for path, text in self.iter_text():
            if path.suffix in {".patch", ".py"}:
                continue
            if any(needle in text for needle in needles):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], hits)

    def test_no_internal_development_state_language(self) -> None:
        banned = [
            "Dra" + "ft",
            "Not " + "final",
            "place" + "holder",
            "<" + "name" + ">",
            "<" + "github-id" + ">",
            "<" + "team" + ">",
        ]
        numbered_stage = re.compile(r"\b" + "Pha" + r"se\s+[0-9]", re.IGNORECASE)
        hits = []
        for path, text in self.iter_text():
            if path.name == "test_content.py":
                continue
            if any(
                term.lower() in text.lower() for term in banned
            ) or numbered_stage.search(text):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], hits)

    def test_no_false_runtime_claims(self) -> None:
        unsafe = [
            "AOT " + "verified",
            "generated source " + "verified",
            "InfiniCore dispatch " + "verified",
            "full non-contiguous support is " + "verified",
            "first prize " + "guaranteed",
        ]
        hits = []
        for path, text in self.iter_text():
            if path.suffix not in {".md", ".txt", ".log"}:
                continue
            lower = text.lower()
            for phrase in unsafe:
                if phrase.lower() in lower:
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{phrase}")
        self.assertEqual([], hits)

    def test_text_files_are_lf(self) -> None:
        hits = []
        for path in ROOT.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and b"\r\n" in path.read_bytes()
            ):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual([], hits)


if __name__ == "__main__":
    unittest.main()
