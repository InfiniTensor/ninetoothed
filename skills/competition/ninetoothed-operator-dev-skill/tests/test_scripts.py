from __future__ import annotations

import subprocess
import sys
import unittest
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptTest(unittest.TestCase):
    def test_microbenchmark_times_baseline_and_candidate_separately(self) -> None:
        path = ROOT / "scripts" / "run_ntops_microbenchmark.py"
        spec = spec_from_file_location("audit_microbenchmark", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        calls = []

        class Torch:
            @staticmethod
            def add(left, right):
                calls.append("baseline")
                return left + right

        class CandidateTorch:
            @staticmethod
            def add(left, right):
                calls.append("candidate")
                return left + right

        ntops = type("Ntops", (), {"torch": CandidateTorch})()
        args = Namespace(operator="add")

        self.assertEqual(3, module._call_baseline(Torch, args, (1, 2)))
        self.assertEqual(["baseline"], calls)
        calls.clear()
        self.assertEqual(3, module._call_candidate(ntops, args, (1, 2)))
        self.assertEqual(["candidate"], calls)

    def test_scripts_compile(self) -> None:
        scripts = sorted((ROOT / "scripts").glob("*.py"))
        self.assertGreaterEqual(len(scripts), 8)
        for path in scripts:
            with self.subTest(script=path.name):
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")

    def test_cli_help(self) -> None:
        scripts = sorted((ROOT / "scripts").glob("*.py"))
        for path in scripts:
            with self.subTest(script=path.name):
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_validation_clis(self) -> None:
        for name in [
            "validate_skill_package.py",
            "check_no_secrets.py",
            "check_false_verified_claims.py",
            "check_markdown_links.py",
        ]:
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / name), str(ROOT)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
