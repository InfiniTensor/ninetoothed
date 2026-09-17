from __future__ import annotations

import csv
import json
import math
import re
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackageStructureTest(unittest.TestCase):
    def test_required_top_level_files(self) -> None:
        required = {
            "SKILL.md",
            "README.md",
            "README.zh-CN.md",
            "HONOR_CODE.md",
            "REFERENCE.md",
            "PR_DESCRIPTION.md",
            "SUBMISSION_CHECKLIST.md",
            "SUBMISSION_COMMANDS.md",
        }
        self.assertEqual(
            [], sorted(name for name in required if not (ROOT / name).is_file())
        )

    def test_required_directories(self) -> None:
        required = {"agents", "references", "scripts", "examples", "tests", "reports"}
        self.assertEqual(
            [], sorted(name for name in required if not (ROOT / name).is_dir())
        )

    def test_report_pdf_exists(self) -> None:
        path = ROOT / "reports" / "123123_九齿skill创新挑战_T3-1-1_赛题报告.pdf"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 100_000)
        self.assertTrue(path.read_bytes().startswith(b"%PDF-"))

    def test_frontmatter(self) -> None:
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        end = text.find("\n---\n", 4)
        self.assertGreater(end, 4)
        block = text[4:end]
        keys = re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", block, re.MULTILINE)
        self.assertEqual(["name", "description"], keys)
        self.assertIn("name: ninetoothed-operator-dev-skill", block)
        self.assertIn("NineToothed", block)

    def test_skill_length_and_execution_focus(self) -> None:
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text.splitlines()), 150)
        self.assertLessEqual(len(text.splitlines()), 230)
        for phrase in [
            "Implement The Minimal Correct Change",
            "Prove Correctness",
            "Benchmark Only After Correctness",
            "Produce An Applicable Patch",
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_trigger_cases(self) -> None:
        path = ROOT / "tests" / "fixtures" / "trigger_cases.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["positive"]), 10)
        self.assertGreaterEqual(len(data["negative"]), 10)
        self.assertTrue(
            all("NineToothed" in item or "ntops" in item for item in data["positive"])
        )
        self.assertTrue(
            all(
                "NineToothed" not in item and "ntops" not in item
                for item in data["negative"]
            )
        )

    def test_selftest_readmes(self) -> None:
        required = {
            "SELFTEST-EW-001",
            "SELFTEST-RED-001",
            "SELFTEST-LAYOUT-001",
            "SELFTEST-PERF-AOT-001",
            "SELFTEST-IMPL-001",
        }
        headings = {
            "Task Description",
            "Agent Execution Summary",
            "Production And Test Files Changed",
            "Correctness",
            "Benchmark",
            "No-Skill Versus Skill",
            "Unsupported Or Unverified",
        }
        for name in required:
            text = (ROOT / "examples" / "selftests" / name / "README.md").read_text(
                encoding="utf-8"
            )
            for heading in headings:
                with self.subTest(case=name, heading=heading):
                    self.assertIn(f"## {heading}", text)

    def test_two_real_benchmark_records(self) -> None:
        records = []
        for path in sorted((ROOT / "examples" / "selftests").glob("*/benchmark.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(1, len(rows), path)
            row = rows[0]
            self.assertEqual("PASS", row["correctness_status"])
            self.assertGreater(float(row["baseline_median_ms"]), 0)
            self.assertGreater(float(row["candidate_median_ms"]), 0)
            self.assertGreater(int(row["warmup"]), 0)
            self.assertGreater(int(row["repeat"]), 0)
            repeat = int(row["repeat"])
            baseline_samples = [
                float(value) for value in row["baseline_samples_ms"].split(";")
            ]
            candidate_samples = [
                float(value) for value in row["candidate_samples_ms"].split(";")
            ]
            self.assertEqual(repeat, len(baseline_samples))
            self.assertEqual(repeat, len(candidate_samples))
            self.assertTrue(
                math.isclose(
                    float(row["baseline_mean_ms"]),
                    statistics.mean(baseline_samples),
                    rel_tol=1e-5,
                    abs_tol=2e-6,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["baseline_median_ms"]),
                    statistics.median(baseline_samples),
                    rel_tol=1e-5,
                    abs_tol=2e-6,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["candidate_mean_ms"]),
                    statistics.mean(candidate_samples),
                    rel_tol=1e-5,
                    abs_tol=2e-6,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["candidate_median_ms"]),
                    statistics.median(candidate_samples),
                    rel_tol=1e-5,
                    abs_tol=2e-6,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["speedup_median"]),
                    float(row["baseline_median_ms"])
                    / float(row["candidate_median_ms"]),
                    rel_tol=1e-5,
                    abs_tol=2e-6,
                )
            )
            records.append(row)
        self.assertGreaterEqual(len(records), 2)

    def test_real_implementation_patch(self) -> None:
        path = (
            ROOT
            / "examples"
            / "selftests"
            / "SELFTEST-IMPL-001"
            / "implementation.patch"
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("src/ntops/torch/avg_pool2d.py", text)
        self.assertIn("src/ntops/torch/max_pool2d.py", text)
        self.assertIn("tests/test_avg_pool2d.py", text)
        self.assertIn("tests/test_max_pool2d.py", text)
        self.assertIn("if isinstance(kernel_size, int):", text)

    def test_patch_evidence_passes(self) -> None:
        logs = sorted((ROOT / "examples" / "selftests").glob("*/**/*apply_check.log"))
        self.assertGreaterEqual(len(logs), 5)
        for path in logs:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("returncode=0", text)
                self.assertIn("strict_returncode=0", text)
                self.assertIn("patch_crlf_count=0", text)


if __name__ == "__main__":
    unittest.main()
