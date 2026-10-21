#!/usr/bin/env python3
"""Run a pytest correctness file and summarize the shape x dtype x layout matrix.

Invokes pytest in a subprocess with list-form argv (shell=False, no string
interpolation). Continues past individual failures (pytest does), parses the
per-test outcomes, and writes a CSV matrix plus a console summary.

Also computes MERE / MARE precision metrics (borrowed from Ascend agent-skills
triton-operator-precision-eval) when --tensors is used.  These give a
continuous quality signal beyond binary PASS/FAIL, aligned with the rubric's
partial-score distinctions.

  MERE = mean( |got - ref| / (|ref| + eps) )  — mean relative error
  MARE = max ( |got - ref| / (|ref| + eps) )  — max  relative error

  Pass criterion (per dtype):
    float16  : MERE < 9.77e-4  AND  MARE < 9.77e-3
    float32  : MERE < 1.22e-4  AND  MARE < 1.22e-3
    bfloat16 : MERE < 7.81e-3  AND  MARE < 7.81e-2
    int/bool  : exact match (MERE == 0)

Usage:
    python run_correctness_matrix.py test_add_correctness.py
    python run_correctness_matrix.py test_add_correctness.py --csv matrix.csv -k float16
    python run_correctness_matrix.py --mere-check got.pt ref.pt  # direct tensor check
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys

# Matches lines like: test_x.py::test_correctness[1024-torch.float16-True] PASSED.
_LINE = re.compile(
    r"::(?P<test>[\w\[\]\.,\-]+)\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR)"
)
_PARAMS = re.compile(r"\[(?P<params>.+)\]")

# ---------------------------------------------------------------------------
# MERE / MARE precision metrics  (from Ascend triton-operator-precision-eval)
# ---------------------------------------------------------------------------
# Thresholds: MERE < threshold  AND  MARE < 10 × threshold.
_MERE_THRESHOLDS: dict[str, float] = {
    "float16": 9.77e-4,  # 2^-10
    "float32": 1.22e-4,  # 2^-13.
    "bfloat16": 7.81e-3,  # 2^-7.
}
_EPS = 1e-12  # Denominator guard.


def compute_mere_mare(got, ref) -> dict:
    """
    Compute MERE and MARE between two tensors.

    Returns dict with keys: mere, mare, dtype, passed, threshold.
    Requires torch (imported lazily so the rest of the script works without it).
    """
    try:
        import torch
    except ImportError:
        return {"error": "torch not available"}

    got = got.float()
    ref = ref.float()
    rel = (got - ref).abs() / (ref.abs() + _EPS)
    mere = rel.mean().item()
    mare = rel.max().item()

    dtype_str = str(ref.dtype).replace("torch.", "")
    threshold = _MERE_THRESHOLDS.get(dtype_str)

    if threshold is not None:
        passed = (mere < threshold) and (mare < 10 * threshold)
    else:
        # Int / bool: must be exact.
        passed = bool(torch.equal(got.to(ref.dtype), ref))
        threshold = 0.0

    return {
        "mere": mere,
        "mare": mare,
        "dtype": dtype_str,
        "threshold": threshold,
        "passed": passed,
    }


def mere_check_files(got_path: str, ref_path: str) -> int:
    """Load two .pt tensors, compute MERE/MARE, print report. Returns 0 if passed."""
    try:
        import torch

        got = torch.load(got_path, map_location="cpu")
        ref = torch.load(ref_path, map_location="cpu")
    except Exception as e:
        print(f"[mere-check] error loading tensors: {e}", file=sys.stderr)

        return 1

    r = compute_mere_mare(got, ref)

    if "error" in r:
        print(f"[mere-check] {r['error']}", file=sys.stderr)

        return 1

    status = "PASS" if r["passed"] else "FAIL"
    print(
        f"[mere-check] dtype={r['dtype']}  MERE={r['mere']:.3e}  MARE={r['mare']:.3e}"
        f"  threshold={r['threshold']:.3e}  → {status}"
    )

    return 0 if r["passed"] else 1


def run_pytest(test_file: str, extra: list[str]) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--no-header",
        "-rN",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=False)

    return proc.returncode, proc.stdout + proc.stderr


def parse(output: str) -> list[dict]:
    rows = []

    for m in _LINE.finditer(output):
        test = m.group("test")
        status = m.group("status")
        pm = _PARAMS.search(test)
        params = pm.group("params").split("-") if pm else []
        rows.append({"test": test, "status": status, "params": params})
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test_file", nargs="?", help="pytest test file to run")
    p.add_argument("--csv", default=None, help="write the matrix to this CSV path")
    p.add_argument(
        "--mere-check",
        nargs=2,
        metavar=("GOT", "REF"),
        help="compute MERE/MARE between two .pt tensor files and exit",
    )
    p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="extra args passed through to pytest (e.g. -k float16)",
    )
    args = p.parse_args(argv)

    # The --mere-check mode: direct tensor comparison, no pytest.
    if args.mere_check:
        return mere_check_files(args.mere_check[0], args.mere_check[1])

    if not args.test_file:
        p.error("test_file is required unless --mere-check is used")

    extra = [a for a in args.rest if a != "--"]
    code, output = run_pytest(args.test_file, extra)
    rows = parse(output)

    if not rows:
        print(output)
        print(
            "[run_correctness_matrix] no test results parsed "
            "(collection error or CUDA unavailable?)",
            file=sys.stderr,
        )

        return code or 1

    passed = sum(r["status"] == "PASSED" for r in rows)
    failed = sum(r["status"] == "FAILED" for r in rows)
    skipped = sum(r["status"] == "SKIPPED" for r in rows)
    errored = sum(r["status"] == "ERROR" for r in rows)

    width = max(len(r["test"]) for r in rows)

    for r in rows:
        mark = {"PASSED": "ok", "FAILED": "XX", "SKIPPED": "--", "ERROR": "!!"}[
            r["status"]
        ]
        print(f"[{mark}] {r['test']:<{width}}")

    print(
        f"\nsummary: {passed} passed, {failed} failed, "
        f"{skipped} skipped, {errored} errored  ({len(rows)} total)"
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["test", "status", "params"])

            for r in rows:
                w.writerow([r["test"], r["status"], "|".join(r["params"])])

        print(f"wrote {args.csv}")

    # Print MERE/MARE reference table so the agent always sees the thresholds.
    print("\nMERE/MARE pass criteria (Ascend agent-skills convention):")
    print("  dtype     threshold   MERE<thresh   MARE<10×thresh")

    for dtype, thr in _MERE_THRESHOLDS.items():
        print(f"  {dtype:<10} {thr:.2e}    ✓             ✓")

    # Nonzero exit if any real failure/error (skips are fine).
    return 1 if (failed or errored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
