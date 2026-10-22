#!/usr/bin/env python3
"""Run required real-GPU differential validation and save an auditable report."""

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def checkpoint(report, stream, started):
    passed = [case for case in report["cases"] if case["status"] == "PASS"]
    report["passed_cases"] = len(passed)
    report["passed_programs"] = sorted({case["program"] for case in passed})
    report["passed_categories"] = sorted({case["category"] for case in passed})
    report["elapsed_validation_seconds"] = round(time.perf_counter() - started, 3)
    stream.seek(0)
    stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    stream.truncate()
    stream.flush()


def validate(device, stream):
    report = {
        "status": "UNVERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "triton",
        "python": platform.python_version(),
        "cases": [],
        "rtol": 1e-3,
        "atol": 1e-3,
        "integer_and_bool_comparison": "exact",
        "limitations": [
            "Correctness validation, not a performance benchmark.",
            "Results apply only to the GPU and cases identified in this report.",
            "Dot covers one scalar float32 2D matmul with M/N output tiles and a complete K domain; "
            "it does not establish arbitrary-layout, split-K, Tensor Core, or performance coverage.",
        ],
    }
    started = time.perf_counter()
    exit_code = 2
    checkpoint(report, stream, started)

    try:
        import numpy
        import sympy

        from tests.test_interpreter_gpu import (
            GPU_CASES,
            SEED,
            require_gpu,
            run_gpu_case,
        )

        report["total_cases"] = len(GPU_CASES)
        torch, triton = require_gpu(device)
        report.update(
            status="RUNNING",
            numpy_version=numpy.__version__,
            sympy_version=sympy.__version__,
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            torch_cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(device),
            compute_capability=list(torch.cuda.get_device_capability(device)),
            device_index=device,
            seed=SEED,
        )

        for case in GPU_CASES:
            report["active_case"] = case.name
            checkpoint(report, stream, started)

            try:
                result = run_gpu_case(case, torch, device)
            except Exception as error:
                result = {
                    "name": case.name,
                    "category": case.category,
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }

            report["cases"].append(result)
            print(f"{result['status']}: {case.name}", flush=True)

        report.pop("active_case", None)
        checkpoint(report, stream, started)
        complete = (
            report["passed_cases"] == len(GPU_CASES)
            and len(report["passed_programs"]) >= 3
        )
        report["status"] = "PASS" if complete else "FAIL"
        exit_code = 0 if complete else 1
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "KeyboardInterrupt: validation did not complete"
        exit_code = 130
        print(report["error"], file=sys.stderr)
    except Exception as error:
        report["status"] = "UNVERIFIED"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(f"UNVERIFIED: {error}", file=sys.stderr)

    checkpoint(report, stream, started)

    return exit_code, report["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "interpreter_gpu_validation.json",
        help="New output path; existing files are never overwritten.",
    )
    args = parser.parse_args()

    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also rejects dangling symlinks and concurrent writers.
        stream = args.report.open("x", encoding="utf-8")
    except OSError as error:
        parser.error(
            f"cannot create report {args.report}: {error}; choose a new output path"
        )

    with stream:
        exit_code, status = validate(args.device, stream)

    print(f"{status}: report saved to {args.report}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
