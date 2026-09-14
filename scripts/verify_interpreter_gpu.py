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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "interpreter_gpu_validation.json",
    )
    args = parser.parse_args()
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

    try:
        import numpy
        import sympy

        from tests.test_interpreter_gpu import (
            GPU_CASES,
            SEED,
            require_gpu,
            run_gpu_case,
        )

        torch, triton = require_gpu(args.device)
        report.update(
            numpy_version=numpy.__version__,
            sympy_version=sympy.__version__,
            torch_version=torch.__version__,
            triton_version=triton.__version__,
            torch_cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(args.device),
            compute_capability=list(torch.cuda.get_device_capability(args.device)),
            device_index=args.device,
            seed=SEED,
        )

        for case in GPU_CASES:
            try:
                result = run_gpu_case(case, torch, args.device)
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

        passed = [case for case in report["cases"] if case["status"] == "PASS"]
        report["passed_cases"] = len(passed)
        report["total_cases"] = len(GPU_CASES)
        report["passed_programs"] = sorted({case["program"] for case in passed})
        report["passed_categories"] = sorted({case["category"] for case in passed})
        complete = len(passed) == len(GPU_CASES) and len(report["passed_programs"]) >= 3
        report["status"] = "PASS" if complete else "FAIL"
        exit_code = 0 if complete else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(f"UNVERIFIED: {error}", file=sys.stderr)

    report["elapsed_validation_seconds"] = round(time.perf_counter() - started, 3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{report['status']}: report saved to {args.report}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
