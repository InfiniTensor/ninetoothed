#!/usr/bin/env python3
"""Measure masks, SASS instructions, registers, and cubin payload size."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_count(path):
    return len(re.findall(r"\bmask\s*=", path.read_text(encoding="utf-8")))


def run_tool(command):
    return subprocess.run(
        command,
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def cubin_metrics(path, nvdisasm, cuobjdump):
    sass = run_tool([nvdisasm, "--print-code", str(path)])
    resource_usage = run_tool([cuobjdump, "--dump-resource-usage", str(path)])
    instructions = sum(
        re.match(r"^\s*/\*[0-9a-fA-F]+\*/", line) is not None
        for line in sass.splitlines()
    )
    register_match = re.search(r"\bREG:([0-9]+)\b", resource_usage)

    if register_match is None:
        raise RuntimeError(f"Cannot find register usage in {path}.")

    return {
        "sass_instruction_count": instructions,
        "register_count": int(register_match.group(1)),
        "cubin_payload_bytes": path.stat().st_size,
        "cubin_sha256": sha256(path),
        "sass": sass,
        "resource_usage": resource_usage,
    }


def reduction(baseline, submitted):
    return (baseline - submitted) / baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        nargs=5,
        metavar=(
            "CASE_ID",
            "BASELINE_SOURCE",
            "SUBMITTED_SOURCE",
            "BASELINE_CUBIN",
            "SUBMITTED_CUBIN",
        ),
        required=True,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path)
    parser.add_argument("--nvdisasm", default=shutil.which("nvdisasm"))
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump"))
    args = parser.parse_args()

    if not args.nvdisasm or not args.cuobjdump:
        raise SystemExit("nvdisasm and cuobjdump must be available or specified")

    rows = []

    for (
        case_id,
        baseline_source,
        submitted_source,
        baseline_cubin,
        submitted_cubin,
    ) in args.case:
        baseline_source = Path(baseline_source)
        submitted_source = Path(submitted_source)
        baseline_cubin = Path(baseline_cubin)
        submitted_cubin = Path(submitted_cubin)

        for path in (
            baseline_source,
            submitted_source,
            baseline_cubin,
            submitted_cubin,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        baseline_metrics = cubin_metrics(baseline_cubin, args.nvdisasm, args.cuobjdump)
        submitted_metrics = cubin_metrics(
            submitted_cubin, args.nvdisasm, args.cuobjdump
        )
        baseline_masks = mask_count(baseline_source)
        submitted_masks = mask_count(submitted_source)

        row = {
            "case_id": case_id,
            "baseline_generated_source": str(baseline_source),
            "submitted_generated_source": str(submitted_source),
            "baseline_generated_source_sha256": sha256(baseline_source),
            "submitted_generated_source_sha256": sha256(submitted_source),
            "baseline_cubin": str(baseline_cubin),
            "submitted_cubin": str(submitted_cubin),
            "baseline_cubin_sha256": baseline_metrics["cubin_sha256"],
            "submitted_cubin_sha256": submitted_metrics["cubin_sha256"],
            "baseline_mask_arg_count": baseline_masks,
            "submitted_mask_arg_count": submitted_masks,
            "mask_arg_reduction": reduction(baseline_masks, submitted_masks),
            "baseline_sass_instruction_count": baseline_metrics[
                "sass_instruction_count"
            ],
            "submitted_sass_instruction_count": submitted_metrics[
                "sass_instruction_count"
            ],
            "sass_instruction_reduction": reduction(
                baseline_metrics["sass_instruction_count"],
                submitted_metrics["sass_instruction_count"],
            ),
            "baseline_register_count": baseline_metrics["register_count"],
            "submitted_register_count": submitted_metrics["register_count"],
            "baseline_cubin_bytes": baseline_metrics["cubin_payload_bytes"],
            "submitted_cubin_bytes": submitted_metrics["cubin_payload_bytes"],
        }
        rows.append(row)

        if args.derived_dir is not None:
            args.derived_dir.mkdir(parents=True, exist_ok=True)

            for variant, metrics in (
                ("baseline", baseline_metrics),
                ("submitted", submitted_metrics),
            ):
                prefix = args.derived_dir / f"{case_id}.{variant}"
                Path(f"{prefix}.sass.txt").write_text(metrics["sass"], encoding="utf-8")
                Path(f"{prefix}.resource.txt").write_text(
                    metrics["resource_usage"], encoding="utf-8"
                )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, indent=2) + "\n"
    args.json_out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
