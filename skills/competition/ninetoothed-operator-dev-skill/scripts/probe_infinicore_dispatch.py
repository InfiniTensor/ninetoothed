#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_outputs(
    report: dict[str, object], output_json: Path, output_md: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# InfiniCore Dispatch Probe",
        "",
        f"- status: `{report['status']}`",
        f"- operator: `{report['operator']}`",
        f"- python: `{report['python']}`",
        f"- dispatch_status: `{report['dispatch_status']}`",
        f"- correctness_status: `{report['correctness_status']}`",
        "",
        "## Imports",
    ]
    imports = report.get("imports", {})
    for name, value in imports.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Runtime State"])
    for key in [
        "cuda_available",
        "device_name",
        "infinicore_use_ntops",
        "has_infinicore_ntops",
        "has_ntops_torch_silu",
    ]:
        lines.append(f"- {key}: `{report.get(key, 'NA')}`")
    lines.extend(["", "## Evidence"])
    for item in report.get("evidence", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers", [])
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None.")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe InfiniCore ntops dispatch without faking evidence."
    )
    parser.add_argument("--operator", required=True, choices=["silu"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    report: dict[str, object] = {
        "timestamp": timestamp(),
        "operator": args.operator,
        "python": sys.executable,
        "status": "BLOCKED",
        "dispatch_status": "BLOCKED",
        "correctness_status": "BLOCKED",
        "imports": {},
        "evidence": [],
        "blockers": [],
    }

    try:
        import torch  # type: ignore

        report["imports"]["torch"] = f"OK {getattr(torch, '__version__', 'no_version')}"
    except Exception as exc:
        report["imports"]["torch"] = f"FAILED {type(exc).__name__}: {exc}"
        report["blockers"].append("BLOCKED: torch import failed")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: torch import failed")
        return 2

    try:
        import infinicore  # type: ignore

        report["imports"]["infinicore"] = (
            f"OK {getattr(infinicore, '__version__', 'no_version')}"
        )
    except Exception as exc:
        report["imports"]["infinicore"] = f"FAILED {type(exc).__name__}: {exc}"
        report["blockers"].append(
            f"BLOCKED: infinicore import failed: {type(exc).__name__}"
        )
        if "infinicore.lib" in str(exc):
            report["blockers"].append(
                "BLOCKED: native library module infinicore.lib is missing"
            )
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: infinicore import failed")
        return 2

    report["infinicore_use_ntops"] = getattr(infinicore, "use_ntops", "missing")
    report["has_infinicore_ntops"] = hasattr(infinicore, "ntops")
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA"
    )

    try:
        import infinicore.nn.functional as functional  # type: ignore

        report["imports"]["infinicore.nn.functional"] = "OK"
    except Exception as exc:
        report["imports"]["infinicore.nn.functional"] = (
            f"FAILED {type(exc).__name__}: {exc}"
        )
        report["blockers"].append("BLOCKED: functional import failed")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: functional import failed")
        return 2

    has_silu = False
    try:
        has_silu = bool(hasattr(infinicore.ntops.torch, "silu"))
    except Exception as exc:
        report["blockers"].append(
            f"BLOCKED: cannot inspect infinicore.ntops.torch.silu: {type(exc).__name__}"
        )
    report["has_ntops_torch_silu"] = has_silu

    if not torch.cuda.is_available():
        report["blockers"].append("BLOCKED: CUDA unavailable")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: CUDA unavailable")
        return 2

    if not getattr(infinicore, "use_ntops", False):
        report["blockers"].append("BLOCKED: infinicore.use_ntops is not true")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: use_ntops not true")
        return 2

    if not has_silu:
        report["blockers"].append("BLOCKED: infinicore.ntops.torch.silu missing")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: silu wrapper missing")
        return 2

    flag = {"called": False}
    try:
        original = infinicore.ntops.torch.silu

        def wrapped(*wrapped_args, **wrapped_kwargs):
            flag["called"] = True
            return original(*wrapped_args, **wrapped_kwargs)

        infinicore.ntops.torch.silu = wrapped
        report["evidence"].append(
            "Monkeypatch installed on infinicore.ntops.torch.silu."
        )
    except Exception as exc:
        report["blockers"].append(f"BLOCKED: monkeypatch failed: {type(exc).__name__}")
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: monkeypatch failed")
        return 2

    try:
        x = torch.randn((8,), device="cuda", dtype=torch.float32)
        y = functional.silu(x)
        ref = torch.nn.functional.silu(x)
        correct = bool(torch.allclose(y, ref, rtol=1e-4, atol=1e-4))
    except Exception as exc:
        report["blockers"].append(
            f"BLOCKED: runtime probe exception: {type(exc).__name__}: {exc}"
        )
        report["dispatch_status"] = "DISPATCH_VERIFIED" if flag["called"] else "BLOCKED"
        write_outputs(report, Path(args.output_json), Path(args.output_md))
        print("BLOCKED probe_infinicore_dispatch.py: runtime probe exception")
        return 2

    if flag["called"]:
        report["dispatch_status"] = "DISPATCH_VERIFIED"
        report["evidence"].append(
            "DISPATCH_VERIFIED: monkeypatch flag was set by functional call."
        )
    else:
        report["dispatch_status"] = "BLOCKED"
        report["blockers"].append(
            "BLOCKED: functional call did not reach ntops.torch.silu"
        )

    if correct:
        report["correctness_status"] = "PASS"
        report["evidence"].append(
            "Output matched torch.nn.functional.silu for the probe tensor."
        )
    else:
        report["correctness_status"] = "FAIL"
        report["blockers"].append("BLOCKED: output did not match torch reference")

    report["status"] = (
        "PASS"
        if report["dispatch_status"] == "DISPATCH_VERIFIED" and correct
        else "BLOCKED"
    )
    write_outputs(report, Path(args.output_json), Path(args.output_md))
    print(f"{report['status']} probe_infinicore_dispatch.py")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
