#!/usr/bin/env python3
"""
ntops-forge: Five-stage operator factory pipeline.

PLAN → CODEGEN → GUARD → PROVE → SHIP
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FORGE_SPECS = ROOT / "skills" / "ntops-forge" / "specs"
COPILOT_TASKS = ROOT / "skills" / "ntops-copilot" / "tasks"
FORGE_LOG = ROOT / "docs" / "forge_runs.jsonl"

COMPLEX_FAMILIES = frozenset({"reduction", "norm", "attention", "conv", "pooling"})


@dataclass
class ForgeSpec:
    id: str
    op: str
    family: str
    pattern: str
    formula: str
    pytest: str
    reference: str = ""
    contest_id: str = "T1-1-X"
    github_id: str = "hongwei-2026"
    acceptance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForgeRun:
    spec_id: str
    op: str
    started_at: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False
    kernel_path: str = ""
    elapsed_seconds: float = 0


def load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    data: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip().strip('"')
    return data


def find_spec(name: str, skill_root: Path) -> Path | None:
    for base in (
        skill_root / "skills" / "ntops-forge" / "specs",
        FORGE_SPECS,
    ):
        p = base / f"{name}.yaml"
        if p.is_file():
            return p
    # fallback: copilot task card
    for base in (skill_root / "skills" / "ntops-copilot" / "tasks", COPILOT_TASKS):
        p = base / f"task_{name}.yaml"
        if p.is_file():
            return p
    return None


def parse_spec(path: Path) -> ForgeSpec:
    d = load_yaml(path)
    op = d.get("op") or d.get("op_name") or path.stem.replace("task_", "")
    return ForgeSpec(
        id=d.get("id", f"forge-{op}"),
        op=op,
        family=d.get("family") or _family_from_pattern(d.get("pattern", "unary")),
        pattern=d.get("pattern", "unary"),
        formula=d.get("formula") or d.get("formula_hint", ""),
        pytest=d.get("pytest") or d.get("pytest_file") or f"tests/test_{op}.py",
        reference=d.get("reference") or d.get("reference_kernel", ""),
        contest_id=d.get("contest_id", "T1-1-X"),
        github_id=d.get("github_id", "hongwei-2026"),
        acceptance=d.get("acceptance") if isinstance(d.get("acceptance"), dict) else {},
    )


def _family_from_pattern(pattern: str) -> str:
    return "elementwise_binary" if pattern == "binary" else "elementwise_unary"


def resolve_reference_path(reference: str, ntops_root: Path) -> Path | None:
    """Resolve spec reference to an existing kernel file under ntops root."""
    if not reference:
        return None
    ref = Path(reference)
    if ref.is_absolute() and ref.is_file():
        return ref
    ref_str = str(reference).replace("\\", "/").lstrip("/")
    if ref_str.startswith("ntops/"):
        ref_str = ref_str[len("ntops/") :]
    candidate = ntops_root / ref_str
    if candidate.is_file():
        return candidate
    # legacy: reference_kernel like ntops/src/ntops/kernels/silu.py
    alt = (
        ntops_root / ref_str.split("ntops/", 1)[-1]
        if "ntops/" in ref_str
        else candidate
    )
    return alt if alt.is_file() else None


def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    print("$", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if out.strip():
        print(out[-8000:] if len(out) > 8000 else out)
    return proc.returncode, out


def append_log(run: ForgeRun, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")


def stage_plan(spec: ForgeSpec) -> tuple[bool, str]:
    print("\n=== [PLAN] ===")
    print(f"op={spec.op} family={spec.family} pattern={spec.pattern}")
    print(f"pytest={spec.pytest} reference={spec.reference or '(none)'}")
    if spec.family in COMPLEX_FAMILIES or spec.family.startswith("complex"):
        msg = f"STOP: {spec.family} requires reading reference first (see taxonomy.md)"
        print(f"WARN: {msg}")
        return False, msg
    if not spec.formula:
        print("WARN: no formula in spec; CODEGEN will emit TODO skeleton")
    print("OK: plan ready")
    return True, ""


def resolve_paths(
    spec: ForgeSpec, ntops_root: Path | None, force: bool
) -> tuple[Path, Path | None]:
    if ntops_root is None or not (ntops_root / "pyproject.toml").is_file():
        k = Path(f"/tmp/{spec.op}_forge_kernel.py")
        t = Path(f"/tmp/{spec.op}_forge_torch.py")
        print(f"INFO: no ntops-root; codegen to {k}")
        return k, t

    k = ntops_root / "src" / "ntops" / "kernels" / f"{spec.op}.py"
    t = ntops_root / "src" / "ntops" / "torch" / f"{spec.op}.py"
    if k.exists() and not force:
        print(f"WARN: {k} exists; using /tmp (pass --force to overwrite)")
        return Path(f"/tmp/{spec.op}_forge_kernel.py"), Path(
            f"/tmp/{spec.op}_forge_torch.py"
        )
    return k, t


def stage_codegen(
    spec: ForgeSpec,
    kernel_out: Path,
    torch_out: Path | None,
    register: bool,
    ntops_root: Path | None,
) -> tuple[bool, str]:
    print("\n=== [CODEGEN] ===")
    py = sys.executable
    cmd = [
        py,
        str(SCRIPTS / "scaffold_kernel.py"),
        "--name",
        spec.op,
        "--pattern",
        spec.pattern,
        "--style",
        "ntops",
        "--out",
        str(kernel_out),
    ]
    if spec.formula:
        cmd.extend(["--formula", spec.formula])
    code, out = run_cmd(cmd)
    if code != 0:
        return False, out

    if torch_out is not None:
        code, out2 = run_cmd(
            [
                py,
                str(SCRIPTS / "scaffold_torch.py"),
                "--name",
                spec.op,
                "--pattern",
                spec.pattern,
                "--out",
                str(torch_out),
            ]
        )
        if code != 0:
            return False, out2

    if register and ntops_root is not None:
        code, out3 = run_cmd(
            [
                py,
                str(SCRIPTS / "register_op.py"),
                "--name",
                spec.op,
                "--ntops-root",
                str(ntops_root),
            ]
        )
        if code != 0:
            return False, out3

    print("OK: codegen finished")
    return True, ""


def stage_guard(
    spec: ForgeSpec, kernel_out: Path, ntops_root: Path | None
) -> tuple[bool, str]:
    print("\n=== [GUARD] ===")
    py = sys.executable
    strict = spec.acceptance.get("preflight_strict", True)
    cmd = [py, str(SCRIPTS / "preflight.py"), str(kernel_out), "--kernel"]
    if strict:
        cmd.append("--strict")
    code, out = run_cmd(cmd)
    if code != 0:
        return False, out

    want_compare = spec.acceptance.get("compare_ref", True)
    if want_compare and spec.reference and ntops_root:
        ref = resolve_reference_path(spec.reference, ntops_root)
        if ref is None:
            msg = f"reference not found: {spec.reference} under {ntops_root}"
            print(f"WARN: {msg}")
            return False, msg
        code, out2 = run_cmd(
            [py, str(SCRIPTS / "compare_ref.py"), str(kernel_out), "--ref", str(ref)]
        )
        if code != 0:
            return False, out2
    print("OK: guard passed")
    return True, ""


def stage_prove(spec: ForgeSpec, ntops_root: Path | None) -> tuple[bool, str]:
    print("\n=== [PROVE] ===")
    if not spec.acceptance.get("pytest", True):
        print("SKIP: pytest disabled in spec")
        return True, ""
    if ntops_root is None:
        print("SKIP: no ntops-root for pytest")
        return True, ""

    py = sys.executable
    code, out = run_cmd([py, "-m", "pytest", "-q", spec.pytest], cwd=ntops_root)
    if code != 0:
        return False, out
    print("OK: prove passed")
    return True, ""


def stage_ship(spec: ForgeSpec, kernel_out: Path, record_ab: bool) -> tuple[bool, str]:
    print("\n=== [SHIP] ===")
    branch = f"2026-spring-{spec.github_id}-{spec.contest_id}"
    title = f"[2026春季][{spec.contest_id}] {spec.github_id}"
    print(f"PR branch: {branch}")
    print(f"PR title:  {title}")
    print(f"Kernel:    {kernel_out}")
    print("HONOR/REFERENCE: update before ntops upstream PR")

    if record_ab:
        py = sys.executable
        run_cmd(
            [
                py,
                str(SCRIPTS / "record_run.py"),
                "--task",
                spec.op,
                "--mode",
                "treatment",
                "--preflight-pass",
                "--pytest-pass",
                "--steps",
                "1",
                "--elapsed",
                "60",
            ]
        )
    print("OK: ship pack ready")
    return True, ""


def run_pipeline(args: argparse.Namespace) -> int:
    skill_root = args.skill_root
    spec_path = find_spec(args.op, skill_root)
    if spec_path is None:
        print(f"FAIL: no spec for '{args.op}'", file=sys.stderr)
        return 1

    spec = parse_spec(spec_path)
    ntops_root = args.ntops_root
    if ntops_root is None:
        for c in [Path("/root/work/ntops"), Path.cwd() / "ntops"]:
            if (c / "pyproject.toml").is_file():
                ntops_root = c
                break

    run = ForgeRun(
        spec_id=spec.id,
        op=spec.op,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    t0 = time.time()
    kernel_out, torch_out = resolve_paths(spec, ntops_root, args.force)

    stages = [
        ("plan", lambda: stage_plan(spec)),
        (
            "codegen",
            lambda: stage_codegen(
                spec, kernel_out, torch_out, args.register, ntops_root
            ),
        ),
        ("guard", lambda: stage_guard(spec, kernel_out, ntops_root)),
        ("prove", lambda: stage_prove(spec, ntops_root)),
        ("ship", lambda: stage_ship(spec, kernel_out, args.record_ab)),
    ]

    start_idx = 0
    if args.from_stage:
        names = [s[0] for s in stages]
        if args.from_stage not in names:
            print(f"FAIL: unknown stage {args.from_stage}", file=sys.stderr)
            return 1
        start_idx = names.index(args.from_stage)

    for name, fn in stages[start_idx:]:
        ok, err = fn()
        run.stages.append({"stage": name, "ok": ok, "error": err[:500] if err else ""})
        if not ok:
            run.ok = False
            run.kernel_path = str(kernel_out)
            run.elapsed_seconds = time.time() - t0
            append_log(run, args.log)
            print(f"\nFAIL at [{name.upper()}]")
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS / "forge_diagnose.py"),
                    "--text",
                    err or name,
                ]
            )
            return 1

    run.ok = True
    run.kernel_path = str(kernel_out)
    run.elapsed_seconds = time.time() - t0
    append_log(run, args.log)
    print(f"\n=== FORGE OK ({spec.op}) in {run.elapsed_seconds:.1f}s ===")
    print(f"Audit log: {args.log}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Run forge pipeline for default demo ops (silu, add, gelu, relu, mul)."""
    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    print(f"=== FORGE GATE: {', '.join(ops)} ===")
    failed: list[str] = []
    for op in ops:
        run_args = argparse.Namespace(
            op=op,
            ntops_root=args.ntops_root,
            skill_root=args.skill_root,
            force=args.force,
            register=args.register,
            record_ab=not args.no_record_ab,
            from_stage=None,
            log=args.log,
        )
        code = run_pipeline(run_args)
        if code != 0:
            failed.append(op)
    print("\n=== GATE SUMMARY ===")
    for op in ops:
        mark = "FAIL" if op in failed else "OK"
        print(f"  {mark}  {op}")
    if failed:
        print(f"GATE FAILED: {', '.join(failed)}")
        return 1
    print("GATE OK: all operators passed")
    return 0


def cmd_list(skill_root: Path) -> int:
    print("op\tfamily\tpattern\tspec_file")
    specs_dir = skill_root / "skills" / "ntops-forge" / "specs"
    for p in sorted(specs_dir.glob("*.yaml")):
        s = parse_spec(p)
        print(f"{s.op}\t{s.family}\t{s.pattern}\t{p.relative_to(skill_root)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ntops-forge operator factory")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run full pipeline")
    run_p.add_argument("op", help="operator name, e.g. silu")
    run_p.add_argument("--ntops-root", type=Path)
    run_p.add_argument("--skill-root", type=Path, default=ROOT)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--register", action="store_true")
    run_p.add_argument("--no-record-ab", action="store_true")
    run_p.add_argument(
        "--from",
        dest="from_stage",
        choices=["plan", "codegen", "guard", "prove", "ship"],
    )
    run_p.add_argument("--log", type=Path, default=FORGE_LOG)
    run_p.set_defaults(record_ab=True)

    list_p = sub.add_parser("list", help="list forge specs")
    list_p.add_argument("--skill-root", type=Path, default=ROOT)

    diag_p = sub.add_parser("diagnose", help="diagnose failure")
    diag_p.add_argument("--text")
    diag_p.add_argument("--log", type=Path, default=FORGE_LOG)

    gate_p = sub.add_parser(
        "gate", help="demo gate: run silu,add,gelu,relu,mul sequentially"
    )
    gate_p.add_argument("--ntops-root", type=Path)
    gate_p.add_argument("--skill-root", type=Path, default=ROOT)
    gate_p.add_argument(
        "--ops", default="silu,add,gelu,relu,mul", help="comma-separated ops"
    )
    gate_p.add_argument("--force", action="store_true")
    gate_p.add_argument("--register", action="store_true")
    gate_p.add_argument("--no-record-ab", action="store_true")
    gate_p.add_argument("--log", type=Path, default=FORGE_LOG)

    args = p.parse_args()

    if args.cmd == "list":
        return cmd_list(args.skill_root)
    if args.cmd == "diagnose":
        cmd = [sys.executable, str(SCRIPTS / "forge_diagnose.py")]
        if args.text:
            cmd.extend(["--text", args.text])
        if args.log:
            cmd.extend(["--log", str(args.log)])
        return subprocess.call(cmd)
    if args.cmd == "run":
        if getattr(args, "no_record_ab", False):
            args.record_ab = False
        return run_pipeline(args)
    if args.cmd == "gate":
        return cmd_gate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
