"""Run the interpreter regressions in an environment without GPU packages."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSA_TESTS = (
    "test_ssa_application_lowering.py",
    "test_ssa_first_backend_lowering.py",
    "test_ssa_pass_pipeline.py",
    "test_ssa_program_domain_regressions.py",
    "test_ssa_validation.py",
    "test_ir_immutability.py",
    "test_kernel_ir.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, help="Write the pytest JUnit report.")
    args = parser.parse_args()
    installed = [
        name
        for name in ("torch", "triton")
        if importlib.util.find_spec(name) is not None
    ]

    if installed:
        parser.error(
            "Use a fresh CPU environment without "
            + ", ".join(installed)
            + "; install requirements-cpu.txt only."
        )

    tests = sorted(
        path
        for path in (ROOT / "tests").glob("test_interpreter_*.py")
        if path.name != "test_interpreter_torch.py"
    )
    tests.extend(ROOT / "tests" / name for name in SSA_TESTS)

    if not tests or any(not path.is_file() for path in tests):
        parser.error("The interpreter test checkout is incomplete.")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "-ra",
        "--tb=short",
        *(str(path.relative_to(ROOT)) for path in tests),
        "-k",
        "not test_cpu_interpreter_matches_actual_triton_gpu",
    ]

    if args.junitxml is not None:
        report = args.junitxml.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        command.extend(("--junitxml", str(report)))

    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CUDA_VISIBLE_DEVICES="",
    )
    print(
        f"CPU interpreter: {len(tests)} test modules; Torch/Triton absent; "
        "the actual GPU differential cases are deselected.",
        flush=True,
    )

    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
