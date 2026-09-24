"""Run the documented fault demo and replay its saved comparison independently."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import compare_programs, load_reproducer


def test_demo_exports_the_fault_and_replay_rejects_a_missing_fault(tmp_path):
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path / "case"
    environment = dict(
        os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES=""
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "docs/cpu_interpreter_demo.py"),
            "--debug",
            "--export",
            str(directory),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Correct CPU output: (11,), float32" in completed.stdout
    assert "First bad pass: injected_bad_constant" in completed.stdout
    assert "not a historical bug" in completed.stdout
    assert "paused (2, 0, 0)" in completed.stdout
    reference, inputs, options = load_reproducer(directory / "reference")
    candidate, candidate_inputs, candidate_options = load_reproducer(
        directory / "candidate"
    )
    original_inputs = {name: value.copy() for name, value in inputs.items()}
    comparison = compare_programs(reference, candidate, inputs, **options)
    assert not comparison.equal
    assert comparison.output_differences == ("out",)
    assert comparison.first_operation.opcode == "arith.constant"

    for name, value in inputs.items():
        np.testing.assert_array_equal(value, original_inputs[name])
        np.testing.assert_array_equal(candidate_inputs[name], original_inputs[name])

    reference_result = interpret_program(reference, inputs, **options)
    candidate_result = interpret_program(
        candidate, candidate_inputs, **candidate_options
    )
    expected = original_inputs["x"] * 2 + 1
    np.testing.assert_allclose(
        reference_result.outputs["out"], expected, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        candidate_result.outputs["out"],
        original_inputs["x"] * 3 + 1,
        rtol=1e-3,
        atol=1e-3,
    )
    replay = subprocess.run(
        [sys.executable, str(directory / "replay.py")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "First bad pass (saved boundary): injected_bad_constant" in replay.stdout
    assert (directory / "failure.json").is_file()
    assert "Different outputs: ('out',)" in replay.stdout
    assert "arith.constant" in replay.stdout
    candidate_path = directory / "candidate/program.json"
    candidate_path.write_bytes((directory / "reference/program.json").read_bytes())
    without_fault = subprocess.run(
        [sys.executable, str(directory / "replay.py")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert without_fault.returncode != 0
    assert "no longer reproduces the recorded output difference" in without_fault.stderr
