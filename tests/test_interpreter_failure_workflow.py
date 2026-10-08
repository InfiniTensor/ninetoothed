"""Exercise pass-local diagnosis and automatic, independently runnable failures."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ninetoothed.interpreter import InterpretationError
from ninetoothed.interpreter.debugger import (
    check_passes,
    compare_programs,
    load_reproducer,
)
from ninetoothed.ir import ssa

from .test_interpreter_debugger import (
    _affine_restructured,
    _change_scale,
    _inputs,
    _kernel,
)
from .test_interpreter_provenance import _arithmetic, _branch_case, _split_scale, _value


def _change_offset(program):
    block = program.blocks[0]
    operations = tuple(
        replace(op, attrs=dict(op.attrs, value=2))
        if op.opcode == "arith.constant" and op.attrs["value"] == 1
        else op
        for op in block.operations
    )

    return replace(program, blocks=(replace(block, operations=operations),))


def _replay(directory, *, optimize=False):
    command = [sys.executable]

    if optimize:
        command.append("-O")

    return subprocess.run(
        [*command, str(directory / "replay.py")],
        cwd=directory.parent,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_a_correct_restructure_does_not_hide_a_later_bad_operation(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    originals = {name: value.copy() for name, value in inputs.items()}
    report = check_passes(
        kernel.program,
        (
            ("restructure", lambda _: _kernel(_affine_restructured).program),
            ("bad_offset", _change_offset),
        ),
        inputs,
        tensors=kernel.tensors,
        failure_dir=tmp_path / "failure",
        seed=2026,
    )
    assert not report.passed
    assert report.first_bad_pass == "bad_offset"
    assert report.difference.first_operation is None
    assert not report.adjacent_difference.equal
    assert report.localization.reference == "previous"
    assert report.localization.operation.opcode == "arith.constant"
    assert report.localization.traces_aligned
    assert report.reproducer == tmp_path / "failure"
    assert report.export_error is None
    metadata = json.loads((report.reproducer / "failure.json").read_text())
    assert metadata["report"]["checked_passes"] == ["restructure", "bad_offset"]
    assert metadata["seed"] == 2026
    assert (report.reproducer / "previous/program.ssa").is_file()
    assert _replay(report.reproducer).returncode == 0

    for name, value in inputs.items():
        np.testing.assert_array_equal(value, originals[name])


def test_pass_diagnosis_uses_a_proven_prefix_before_control_flow_diverges():
    original, wrong = _branch_case()
    report = check_passes(original, (("bad_condition", lambda _: wrong),), {})
    assert not report.passed
    assert not report.adjacent_difference.traces_aligned
    assert report.difference.first_operation is None
    assert report.localization.operation.location == "entry:0:arith.constant"
    assert not report.localization.traces_aligned


def test_restructuring_fault_is_observed_at_a_declared_retained_consumer(tmp_path):
    report = check_passes(
        _arithmetic(),
        (("split_scale", lambda program: _split_scale(program, wrong=True)),),
        {"x": np.int32(3)},
        failure_dir=tmp_path / "split",
    )
    assert not report.passed
    assert report.difference.first_operation is None
    location = report.localization
    assert location.basis == "retained_boundary"
    assert location.reference_location == "entry:3:arith.add"
    assert location.operation.location == "entry:4:arith.add"
    assert location.operation.result_name == "%scaled"
    assert location.operation.component == "input"
    assert report.source_candidates[0].opcode == "arith.mul"
    assert _replay(report.reproducer).returncode == 0
    repaired = check_passes(
        _arithmetic(),
        (("split_scale", _split_scale),),
        {"x": np.int32(3)},
    )
    assert repaired.passed
    assert repaired.localization is None


def test_unrecorded_restructuring_does_not_invent_a_retained_boundary():
    kernel = _kernel()
    report = check_passes(
        kernel.program,
        (("rebuild", lambda _: _change_offset(_kernel(_affine_restructured).program)),),
        _inputs(),
        tensors=kernel.tensors,
    )
    assert not report.passed
    assert report.localization is None
    assert report.difference.retained_operation is None


def test_a_store_mask_fault_has_a_location_even_without_ssa_results(tmp_path):
    kernel = _kernel()

    def change_mask(program):
        block = program.blocks[0]
        operations = tuple(
            replace(op, attrs=dict(op.attrs, mask=False))
            if op.opcode == "mem.store"
            else op
            for op in block.operations
        )

        return replace(program, blocks=(replace(block, operations=operations),))

    report = check_passes(
        kernel.program,
        (("bad_mask", change_mask),),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path / "mask",
    )
    assert not report.passed
    assert report.localization.operation.opcode == "mem.store"
    assert report.localization.operation.component == "mask"
    assert report.localization.operation.result_name == "mask"
    replay = _replay(report.reproducer)
    assert replay.returncode == 0, replay.stdout + replay.stderr


def test_cumulative_small_drift_is_still_detected_and_replayed(tmp_path):
    value = _value("%out", "float32")
    original = ssa.Program(
        kind="drift",
        outputs=(value,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(value,), attrs={"value": 0.0}
                    ),
                )
            ),
        ),
    )

    def drift(program):
        block = program.blocks[0]
        op = block.operations[0]

        return replace(
            program,
            blocks=(
                replace(
                    block,
                    operations=(
                        replace(op, attrs={"value": op.attrs["value"] + 0.00075}),
                    ),
                ),
            ),
        )

    report = check_passes(
        original,
        (("drift_1", drift), ("drift_2", drift)),
        {},
        rtol=0,
        atol=0.001,
        failure_dir=tmp_path / "drift",
        seed=7,
    )
    assert report.first_bad_pass == "drift_2"
    assert report.adjacent_difference.equal
    assert not report.difference.equal
    assert report.localization.reference == "original"
    assert _replay(report.reproducer).returncode == 0


def test_comparison_exports_generic_inputs_and_replay_rejects_a_repair(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    inputs["x"] = inputs["x"][::-1]
    inputs["x"].flags.writeable = False
    report = compare_programs(
        kernel.program,
        _change_scale(kernel.program),
        inputs,
        tensors=kernel.tensors,
        failure_dir=tmp_path / "pair",
        seed=83,
    )
    assert not report.equal
    assert report.export_error is None
    _, saved, _ = load_reproducer(report.reproducer / "reference")
    assert saved["x"].strides == inputs["x"].strides
    assert not saved["x"].flags.writeable
    manifest = json.loads((report.reproducer / "reference/manifest.json").read_text())
    assert manifest["seed"] == 83
    assert manifest["inputs"]["x"]["shape"] == [7]
    assert manifest["inputs"]["x"]["dtype"] == "float32"
    replay = _replay(report.reproducer)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "Different outputs: ('out',)" in replay.stdout
    (report.reproducer / "candidate/program.json").write_bytes(
        (report.reproducer / "reference/program.json").read_bytes()
    )
    repaired = _replay(report.reproducer, optimize=True)
    assert repaired.returncode != 0
    assert "no longer reproduces" in repaired.stderr


def test_success_never_creates_a_failure_directory(tmp_path):
    kernel = _kernel()
    directory = tmp_path / "success"
    report = check_passes(
        kernel.program,
        (("identity", lambda program: program),),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=directory,
    )
    assert report.passed
    assert report.reproducer is None
    assert not directory.exists()


def test_existing_directory_preserves_failure_and_never_overwrites(tmp_path):
    kernel = _kernel()
    marker = tmp_path / "owned.txt"
    marker.write_text("preserve")
    report = compare_programs(
        kernel.program,
        _change_scale(kernel.program),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path,
    )
    assert not report.equal
    assert report.reproducer is None
    assert "FileExistsError" in report.export_error
    assert marker.read_text() == "preserve"
    assert tuple(tmp_path.iterdir()) == (marker,)


def _unsupported(program):
    block = program.blocks[0]

    return replace(
        program,
        blocks=(
            replace(
                block, operations=(ssa.Operation(opcode="broken.op"), *block.operations)
            ),
        ),
    )


@pytest.mark.parametrize("workflow", ("comparison", "passes"))
def test_runtime_failure_is_automatically_saved_without_masking_exception(
    tmp_path, workflow
):
    kernel = _kernel()
    directory = tmp_path / workflow

    if workflow == "comparison":
        with pytest.raises(InterpretationError, match="broken.op") as caught:
            compare_programs(
                kernel.program,
                _unsupported(kernel.program),
                _inputs(),
                tensors=kernel.tensors,
                failure_dir=directory,
                seed=19,
            )

        assert caught.value.reproducer == directory
        assert caught.value.export_error is None
    else:
        report = check_passes(
            kernel.program,
            (("unsupported", _unsupported),),
            _inputs(),
            tensors=kernel.tensors,
            failure_dir=directory,
            seed=19,
        )
        assert report.first_bad_pass == "unsupported"
        assert "broken.op" in report.error
        assert report.reproducer == directory

    replay = _replay(directory)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "broken.op" in replay.stdout


def test_reference_failure_is_not_blame_assigned_to_the_first_pass():
    kernel = _kernel()
    visited = []

    def identity(program):
        visited.append(True)

        return program

    with pytest.raises(InterpretationError, match="broken.op"):
        check_passes(
            _unsupported(kernel.program),
            (("identity", identity),),
            _inputs(),
            tensors=kernel.tensors,
        )

    assert not visited


def test_transform_error_exports_diagnostics_without_claiming_executable_pass(tmp_path):
    kernel = _kernel()

    def broken(_):
        raise ValueError("Injected transform failure.")

    report = check_passes(
        kernel.program,
        (("broken", broken),),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path / "transform",
    )
    assert report.first_bad_pass == "broken"
    assert report.error == "Injected transform failure."
    assert report.reproducer is not None
    metadata = json.loads((report.reproducer / "failure.json").read_text())
    assert metadata["phase"] == "transform"
    assert not metadata["replayable"]
    replay = _replay(report.reproducer)
    assert replay.returncode != 0
    assert "executable pass code was not exported" in replay.stderr


def test_export_io_error_does_not_replace_the_execution_error(tmp_path, monkeypatch):
    from ninetoothed.interpreter import failure

    def disk_full(*args, **kwargs):
        raise OSError("Injected disk full.")

    monkeypatch.setattr(failure, "export_reproducer", disk_full)
    kernel = _kernel()

    with pytest.raises(InterpretationError, match="broken.op") as caught:
        compare_programs(
            kernel.program,
            _unsupported(kernel.program),
            _inputs(),
            tensors=kernel.tensors,
            failure_dir=tmp_path / "disk",
        )

    assert caught.value.reproducer is None
    assert caught.value.export_error == "OSError: Injected disk full."
    assert (tmp_path / "disk/INCOMPLETE").is_file()


def test_adjacent_failure_cannot_be_hidden_by_cancellation_against_original(tmp_path):
    literal = _value("%out", "float32")

    def constant(value):
        return ssa.Program(
            kind="cancellation",
            outputs=(literal,),
            blocks=(
                ssa.Block(
                    operations=(
                        ssa.Operation(
                            opcode="arith.constant",
                            results=(literal,),
                            attrs={"value": value},
                        ),
                    )
                ),
            ),
        )

    original = constant(0.0)
    report = check_passes(
        original,
        (
            ("positive", lambda _: constant(0.00075)),
            ("negative", lambda _: constant(-0.00075)),
        ),
        {},
        rtol=0,
        atol=0.001,
        failure_dir=tmp_path / "cancellation",
    )
    assert report.first_bad_pass == "negative"
    assert report.difference.equal
    assert not report.adjacent_difference.equal
    assert report.localization.reference == "previous"
    assert _replay(report.reproducer).returncode == 0


@pytest.mark.parametrize("original_failure", ("mismatch", "exception"))
def test_replay_rejects_a_different_kind_of_failure(tmp_path, original_failure):
    kernel = _kernel()
    directory = tmp_path / original_failure

    if original_failure == "exception":
        with pytest.raises(InterpretationError):
            compare_programs(
                kernel.program,
                _unsupported(kernel.program),
                _inputs(),
                tensors=kernel.tensors,
                failure_dir=directory,
            )

        candidate = json.loads((directory / "candidate/program.json").read_text())
        candidate["blocks"][0]["operations"][0]["opcode"] = "another.unsupported"
    else:
        compare_programs(
            kernel.program,
            _change_scale(kernel.program),
            _inputs(),
            tensors=kernel.tensors,
            failure_dir=directory,
        )
        candidate = json.loads((directory / "candidate/program.json").read_text())
        candidate["blocks"][0]["operations"][0]["opcode"] = "another.unsupported"

    (directory / "candidate/program.json").write_text(json.dumps(candidate))
    replay = _replay(directory)
    assert replay.returncode != 0
    assert "different execution failure" in replay.stderr


def test_invalid_ssa_is_a_diagnostic_snapshot_with_the_verification_cause(tmp_path):
    kernel = _kernel()

    def undefined(program):
        block = program.blocks[0]
        op = block.operations[-1]

        return replace(
            program,
            blocks=(
                replace(
                    block,
                    operations=(
                        *block.operations[:-1],
                        replace(op, operands=("%missing", "out")),
                    ),
                ),
            ),
        )

    report = check_passes(
        kernel.program,
        (("undefined", undefined),),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path / "invalid",
    )
    assert not report.passed
    assert "%missing" in report.error
    assert report.export_error is None
    assert (report.reproducer / "candidate.json").is_file()
    metadata = json.loads((report.reproducer / "failure.json").read_text())
    assert metadata["phase"] == "record"
    assert not metadata["replayable"]


@pytest.mark.parametrize("kind", ("none", "runtime_error"))
def test_invalid_python_pass_result_or_exception_is_captured(tmp_path, kind):
    kernel = _kernel()

    def broken(_):
        if kind == "runtime_error":
            raise RuntimeError("Injected Python pass failure.")

        return None

    report = check_passes(
        kernel.program,
        (("python_pass", broken),),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path / kind,
    )
    assert not report.passed
    assert report.first_bad_pass == "python_pass"
    assert report.export_error is None
    assert report.reproducer is not None
    assert not json.loads((report.reproducer / "failure.json").read_text())[
        "replayable"
    ]


def test_unexpected_export_exception_preserves_the_original_mismatch(
    tmp_path, monkeypatch
):
    from ninetoothed.interpreter import failure

    def broken_export(*args, **kwargs):
        raise RuntimeError("Injected serializer failure.")

    monkeypatch.setattr(failure, "export_failure", broken_export)
    kernel = _kernel()
    report = compare_programs(
        kernel.program,
        _change_scale(kernel.program),
        _inputs(),
        tensors=kernel.tensors,
        failure_dir=tmp_path / "serializer",
    )
    assert not report.equal
    assert report.reproducer is None
    assert report.export_error == "RuntimeError: Injected serializer failure."
