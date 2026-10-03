"""Independent differential and on-disk replay checks using real frontend SSA."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import InterpretationError, interpret_program
from ninetoothed.interpreter.debugger import (
    StepDebugger,
    check_passes,
    compare_programs,
    export_reproducer,
    load_reproducer,
)
from ninetoothed.ir import ssa


def _arrangement(x, out):
    return x.tile((4,)), out.tile((4,))


def _affine(x, out):
    out = x * 2 + 1  # noqa: F841


def _affine_restructured(x, out):
    out = x + x + 1  # noqa: F841


def _affine_restructured_wrong(x, out):
    out = x + x + 2  # noqa: F841


def _kernel(application=_affine):
    return interpret(
        _arrangement,
        application,
        (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")),
    )


def _inputs():
    return {
        "x": np.random.default_rng(2026).normal(size=7).astype(np.float32),
        "out": np.full(7, -731, dtype=np.float32),
    }


def _change_scale(program):
    operations = tuple(
        replace(operation, attrs=dict(operation.attrs, value=3))
        if operation.opcode == "arith.constant" and operation.attrs["value"] == 2
        else operation
        for operation in program.blocks[0].operations
    )

    return replace(program, blocks=(replace(program.blocks[0], operations=operations),))


def test_difference_identifies_first_aligned_operation_without_mutating_inputs():
    kernel = _kernel()
    inputs = _inputs()
    unchanged = {name: value.copy() for name, value in inputs.items()}
    comparison = compare_programs(
        kernel.program, _change_scale(kernel.program), inputs, tensors=kernel.tensors
    )
    assert not comparison.equal
    assert comparison.output_differences == ("out",)
    assert comparison.traces_aligned
    difference = comparison.first_operation
    assert difference is not None
    assert difference.opcode == "arith.constant"
    assert difference.location.startswith("entry:")
    assert difference.program_id == (0, 0, 0)
    operation = next(
        op
        for op in kernel.program.blocks[0].operations
        if op.opcode == "arith.constant" and op.attrs["value"] == 2
    )
    assert difference.result_name == operation.results[0].name

    for name in inputs:
        np.testing.assert_array_equal(inputs[name], unchanged[name])


@pytest.mark.parametrize(
    "application,expected_equal",
    ((_affine_restructured, True), (_affine_restructured_wrong, False)),
)
def test_restructured_programs_do_not_get_a_guessed_operation_location(
    application, expected_equal
):
    reference = _kernel()
    candidate = _kernel(application)
    comparison = compare_programs(
        reference.program, candidate.program, _inputs(), tensors=reference.tensors
    )
    assert comparison.equal is expected_equal
    assert not comparison.traces_aligned
    assert comparison.first_operation is None


def test_pass_checker_stops_at_the_first_semantic_failure():
    kernel = _kernel()
    visited = []

    def after_failure(program):
        visited.append("should-not-run")

        return program

    result = check_passes(
        kernel.program,
        (
            ("identity", lambda program: program),
            (
                "metadata_only",
                lambda program: replace(program, metadata={"note": "test"}),
            ),
            ("broken_scale", _change_scale),
            ("unreachable", after_failure),
        ),
        _inputs(),
        tensors=kernel.tensors,
    )
    assert not result.passed
    assert result.checked_passes == ("identity", "metadata_only", "broken_scale")
    assert result.first_bad_pass == "broken_scale"
    assert result.difference.first_operation.opcode == "arith.constant"
    assert visited == []


def test_pass_checker_records_an_unsupported_operation_as_failure():
    kernel = _kernel()

    def unsupported(program):
        block = program.blocks[0]

        return replace(
            program,
            blocks=(
                replace(
                    block,
                    operations=(ssa.Operation(opcode="broken.op"), *block.operations),
                ),
            ),
        )

    result = check_passes(
        kernel.program,
        (("introduce_unsupported", unsupported),),
        _inputs(),
        tensors=kernel.tensors,
    )
    assert not result.passed
    assert result.first_bad_pass == "introduce_unsupported"
    assert "broken.op" in result.error
    assert "entry:0" in result.error


def test_reproducer_round_trips_arrangement_inputs_and_runs_its_replay_script(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    expected = inputs["x"] * 2 + 1
    original_x = inputs["x"].copy()
    directory = export_reproducer(
        tmp_path / "case",
        kernel.program,
        inputs,
        tensors=kernel.tensors,
        grid=(2,),
        symbols={"test_symbol": 17},
        seed=2026,
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    assert {path.name for path in directory.iterdir()} == {
        "program.json",
        "program.ssa",
        "inputs.npz",
        "manifest.json",
        "replay.py",
    }
    inputs["x"][:] = 999
    program, restored, options = load_reproducer(directory)
    assert ssa.render(program) == ssa.render(kernel.program)
    np.testing.assert_array_equal(restored["x"], original_x)
    assert options["grid"] == [2]
    assert options["symbols"] == {"test_symbol": 17}
    result = interpret_program(program, restored, **options)
    np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-3)
    completed = subprocess.run(
        [sys.executable, str(directory / "replay.py")],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "out" in completed.stdout


def test_reproducer_preserves_exact_alias_bindings(tmp_path):
    value_type = ssa.Type(kind="tensor", shape=("3",), dtype="int32")
    x = ssa.Value(name="x", type=value_type)
    y = ssa.Value(name="y", type=value_type)
    program = ssa.Program(
        kind="aliases", inputs=(x, y), outputs=(x, y), blocks=(ssa.Block(),)
    )
    shared = np.array([2, 3, 5], dtype=np.int32)
    directory = export_reproducer(
        tmp_path / "alias", program, {"x": shared, "y": shared}
    )
    restored_program, restored, options = load_reproducer(directory)
    assert restored["x"] is restored["y"]
    result = interpret_program(restored_program, restored, **options)
    np.testing.assert_array_equal(result.outputs["x"], shared)


def test_comparison_preserves_write_then_read_through_exact_aliases():
    tensor_type = ssa.Type(kind="tensor", shape=("3",), dtype="float32")
    x, alias, out = (
        ssa.Value(name=name, type=tensor_type) for name in ("x", "alias", "out")
    )
    one = ssa.Value(name="%one", type=ssa.Type(kind="scalar", dtype="float32"))
    operations = (
        ssa.Operation(opcode="arith.constant", results=(one,), attrs={"value": 1}),
        ssa.Operation(opcode="mem.store", operands=("%one", "x")),
        ssa.Operation(opcode="mem.store", operands=("alias", "out")),
    )
    program = ssa.Program(
        kind="alias_effect",
        inputs=(x, alias, out),
        outputs=(out,),
        blocks=(ssa.Block(operations=operations),),
    )
    candidate = replace(
        program,
        blocks=(
            ssa.Block(
                operations=(
                    *operations[:2],
                    ssa.Operation(opcode="mem.store", operands=("%one", "out")),
                )
            ),
        ),
    )
    shared = np.array([2, 3, 5], dtype=np.float32)
    inputs = {"x": shared, "alias": shared, "out": np.full(3, -731, dtype=np.float32)}
    assert compare_programs(program, candidate, inputs).equal
    np.testing.assert_array_equal(shared, [2, 3, 5])
    np.testing.assert_array_equal(inputs["out"], [-731, -731, -731])


@pytest.mark.parametrize("partial", (False, True))
def test_comparison_and_export_preserve_distinct_overlapping_views(tmp_path, partial):
    kernel = _kernel()
    storage = np.arange(8, dtype=np.float32)
    inputs = {"x": storage[:7], "out": storage[1:] if partial else storage[:7]}
    assert inputs["x"] is not inputs["out"]

    report = compare_programs(
        kernel.program, kernel.program, inputs, tensors=kernel.tensors
    )
    assert report.equal
    directory = export_reproducer(
        tmp_path / "overlap", kernel.program, inputs, tensors=kernel.tensors
    )
    program, restored, options = load_reproducer(directory)
    assert restored["x"] is not restored["out"]
    assert np.shares_memory(restored["x"], restored["out"])
    assert compare_programs(program, kernel.program, restored, **options).equal
    np.testing.assert_array_equal(storage, np.arange(8, dtype=np.float32))


def _raw_pointer_program():
    tensor_type = ssa.Type(kind="tensor", shape=("3",), dtype="float32")
    x = ssa.Value(name="x", type=tensor_type)
    pointer = ssa.Value(name="%ptr", type=ssa.Type(kind="pointer", dtype="float32"))
    loaded = ssa.Value(name="%loaded", type=ssa.Type(kind="scalar", dtype="float32"))

    return ssa.Program(
        kind="strided_pointer_replay",
        inputs=(x,),
        outputs=(loaded,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="mem.data_ptr", operands=("x",), results=(pointer,)
                    ),
                    ssa.Operation(
                        opcode="mem.load", operands=("%ptr",), results=(loaded,)
                    ),
                )
            ),
        ),
    )


def test_comparison_and_replay_preserve_noncontiguous_pointer_rejection(tmp_path):
    program = _raw_pointer_program()
    inputs = {"x": np.arange(6, dtype=np.float32)[::2]}

    with pytest.raises(InterpretationError, match="C-contiguous"):
        interpret_program(program, inputs)

    with pytest.raises(InterpretationError, match="C-contiguous"):
        compare_programs(program, program, inputs)

    export_reproducer(tmp_path / "strided", program, inputs)
    restored_program, restored, options = load_reproducer(tmp_path / "strided")
    assert restored["x"].strides == inputs["x"].strides

    with pytest.raises(InterpretationError, match="C-contiguous"):
        interpret_program(restored_program, restored, **options)


@pytest.mark.parametrize("layout", ("positive", "negative", "transpose", "broadcast"))
def test_reproducer_preserves_strides_values_and_readonly_flags(tmp_path, layout):
    data = np.arange(12, dtype=np.float32)
    value = {
        "positive": data[::2],
        "negative": data[::-1],
        "transpose": data.reshape(3, 4).T,
        "broadcast": np.broadcast_to(data[:1], (6,)),
    }[layout]
    value.flags.writeable = False
    x = ssa.Value(
        name="x",
        type=ssa.Type(
            kind="tensor", shape=tuple(map(str, value.shape)), dtype="float32"
        ),
    )
    program = ssa.Program(
        kind="strides", inputs=(x,), outputs=(x,), blocks=(ssa.Block(),)
    )
    export_reproducer(tmp_path / layout, program, {"x": value})
    _program, restored, _options = load_reproducer(tmp_path / layout)
    assert restored["x"].strides == value.strides
    assert not restored["x"].flags.writeable
    np.testing.assert_array_equal(restored["x"], value)
    assert not np.shares_memory(restored["x"], value)


def test_program_comparison_keeps_readonly_output_semantics():
    value = ssa.Value(
        name="x", type=ssa.Type(kind="tensor", shape=("3",), dtype="float32")
    )
    program = ssa.Program(
        kind="readonly",
        inputs=(value,),
        outputs=(value,),
        blocks=(
            ssa.Block(
                operations=(ssa.Operation(opcode="mem.store", operands=("x", "x")),)
            ),
        ),
    )
    array = np.arange(3, dtype=np.float32)
    array.flags.writeable = False

    with pytest.raises(InterpretationError, match="read-only"):
        compare_programs(program, program, {"x": array})


def test_reproducer_rejects_overwrite_and_manifest_input_mismatch(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    directory = export_reproducer(
        tmp_path / "case", kernel.program, inputs, tensors=kernel.tensors
    )
    original = (directory / "program.json").read_bytes()

    with pytest.raises(FileExistsError):
        export_reproducer(directory, kernel.program, inputs, tensors=kernel.tensors)

    assert (directory / "program.json").read_bytes() == original
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["x"]["shape"] = [99]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_reproducer(directory)


def test_reproducer_rejects_object_arrays_without_pickle(tmp_path):
    kernel = _kernel()
    inputs = _inputs()
    inputs["x"] = inputs["x"].astype(object)

    with pytest.raises(TypeError, match="numeric"):
        export_reproducer(
            tmp_path / "object", kernel.program, inputs, tensors=kernel.tensors
        )

    assert not (tmp_path / "object" / "inputs.npz").exists()


def _dot_arrangement(a, b, out):
    return a.tile((4, 4)), b.tile((4, 4)), out.tile((4, 4))


def _dot_application(a, b, out):
    out = a @ b  # noqa: F841


def _decomposed_dot():
    return interpret(
        _dot_arrangement,
        _dot_application,
        (
            Tensor(2, name="a", dtype="float32", other=0),
            Tensor(2, name="b", dtype="float32", other=0),
            Tensor(2, name="out", dtype="float32"),
        ),
        backend="triton",
    )


def _dot_inputs():
    return {
        "a": np.arange(9, dtype=np.float32).reshape(3, 3),
        "b": np.arange(6, dtype=np.float32).reshape(3, 2),
        "out": np.full((3, 2), -731, dtype=np.float32),
    }


def test_decomposed_dot_difference_identifies_the_output_lane():
    kernel = _decomposed_dot()
    block = kernel.program.blocks[0]
    operations = tuple(
        replace(operation, attrs=dict(operation.attrs, value=1.0))
        if operation.opcode == "arith.constant"
        and operation.results[0].type.dtype == "float32"
        and operation.attrs["value"] == 0.0
        else operation
        for operation in block.operations
    )
    candidate = replace(kernel.program, blocks=(replace(block, operations=operations),))
    comparison = compare_programs(
        kernel.program, candidate, _dot_inputs(), tensors=kernel.tensors
    )
    assert not comparison.equal
    assert comparison.traces_aligned
    assert comparison.first_operation.opcode == "arith.constant"
    assert comparison.first_operation.program_id == (0, 0, 0)
    assert comparison.first_operation.lane == (0, 0)


def test_step_debugger_discards_values_from_the_previous_output_lane():
    kernel = _decomposed_dot()
    debugger = StepDebugger(commands=(), stop_on_entry=False, output=lambda _line: None)
    inspected_transition = []

    def callback(event):
        debugger(event)

        if event.opcode == "index.offset" and event.lane == (0, 1):
            with pytest.raises(KeyError):
                debugger.inspect("%acc_iter")

            inspected_transition.append(event.lane)

    inputs = _dot_inputs()
    result = interpret_program(
        kernel.program, inputs, tensors=kernel.tensors, callback=callback
    )
    assert inspected_transition
    np.testing.assert_allclose(
        result.outputs["out"], inputs["a"] @ inputs["b"], rtol=1e-3, atol=1e-3
    )
