"""Execute frontend dtype references using the actual referenced CPU value."""

from dataclasses import replace

import numpy as np
import pytest

from ninetoothed.frontend.python import from_source
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.memory import TensorRef
from ninetoothed.ir import TensorSpec, ssa

_SOURCES = (
    ("parameter", "", "y", "int16"),
    ("arithmetic", "z = y + y", "z", "int16"),
    ("promoted", "z = y + f", "z", "float32"),
    ("cast", "z = y.to(int32)", "z", "int32"),
    ("dynamic_cast", "z = x.to(y.dtype)", "z", "int16"),
    ("constructor_chain", "z = zeros((3,), dtype=y.dtype)", "z", "int16"),
)


def _program(body):
    tensors = (
        TensorSpec(ndim=1, shape=(3,), dtype="float64", name="x"),
        TensorSpec(ndim=1, shape=(3,), dtype=None, name="y"),
        TensorSpec(ndim=1, shape=(3,), dtype=None, name="f"),
        TensorSpec(ndim=1, shape=(3,), dtype="float64", name="out"),
    )
    source = "def app(x, y, f, out):\n" + "".join(
        f"    {line}\n" for line in body.splitlines() if line
    )

    return from_source(source, tensors, kind="dtype_provenance"), tensors


def _execute(program, tensors, *, y=None):
    inputs = {
        "x": np.array([0.25, 1.75, 3.5], dtype=np.float64),
        "y": np.array([2, 3, 4], dtype=np.int16) if y is None else y,
        "f": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "out": np.full(3, -123, dtype=np.float64),
    }
    original = {name: value.copy() for name, value in inputs.items() if name != "out"}
    result = interpret_program(program, inputs, tensors=tensors, trace=True)

    for name, expected in original.items():
        np.testing.assert_array_equal(inputs[name], expected)

    return result


def _last_snapshot(result, opcode):
    events = [event for event in result.trace if event.opcode == opcode]
    assert events

    return next(iter(events[-1].results.values()))


@pytest.mark.parametrize("label,setup,reference,dtype", _SOURCES, ids=lambda x: x)
def test_frontend_cast_uses_referenced_runtime_dtype(label, setup, reference, dtype):
    del label
    program, tensors = _program(f"{setup}\nout = x.to({reference}.dtype)")
    result = _execute(program, tensors)
    expected = np.array([0.25, 1.75, 3.5], dtype=np.float64).astype(dtype)
    snapshot = _last_snapshot(result, "tensor.cast")
    assert snapshot["dtype"] == dtype
    np.testing.assert_array_equal(snapshot["value"], expected)
    np.testing.assert_array_equal(result.outputs["out"], expected.astype(np.float64))


@pytest.mark.parametrize("constructor", ("zeros", "empty", "full"))
@pytest.mark.parametrize("label,setup,reference,dtype", _SOURCES, ids=lambda x: x)
def test_frontend_constructor_uses_referenced_runtime_dtype(
    constructor, label, setup, reference, dtype
):
    del label
    fill = ", 1.75" if constructor == "full" else ""
    program, tensors = _program(
        f"{setup}\nout = {constructor}((3,){fill}, dtype={reference}.dtype)"
    )
    result = _execute(program, tensors)
    opcode = "tensor.full" if constructor == "full" else "tensor.zeros"
    snapshot = _last_snapshot(result, opcode)
    expected = np.full(3, 1.75 if constructor == "full" else 0, dtype=dtype)
    assert snapshot["dtype"] == dtype
    np.testing.assert_array_equal(snapshot["value"], expected)
    np.testing.assert_array_equal(result.outputs["out"], expected.astype(np.float64))


@pytest.mark.parametrize("scalar", (False, True))
def test_dynamic_parameter_dtype_is_resolved_on_each_execution(scalar):
    program, tensors = _program("out = x.to(y.dtype)")

    if scalar:
        tensors = tuple(
            replace(tensor, ndim=0, shape=()) if tensor.name == "y" else tensor
            for tensor in tensors
        )
        program = from_source(
            "def app(x, y, f, out):\n    out = x.to(y.dtype)\n",
            tensors,
            kind="dtype_provenance_scalar",
        )

    for dtype in ("int16", "float32", "int32"):
        y = np.asarray(2, dtype=dtype) if scalar else np.full(3, 2, dtype=dtype)
        result = _execute(program, tensors, y=y)
        assert _last_snapshot(result, "tensor.cast")["dtype"] == dtype
        expected = np.array([0.25, 1.75, 3.5], dtype=np.float64).astype(dtype)
        np.testing.assert_array_equal(
            result.outputs["out"], expected.astype(np.float64)
        )


@pytest.mark.parametrize(
    "form", ("cast_attribute", "cast_operand", "zeros", "empty", "full")
)
def test_dtype_query_does_not_read_its_tensor_source(form, monkeypatch):
    constructor = form if form in {"zeros", "empty", "full"} else None
    fill = ", 1.75" if constructor == "full" else ""
    body = (
        f"out = {constructor}((3,){fill}, dtype=y.dtype)"
        if constructor
        else "out = x.to(y.dtype)"
    )
    program, tensors = _program(body)

    if form == "cast_operand":
        operations = tuple(
            replace(op, operands=(op.operands[0], "y"), attrs={"dtype": None})
            if op.opcode == "tensor.cast"
            else op
            for op in program.blocks[0].operations
        )
        program = replace(
            program, blocks=(replace(program.blocks[0], operations=operations),)
        )

    read = TensorRef.read

    def guarded_read(self, *args, **kwargs):
        if self.spec is not None and self.spec.name == "y":
            raise AssertionError("A dtype query must not read the source tensor.")
        return read(self, *args, **kwargs)

    monkeypatch.setattr(TensorRef, "read", guarded_read)
    result = _execute(program, tensors)
    assert all(
        access.storage != "storage:y"
        for event in result.trace
        for access in event.memory or ()
    )

    if constructor:
        event = next(
            event
            for event in result.trace
            if event.opcode in {"tensor.zeros", "tensor.full"}
        )
        assert event.memory == ()
        expected = np.full(3, 1 if constructor == "full" else 0, dtype=np.float64)
    else:
        expected = np.array([0, 1, 3], dtype=np.float64)

    np.testing.assert_array_equal(result.outputs["out"], expected)


def test_pointer_dtype_reference_uses_storage_dtype_without_a_load():
    program, tensors = _program("out = x.to(y.dtype)")
    pointer = ssa.Value(
        name="%dtype_pointer", type=ssa.Type(kind="pointer", dtype=None)
    )
    operations = (
        ssa.Operation(opcode="mem.data_ptr", operands=("y",), results=(pointer,)),
        *(
            replace(op, operands=(op.operands[0], pointer.name), attrs={"dtype": None})
            if op.opcode == "tensor.cast"
            else op
            for op in program.blocks[0].operations
        ),
    )
    program = replace(
        program, blocks=(replace(program.blocks[0], operations=operations),)
    )
    result = _execute(program, tensors)
    assert _last_snapshot(result, "tensor.cast")["dtype"] == "int16"
    np.testing.assert_array_equal(result.outputs["out"], [0, 1, 3])
    assert all(
        access.storage != "storage:y"
        for event in result.trace
        for access in event.memory or ()
    )
