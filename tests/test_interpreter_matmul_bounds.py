"""Matmul decomposition preserves constant and arranged symbolic K extents."""

from dataclasses import replace

import numpy as np
import pytest

from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import Context, DecomposeLinalg, lower_for_target
from ninetoothed.frontend.python import from_source
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import compare_programs
from ninetoothed.ir import TensorSpec, ssa
from ninetoothed.ir.provenance import operation_locations


def _program(extent, dtype):
    program = from_source(
        "\ndef matmul(a, b, out):\n    out = a @ b\n",
        tuple(
            TensorSpec(ndim=2, shape=shape, dtype=dtype, name=name)
            for name, shape in (
                ("a", ("m", extent)),
                ("b", (extent, "n")),
                ("out", ("m", "n")),
            )
        ),
        kind="matmul",
    )
    assert program is not None

    return program


def _inputs(inner, dtype):
    a = (np.arange(3 * inner).reshape(3, inner) - 3).astype(dtype)
    b = (np.arange(inner * 2).reshape(inner, 2) % 5 - 2).astype(dtype)

    if dtype == "int32":
        # Values above float32's exact range expose accidental float accumulation.
        a += 2**24
    else:
        a /= 3
        b /= 5

    return {"a": a, "b": b, "out": np.full((3, 2), -731, dtype=dtype)}


def _loop_and_definitions(program):
    operations = program.blocks[0].operations
    loop = next(op for op in operations if op.opcode == "scf.for")
    definitions = {value.name: op for op in operations for value in op.results}

    return loop, definitions


def _assert_equal(actual, expected):
    assert actual.dtype == expected.dtype

    if expected.dtype.kind == "i":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ("float32", "int32"))
@pytest.mark.parametrize(
    "extent,inner,symbols,bound_opcode",
    (
        ("0", 0, {}, "arith.constant"),
        ("1", 1, {}, "arith.constant"),
        ("4", 4, {}, "arith.constant"),
        ("k", 4, {}, None),
        ("2*k", 4, {"k": 2}, "shape.dim"),
        ("2*2", 4, {}, "shape.dim"),
    ),
    ids=("zero", "one", "fixed", "symbolic", "compound", "compound-fixed"),
)
def test_matmul_bounds_preserve_values_and_provenance(
    dtype, extent, inner, symbols, bound_opcode
):
    reference = _program(extent, dtype)
    decomposed = DecomposeLinalg().run(
        reference, Context(backend=Target.CUDA, compiler_options={}, kernel_metadata={})
    )
    ssa.verify_program(decomposed)
    loop, definitions = _loop_and_definitions(decomposed)

    if bound_opcode is None:
        assert loop.operands[1] == "k"
    else:
        bound = definitions[loop.operands[1]]
        assert bound.opcode == bound_opcode
        assert bound.results[0].type == ssa.Type(kind="index", dtype="int64")

        if bound_opcode == "arith.constant":
            assert type(bound.attrs["value"]) is int
            assert bound.attrs["value"] == inner
        else:
            assert bound.operands == ("a",)
            assert bound.attrs["dim"] == -1

    initial = definitions[loop.operands[3]]
    assert initial.results[0].type.dtype == dtype
    assert type(initial.attrs["value"]) is int
    assert initial.attrs["value"] == 0
    locations = dict(operation_locations(decomposed))
    history = decomposed.metadata["provenance"]["passes"][-1]
    assert all(op.origins for op in locations.values())
    assert {entry["relation"] for entry in history["relations"]} == {"split"}
    outputs = [
        location for entry in history["relations"] for location in entry["outputs"]
    ]
    assert len(outputs) == len(set(outputs)) == len(locations)
    mappings = history["value_mappings"]
    assert {entry["projection"] for entry in mappings} == {
        "lane",
        "matmul_lhs",
        "matmul_rhs",
        "matmul_term",
        "matmul_prefix",
    }

    for mapping in mappings:
        target = locations[mapping["target_location"]]
        assert mapping["target_result"] in {value.name for value in target.results}

    inputs = _inputs(inner, dtype)
    expected = inputs["a"] @ inputs["b"]
    result = interpret_program(decomposed, inputs, symbols=symbols, trace=True)
    _assert_equal(result.outputs["out"], expected)
    extracts = [event for event in result.trace if event.opcode == "tensor.extract"]
    assert len(extracts) == 2 * expected.size * inner
    assert sum(event.opcode == "mem.store" for event in result.trace) == expected.size
    comparison = compare_programs(reference, decomposed, inputs, symbols=symbols)
    assert comparison.equal

    if inner:
        assert not comparison.mapping_issues

    for backend in ("cuda", "triton"):
        lowered = lower_for_target(reference, backend=backend)
        actual = interpret_program(lowered, _inputs(inner, dtype), symbols=symbols)
        _assert_equal(actual.outputs["out"], expected)


@pytest.mark.parametrize("source", ("rhs", "attribute"))
@pytest.mark.parametrize("inner", (0, 4))
def test_numeric_matmul_bound_without_lhs_shape(source, inner):
    reference = _program(str(inner), "int32")
    inputs = tuple(
        replace(value, type=replace(value.type, shape=()))
        if value.name == "a" or (source == "attribute" and value.name == "b")
        else value
        for value in reference.inputs
    )
    block = reference.blocks[0]
    operations = block.operations

    if source == "attribute":
        operations = (
            replace(operations[0], attrs={"k": str(inner)}),
            *operations[1:],
        )

    reference = replace(
        reference,
        inputs=inputs,
        blocks=(replace(block, operations=operations),),
    )
    lowered = lower_for_target(reference, backend="cuda")
    loop, definitions = _loop_and_definitions(lowered)
    bound = definitions[loop.operands[1]]
    assert bound.opcode == "arith.constant"
    assert bound.attrs["value"] == inner
    inputs = _inputs(inner, "int32")
    result = interpret_program(lowered, inputs)
    np.testing.assert_array_equal(result.outputs["out"], inputs["a"] @ inputs["b"])
