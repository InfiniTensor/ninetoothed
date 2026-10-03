"""Shape-call boundaries preserve represented syntax and backend integer rules."""

import numpy as np
import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.expressions import evaluate
from ninetoothed.ir import IndexExpr, TensorLayout, TensorSpec, ssa
from ninetoothed.naming import make_next_power_of_2


@pytest.mark.parametrize(
    "source",
    (
        "int('10', base=16)",
        "int(**options)",
        "cdiv(N, y=8)",
        "maximum(a, b, out=target)",
    ),
)
def test_unrepresented_call_keywords_are_rejected_instead_of_dropped(source):
    with pytest.raises(ValueError, match="keyword"):
        IndexExpr.parse(source)


@pytest.mark.parametrize(
    "source,symbols,expected",
    (
        ("int('10', 16)", {}, 16),
        ("cdiv(N, 8)", {"N": 17}, 3),
        ("maximum(a, b)", {"a": 2, "b": 7}, 7),
    ),
)
def test_positional_calls_still_round_trip_and_evaluate(source, symbols, expected):
    expression = IndexExpr.parse(source)
    assert IndexExpr.parse(expression.render()) == expression
    assert evaluate(expression, symbols) == expected


def _backend_integer_reference(value):
    # Triton 3.1.0's public utility fills bits with these six shifts. This
    # independent reference is restricted to the signed 64-bit input range.
    value = int(value) - 1

    for shift in (1, 2, 4, 8, 16, 32):
        value |= value >> shift

    return value + 1


@pytest.mark.parametrize(
    "value",
    (
        -(2**63),
        -17,
        -2,
        -1,
        0,
        1,
        2,
        3,
        7,
        8,
        9,
        31,
        32,
        33,
        2**31 - 1,
        2**31,
        2**31 + 1,
        2**62 - 1,
        2**62,
        2**62 + 1,
        2**63 - 1,
    ),
)
@pytest.mark.parametrize("prefix", ("", "triton."))
def test_power_of_two_calls_match_backend_integer_boundaries(value, prefix):
    expression = IndexExpr.parse(prefix + "next_power_of_2(n)")
    actual = evaluate(expression, {"n": value})
    assert actual == _backend_integer_reference(value)
    assert type(actual) is int


@pytest.mark.parametrize(
    "value,expected",
    ((False, 0), (True, 1), (np.int64(0), 0), (np.int32(17), 32), (0.0, 0), (2.9, 2)),
)
def test_existing_integer_normalization_remains_compatible(value, expected):
    assert evaluate(IndexExpr.parse("next_power_of_2(n)"), {"n": value}) == expected


def _shape_program(size, named, load):
    padded = make_next_power_of_2("N") if named else "next_power_of_2(N)"
    x = ssa.Value(name="x", type=ssa.Type(kind="tensor", shape=("N",), dtype="float32"))
    result = ssa.Value(
        name="%result",
        type=ssa.Type(
            kind="tensor" if load else "scalar",
            shape=(padded,) if load else (),
            dtype="float32" if load else "int64",
        ),
    )
    layout = TensorLayout(
        source_shape=(IndexExpr.parse("N"),),
        source_strides=(IndexExpr.parse(1),),
        view_shape=(IndexExpr.parse("N"),),
        application_shape=(IndexExpr.parse(padded),),
    )
    spec = TensorSpec(
        name="x",
        ndim=1,
        shape=(padded,),
        dtype="float32",
        layout=layout,
        attrs={"source_shape": ("N",), "application_shape": (padded,)},
    )
    operation = ssa.Operation(
        opcode="mem.load" if load else "shape.dim", operands=("x",), results=(result,)
    )
    program = ssa.Program(
        kind="shape_boundary",
        inputs=(x,),
        outputs=(result,),
        blocks=(ssa.Block(operations=(operation,)),),
    )

    return program, spec, np.empty(size, dtype=np.float32)


@pytest.mark.parametrize("named", (False, True))
@pytest.mark.parametrize("size", (0, 1, 2, 3, 17))
def test_public_shape_metadata_uses_the_same_rounding_rule(size, named):
    program, spec, x = _shape_program(size, named, False)
    result = interpret_program(program, {"x": x}, tensors=(spec,), trace=True)
    assert result.outputs["%result"] == _backend_integer_reference(size)


@pytest.mark.parametrize("named", (False, True))
def test_empty_padded_view_load_stays_empty_without_a_memory_access(named):
    program, spec, x = _shape_program(0, named, True)
    result = interpret_program(program, {"x": x}, tensors=(spec,), trace=True)
    assert result.outputs["%result"].shape == (0,)
    assert result.outputs["%result"].dtype == np.float32
    assert result.trace[-1].memory == ()
