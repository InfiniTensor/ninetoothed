"""Independent, hand-written SSA checks for CPU execution semantics."""

import numpy as np
import pytest

from ninetoothed.interpreter import (
    InterpretationError,
    UnsupportedOperationError,
    interpret_program,
)
from ninetoothed.ir import ssa


def _value(name, shape=(), dtype="float32", kind=None):
    return ssa.Value(
        name=name,
        type=ssa.Type(
            kind=kind or ("tensor" if shape else "scalar"),
            shape=tuple(map(str, shape)),
            dtype=dtype,
        ),
    )


def _constant(value, literal):
    return ssa.Operation(
        opcode="arith.constant", results=(value,), attrs={"value": literal}
    )


def _program(inputs, outputs, *operations):
    return ssa.verify_program(
        ssa.Program(
            kind="cpu_semantics",
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            blocks=(ssa.Block(operations=tuple(operations)),),
        )
    )


@pytest.mark.parametrize(
    "opcode,lhs,rhs,expected",
    (
        ("arith.add", [3, -7, 0], [4, 3, -5], [7, -4, -5]),
        ("arith.sub", [3, -7, 0], [4, 3, -5], [-1, -10, 5]),
        ("arith.mul", [3, -7, 0], [4, 3, -5], [12, -21, 0]),
        ("arith.floordiv", [7, -7, 0], [3, 3, 2], [2, -3, 0]),
        ("arith.bitwise_and", [3, 7, 0], [6, 3, 2], [2, 3, 0]),
        ("arith.bitwise_xor", [3, 7, 0], [6, 3, 2], [5, 4, 2]),
    ),
)
def test_integer_binary_operations_are_exact(opcode, lhs, rhs, expected):
    a, b, result = (_value(name, (3,), "int32") for name in ("a", "b", "%result"))
    program = _program(
        (a, b),
        (result,),
        ssa.Operation(opcode=opcode, operands=(a.name, b.name), results=(result,)),
    )
    actual = interpret_program(
        program,
        {"a": np.array(lhs, dtype=np.int32), "b": np.array(rhs, dtype=np.int32)},
    ).outputs[result.name]
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.int32))


def _masked_load_program(offsets, mask, *, source_size=3):
    x = _value("x", (source_size,))
    ptr = _value("%ptr", kind="pointer")
    indices = _value("%indices", (len(offsets),), "int32")
    addresses = _value("%addresses", (len(offsets),), kind="pointer")
    predicate = _value("%mask", (len(mask),), "bool")
    result = _value("%loaded", (len(offsets),))
    program = _program(
        (x,),
        (result,),
        ssa.Operation(opcode="mem.data_ptr", operands=("x",), results=(ptr,)),
        _constant(indices, tuple(offsets)),
        ssa.Operation(
            opcode="arith.add", operands=(ptr.name, indices.name), results=(addresses,)
        ),
        _constant(predicate, tuple(mask)),
        ssa.Operation(
            opcode="mem.load",
            operands=(addresses.name, predicate.name),
            results=(result,),
            attrs={"other": -99},
        ),
    )

    return program


def test_masked_load_never_dereferences_inactive_out_of_bounds_lanes():
    program = _masked_load_program(
        (-100, 0, 2, 3, 999), (False, True, True, False, False)
    )
    x = np.array([5, 7, 11], dtype=np.float32)
    result = interpret_program(program, {"x": x})
    np.testing.assert_array_equal(
        result.outputs["%loaded"], np.array([-99, 5, 11, -99, -99], dtype=np.float32)
    )
    np.testing.assert_array_equal(x, np.array([5, 7, 11], dtype=np.float32))


def test_all_false_mask_can_load_from_an_empty_buffer():
    program = _masked_load_program((-1, 0, 1000), (False, False, False), source_size=0)
    result = interpret_program(program, {"x": np.empty(0, dtype=np.float32)})
    np.testing.assert_array_equal(
        result.outputs["%loaded"], np.full(3, -99, dtype=np.float32)
    )


@pytest.mark.parametrize("offset", (-1, 3, 1000))
def test_active_out_of_bounds_load_reports_the_operation_location(offset):
    program = _masked_load_program((offset,), (True,))

    with pytest.raises(InterpretationError) as caught:
        interpret_program(program, {"x": np.ones(3, dtype=np.float32)})

    assert "mem.load" in str(caught.value)
    assert "entry:4" in str(caught.value)


def _masked_store_program(offsets, mask):
    out = _value("out", (5,), "int32")
    values = _value("values", (len(offsets),), "int32")
    ptr = _value("%ptr", dtype="int32", kind="pointer")
    indices = _value("%indices", (len(offsets),), "int32")
    addresses = _value("%addresses", (len(offsets),), "int32", kind="pointer")
    predicate = _value("%mask", (len(mask),), "bool")

    return _program(
        (out, values),
        (out,),
        ssa.Operation(opcode="mem.data_ptr", operands=("out",), results=(ptr,)),
        _constant(indices, tuple(offsets)),
        ssa.Operation(
            opcode="arith.add",
            operands=(ptr.name, indices.name),
            results=(addresses,),
        ),
        _constant(predicate, tuple(mask)),
        ssa.Operation(
            opcode="mem.store", operands=("values", addresses.name, predicate.name)
        ),
    )


def test_masked_store_changes_only_active_valid_addresses():
    program = _masked_store_program(
        (-100, 1, 3, 5, 999), (False, True, True, False, False)
    )
    out = np.full(5, -731, dtype=np.int32)
    values = np.array([2, 3, 5, 7, 11], dtype=np.int32)
    result = interpret_program(program, {"out": out, "values": values})
    np.testing.assert_array_equal(
        out, np.array([-731, 3, -731, 5, -731], dtype=np.int32)
    )
    np.testing.assert_array_equal(result.outputs["out"], out)


@pytest.mark.parametrize("offset", (-1, 5, 1000))
def test_active_out_of_bounds_store_reports_the_operation_location(offset):
    program = _masked_store_program((offset,), (True,))
    out = np.full(5, -731, dtype=np.int32)

    with pytest.raises(InterpretationError) as caught:
        interpret_program(
            program, {"out": out, "values": np.array([3], dtype=np.int32)}
        )

    assert "mem.store" in str(caught.value)
    assert "entry:4" in str(caught.value)
    np.testing.assert_array_equal(out, np.full(5, -731, dtype=np.int32))


def test_unknown_operation_fails_closed_with_opcode_and_location():
    program = _program((), (), ssa.Operation(opcode="test.unsupported"))

    with pytest.raises(UnsupportedOperationError) as caught:
        interpret_program(program, {})

    assert "test.unsupported" in str(caught.value)
    assert "entry:0" in str(caught.value)


def test_unknown_operation_in_nested_region_reports_nested_location():
    condition = _value("condition", dtype="bool")
    program = _program(
        (condition,),
        (),
        ssa.Operation(
            opcode="scf.if",
            operands=(condition.name,),
            regions=(
                ssa.Block(
                    name="then", operations=(ssa.Operation(opcode="test.nested"),)
                ),
            ),
        ),
    )

    with pytest.raises(UnsupportedOperationError) as caught:
        interpret_program(program, {"condition": True})

    assert "test.nested" in str(caught.value)
    assert "entry:0" in str(caught.value)
    assert "region" in str(caught.value) or "then" in str(caught.value)


def test_trace_is_reproducible_and_snapshots_do_not_alias_inputs():
    x, intermediate, result = (_value(name, (3,)) for name in ("x", "%square", "%sum"))
    program = _program(
        (x,),
        (result,),
        ssa.Operation(opcode="arith.mul", operands=("x", "x"), results=(intermediate,)),
        ssa.Operation(
            opcode="arith.add", operands=("x", intermediate.name), results=(result,)
        ),
    )
    data = np.array([1, -2, 3], dtype=np.float32)
    first = interpret_program(program, {"x": data}, trace=True)
    second = interpret_program(program, {"x": data.copy()}, trace=True)
    assert first.trace == second.trace
    assert [event.opcode for event in first.trace] == ["arith.mul", "arith.add"]
    assert all(event.program_id == (0, 0, 0) for event in first.trace)
    assert first.trace[0].location.startswith("entry:0")
    assert "%square" in first.trace[0].results
    frozen_trace = repr(first.trace)
    data[:] = 100
    first.outputs["%sum"][:] = -1
    assert repr(first.trace) == frozen_trace


def test_trace_opcode_filter_keeps_execution_unchanged():
    x, square, result = (_value(name, (2,)) for name in ("x", "%square", "%sum"))
    program = _program(
        (x,),
        (result,),
        ssa.Operation(opcode="arith.mul", operands=("x", "x"), results=(square,)),
        ssa.Operation(
            opcode="arith.add", operands=("x", square.name), results=(result,)
        ),
    )
    result = interpret_program(
        program,
        {"x": np.array([2, 3], dtype=np.float32)},
        trace=True,
        opcodes=("arith.add",),
    )
    assert [event.opcode for event in result.trace] == ["arith.add"]
    np.testing.assert_array_equal(
        result.outputs["%sum"], np.array([6, 12], dtype=np.float32)
    )


@pytest.mark.parametrize("lower,upper,step,expected", ((5, -1, -2, 9), (2, 2, 1, 0)))
def test_handwritten_loop_handles_negative_step_and_zero_iterations(
    lower, upper, step, expected
):
    lo, hi, increment, zero, induction, accumulator, updated, result = (
        _value(name, dtype="int32")
        for name in (
            "%lo",
            "%hi",
            "%step",
            "%zero",
            "%i",
            "%acc",
            "%updated",
            "%result",
        )
    )
    body = ssa.Block(
        name="loop",
        args=(induction, accumulator),
        operations=(
            ssa.Operation(
                opcode="arith.add",
                operands=(accumulator.name, induction.name),
                results=(updated,),
            ),
            ssa.Operation(opcode="scf.yield", operands=(updated.name,)),
        ),
    )
    program = _program(
        (),
        (result,),
        _constant(lo, lower),
        _constant(hi, upper),
        _constant(increment, step),
        _constant(zero, 0),
        ssa.Operation(
            opcode="scf.for",
            operands=(lo.name, hi.name, increment.name, zero.name),
            results=(result,),
            attrs={
                "induction": induction.name,
                "iter_args": (
                    {
                        "name": "acc",
                        "initial": zero.name,
                        "block_arg": accumulator.name,
                    },
                ),
            },
            regions=(body,),
        ),
    )
    actual = interpret_program(program, {}, trace=True).outputs[result.name]
    np.testing.assert_array_equal(actual, np.int32(expected))


def _grid_program():
    out = _value("out", (12,), "int32")
    values = {
        name: _value(name, dtype="int32")
        for name in (
            "%pid0",
            "%pid1",
            "%pid2",
            "%six",
            "%two",
            "%hundred",
            "%ten",
            "%offset0",
            "%offset1",
            "%offset01",
            "%offset",
            "%value0",
            "%value1",
            "%value01",
            "%value",
        )
    }
    ptr = _value("%ptr", dtype="int32", kind="pointer")
    address = _value("%address", dtype="int32", kind="pointer")
    operations = [
        ssa.Operation(
            opcode="index.program_id",
            results=(values[f"%pid{axis}"],),
            attrs={"axis": axis},
        )
        for axis in range(3)
    ]
    operations.extend(
        _constant(values[name], literal)
        for name, literal in (("%six", 6), ("%two", 2), ("%hundred", 100), ("%ten", 10))
    )

    for opcode, lhs, rhs, result in (
        ("mul", "%pid0", "%six", "%offset0"),
        ("mul", "%pid1", "%two", "%offset1"),
        ("add", "%offset0", "%offset1", "%offset01"),
        ("add", "%offset01", "%pid2", "%offset"),
        ("mul", "%pid0", "%hundred", "%value0"),
        ("mul", "%pid1", "%ten", "%value1"),
        ("add", "%value0", "%value1", "%value01"),
        ("add", "%value01", "%pid2", "%value"),
    ):
        operations.append(
            ssa.Operation(
                opcode=f"arith.{opcode}", operands=(lhs, rhs), results=(values[result],)
            )
        )

    operations.extend(
        (
            ssa.Operation(opcode="mem.data_ptr", operands=("out",), results=(ptr,)),
            ssa.Operation(
                opcode="arith.add", operands=(ptr.name, "%offset"), results=(address,)
            ),
            ssa.Operation(opcode="mem.store", operands=("%value", address.name)),
        )
    )

    return _program((out,), (out,), *operations)


def test_three_dimensional_grid_trace_filter_and_watch_do_not_filter_execution():
    program = _grid_program()
    out = np.full(12, -1, dtype=np.int32)
    callbacks = []
    result = interpret_program(
        program,
        {"out": out},
        grid=(2, 3, 2),
        trace=True,
        program_ids=((1, 2, 1),),
        opcodes=("mem.store",),
        watch=("%pid0", "%pid1", "%pid2", "%value"),
        callback=callbacks.append,
    )
    expected = np.array(
        [0, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121], dtype=np.int32
    )
    np.testing.assert_array_equal(out, expected)
    assert callbacks == list(result.trace)
    assert len(result.trace) == 1
    event = result.trace[0]
    assert event.program_id == (1, 2, 1)
    assert event.opcode == "mem.store"
    assert {name: snapshot["value"] for name, snapshot in event.watched.items()} == {
        "%pid0": 1,
        "%pid1": 2,
        "%pid2": 1,
        "%value": 121,
    }


def test_step_callback_works_without_retaining_a_trace():
    program = _grid_program()
    out = np.full(12, -1, dtype=np.int32)
    callbacks = []
    result = interpret_program(
        program,
        {"out": out},
        grid=(2, 3, 2),
        opcodes=("mem.store",),
        watch=("%value",),
        callback=callbacks.append,
    )
    assert result.trace == ()
    assert len(callbacks) == 12
    assert [event.program_id for event in callbacks] == [
        (i, j, k) for i in range(2) for j in range(3) for k in range(2)
    ]
    assert [event.watched["%value"]["value"] for event in callbacks] == out.tolist()


def test_explicit_extension_handler_receives_materialized_cpu_operands():
    x = _value("x", (3,))
    result = _value("%scaled", (3,))
    program = _program(
        (x,),
        (result,),
        ssa.Operation(
            opcode="example.scale",
            operands=(x.name,),
            results=(result,),
            attrs={"scale": 3},
        ),
    )
    calls = []

    def scale(operation, operands):
        assert len(operands) == 1
        assert isinstance(operands[0], np.ndarray)
        assert operands[0].dtype == np.float32
        calls.append(operation.opcode)

        return operands[0] * operation.attrs["scale"]

    data = np.array([2, -4, 5], dtype=np.float32)
    result = interpret_program(
        program, {"x": data}, handlers={"example.scale": scale}, trace=True
    )
    assert calls == ["example.scale"]
    np.testing.assert_allclose(
        result.outputs["%scaled"],
        np.array([6, -12, 15], dtype=np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    assert result.trace[0].opcode == "example.scale"


def test_extension_handler_does_not_hide_an_unregistered_operation():
    program = _program((), (), ssa.Operation(opcode="example.unregistered"))

    with pytest.raises(UnsupportedOperationError, match="example.unregistered"):
        interpret_program(
            program, {}, handlers={"example.other": lambda op, args: None}
        )


def test_store_trace_captures_source_before_an_aliasing_write():
    source = _value("source", (3,))
    out = _value("out", (3,))
    program = _program(
        (source, out),
        (out,),
        ssa.Operation(opcode="mem.store", operands=(source.name, out.name)),
    )
    data = np.array([1, 2, 3], dtype=np.float32)
    result = interpret_program(program, {"source": data[::-1], "out": data}, trace=True)
    np.testing.assert_array_equal(data, np.array([3, 2, 1], dtype=np.float32))
    event = result.trace[0]
    # Reading source after the store would incorrectly capture [1, 2, 3].
    assert event.inputs["source"]["value"] == [3.0, 2.0, 1.0]
    assert event.inputs["source"]["dtype"] == "float32"
    assert event.inputs["out"]["shape"] == [3]
    assert event.mask["value"] == [True, True, True]


def test_masked_invalid_addresses_are_safe_even_with_trace_enabled():
    program = _masked_load_program(
        (-100, 0, 2, 3, 999), (False, True, True, False, False)
    )
    result = interpret_program(
        program, {"x": np.array([5, 7, 11], dtype=np.float32)}, trace=True
    )
    event = next(event for event in result.trace if event.opcode == "mem.load")
    assert event.mask == {
        "dtype": "bool",
        "shape": [5],
        "value": [False, True, True, False, False],
    }
    assert event.inputs["%addresses"]["pointer_offsets"] == [-100, 0, 2, 3, 999]
    np.testing.assert_array_equal(
        result.outputs["%loaded"], np.array([-99, 5, 11, -99, -99], dtype=np.float32)
    )


def test_masked_store_trace_does_not_dereference_inactive_addresses():
    program = _masked_store_program(
        (-100, 1, 3, 5, 999), (False, True, True, False, False)
    )
    out = np.full(5, -731, dtype=np.int32)
    result = interpret_program(
        program,
        {"out": out, "values": np.array([2, 3, 5, 7, 11], dtype=np.int32)},
        trace=True,
    )
    event = next(event for event in result.trace if event.opcode == "mem.store")
    assert event.mask["value"] == [False, True, True, False, False]
    assert event.inputs["values"]["value"] == [2, 3, 5, 7, 11]
    assert event.inputs["%addresses"]["pointer_offsets"] == [-100, 1, 3, 5, 999]
    np.testing.assert_array_equal(
        out, np.array([-731, 3, -731, 5, -731], dtype=np.int32)
    )


@pytest.mark.parametrize("source_size", (0, 3))
@pytest.mark.parametrize("use_callback", (False, True))
def test_masked_tensor_store_trace_and_watch_never_load_inactive_lanes(
    source_size, use_callback
):
    from ninetoothed.ir import AccessMap, IndexExpr, TensorLayout, TensorSpec

    expr = IndexExpr.parse
    out, values = _value("out", (source_size,)), _value("values", (5,))
    mask = _value("mask", (5,), "bool")
    program = _program(
        (out, values, mask),
        (out,),
        ssa.Operation(opcode="mem.store", operands=("values", "out", "mask")),
    )
    layout = TensorLayout(
        source_shape=(expr(source_size),),
        source_strides=(expr(1),),
        view_shape=(expr(5),),
        application_shape=(expr(5),),
        view_access=AccessMap(
            source_indices=(expr("index"),),
            linear_index=expr("index"),
            predicate=expr(True),
        ),
    )
    spec = TensorSpec(
        ndim=1,
        name="out",
        shape=("5",),
        dtype="float32",
        layout=layout,
        attrs={"source_shape": (str(source_size),)},
    )
    inputs = {
        "out": np.full(source_size, -731, dtype=np.float32),
        "values": np.arange(5, dtype=np.float32),
        "mask": np.arange(5) < source_size,
    }
    callbacks = []
    result = interpret_program(
        program,
        inputs,
        tensors=(spec,),
        trace=not use_callback,
        callback=callbacks.append if use_callback else None,
        watch=("out",),
    )
    np.testing.assert_array_equal(
        result.outputs["out"], np.arange(source_size, dtype=np.float32)
    )
    event = callbacks[0] if use_callback else result.trace[0]
    assert event.mask["value"] == inputs["mask"].tolist()
    assert event.results["out"]["tensor"] == "out"
    assert "value" not in event.results["out"]
    assert event.watched["out"] == event.results["out"]


@pytest.mark.parametrize("dtype", (object, np.complex64, "U4", "S4"))
def test_unsupported_array_dtypes_are_rejected_before_execution(dtype):
    x = _value("x", (3,), dtype=None)
    program = _program((x,), (x,))
    data = np.array([1, 2, 3], dtype=dtype)

    with pytest.raises((TypeError, InterpretationError), match="(?i)dtype|type"):
        interpret_program(program, {"x": data})


def test_array_dtype_must_match_the_declared_ssa_type():
    x = _value("x", (3,), "int32")
    program = _program((x,), (x,))

    with pytest.raises((TypeError, InterpretationError), match="(?i)dtype|type"):
        interpret_program(program, {"x": np.ones(3, dtype=np.float32)})


@pytest.mark.parametrize("axis", (-1, 3, 999))
def test_num_programs_rejects_invalid_axis_with_operation_location(axis):
    count = _value("%count", dtype="int32")
    program = _program(
        (),
        (count,),
        ssa.Operation(
            opcode="call.num_programs", results=(count,), attrs={"axis": axis}
        ),
    )

    with pytest.raises(InterpretationError) as caught:
        interpret_program(program, {}, grid=(2, 3, 2))

    assert "call.num_programs" in str(caught.value)
    assert "entry:0" in str(caught.value)


@pytest.mark.parametrize("decomposition", ("matmul", "transpose"))
def test_isolated_decomposed_offset_without_output_store_is_rejected(
    decomposition,
):
    x = _value("x", (2, 2))
    index = _value("%index", dtype="int64", kind="index")
    program = _program(
        (x,),
        (index,),
        ssa.Operation(
            opcode="index.offset",
            operands=(x.name,),
            results=(index,),
            attrs={"dim": 0, "decomposition": decomposition},
        ),
    )

    with pytest.raises(UnsupportedOperationError) as caught:
        interpret_program(program, {"x": np.ones((2, 2), dtype=np.float32)})

    assert decomposition in str(caught.value)
    assert "entry:0:index.offset" in str(caught.value)
