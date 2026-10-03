"""Keep checked interpreter memory forms compatible with SSA verification."""

import numpy as np
import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.ir import ssa


def _value(name, *, kind="scalar", dtype="int32"):
    return ssa.Value(
        name=name,
        type=ssa.Type(
            kind=kind,
            dtype=dtype,
            shape=("3",) if kind == "tensor" else (),
        ),
    )


def _program(inputs, operation):
    return ssa.Program(
        kind="memory_contract",
        inputs=tuple(inputs),
        outputs=operation.results,
        blocks=(ssa.Block(operations=(operation,)),),
    )


@pytest.mark.parametrize("kind", ("pointer", "tensor"))
@pytest.mark.parametrize("mask_kind", ("scalar", "tensor"))
@pytest.mark.parametrize("arity", (1, 2, 3))
def test_verified_load_executes_pointer_and_whole_view_masks(kind, mask_kind, arity):
    target = _value("target", kind=kind)
    mask = _value("mask", kind=mask_kind, dtype="bool")
    other = _value("other")
    result = _value(
        "%result",
        kind="tensor"
        if kind == "tensor" or (mask_kind == "tensor" and arity > 1)
        else "scalar",
    )
    operands = (target.name, mask.name, other.name)[:arity]
    program = _program(
        (target, mask, other),
        ssa.Operation(opcode="mem.load", operands=operands, results=(result,)),
    )
    data = np.array([2, 3, 5], dtype=np.int32)
    predicate = np.array([True, False, True]) if mask_kind == "tensor" else False
    expected = data if kind == "tensor" else np.int32(2)

    if arity > 1:
        expected = np.where(predicate, expected, -19 if arity == 3 else 0)

    assert ssa.verify_program(program) is program
    actual = interpret_program(
        program, {"target": data, "mask": predicate, "other": np.int32(-19)}
    )
    np.testing.assert_array_equal(actual.outputs[result.name], expected)
    np.testing.assert_array_equal(data, [2, 3, 5])


@pytest.mark.parametrize("kind", ("pointer", "tensor"))
@pytest.mark.parametrize("predicate", (False, True))
def test_verified_masked_store_changes_only_active_checked_destination(kind, predicate):
    target = _value("target", kind=kind)
    value, mask = _value("value"), _value("mask", dtype="bool")
    program = _program(
        (target, value, mask),
        ssa.Operation(
            opcode="mem.store", operands=(value.name, target.name, mask.name)
        ),
    )
    data = np.array([2, 3, 5], dtype=np.int32)
    assert ssa.verify_program(program) is program
    interpret_program(
        program,
        {"target": data, "value": np.int32(7), "mask": predicate},
    )
    expected = [7, 3, 5] if kind == "pointer" else [7, 7, 7]
    np.testing.assert_array_equal(data, expected if predicate else [2, 3, 5])


@pytest.mark.parametrize("opcode", ("mem.load", "mem.store"))
@pytest.mark.parametrize(
    "kind,dtype",
    (
        ("scalar", "int32"),
        ("tensor", "float32"),
        ("scalar", None),
        ("pointer", "bool"),
        ("index", "bool"),
    ),
)
def test_memory_mask_operand_requires_known_boolean_numeric_type(opcode, kind, dtype):
    target = _value("target", kind="pointer")
    value, result = _value("value"), _value("%result")
    mask = _value("mask", kind=kind, dtype=dtype)
    operands = (
        (target.name, mask.name)
        if opcode == "mem.load"
        else (value.name, target.name, mask.name)
    )
    program = _program(
        (target, value, mask),
        ssa.Operation(
            opcode=opcode,
            operands=operands,
            results=(result,) if opcode == "mem.load" else (),
        ),
    )

    with pytest.raises(ssa.VerificationError, match="requires a boolean mask"):
        ssa.verify_program(program)


@pytest.mark.parametrize(
    "opcode,operands,result_count",
    (
        ("mem.load", (), 1),
        ("mem.load", ("target", "mask", "value", "value"), 1),
        ("mem.load", ("target",), 0),
        ("mem.load", ("target",), 2),
        ("mem.store", ("value",), 0),
        ("mem.store", ("value", "target", "mask", "value"), 0),
        ("mem.store", ("value", "target"), 1),
        ("mem.data_ptr", ("target", "mask"), 1),
        ("mem.data_ptr", ("target",), 0),
        ("mem.atomic_add", ("target", "value", "mask"), 1),
        ("mem.atomic_add", ("target", "value"), 0),
    ),
)
def test_memory_contract_rejects_extra_operands_and_wrong_result_count(
    opcode, operands, result_count
):
    target = _value("target", kind="pointer")
    value, mask = _value("value"), _value("mask", dtype="bool")
    results = tuple(_value(f"%result{index}") for index in range(result_count))
    program = _program(
        (target, value, mask),
        ssa.Operation(opcode=opcode, operands=operands, results=results),
    )

    with pytest.raises(ssa.VerificationError, match="requires.*operands.*results"):
        ssa.verify_program(program)


@pytest.mark.parametrize(
    "opcode,kind",
    (
        ("mem.load", "scalar"),
        ("mem.store", "index"),
        ("mem.data_ptr", "pointer"),
        ("mem.atomic_add", "tensor"),
    ),
)
def test_memory_contract_keeps_invalid_target_kinds_rejected(opcode, kind):
    target, value = _value("target", kind=kind), _value("value")
    result = _value("%result", kind="pointer" if opcode == "mem.data_ptr" else "scalar")
    program = _program(
        (target, value),
        ssa.Operation(
            opcode=opcode,
            operands=(value.name, target.name)
            if opcode == "mem.store"
            else (target.name, value.name)
            if opcode == "mem.atomic_add"
            else (target.name,),
            results=() if opcode == "mem.store" else (result,),
        ),
    )

    with pytest.raises(ssa.VerificationError, match="Invalid memory target"):
        ssa.verify_program(program)


def test_data_ptr_must_still_produce_a_pointer():
    target, result = _value("target", kind="tensor"), _value("%result")
    program = _program(
        (target,),
        ssa.Operation(
            opcode="mem.data_ptr", operands=(target.name,), results=(result,)
        ),
    )

    with pytest.raises(ssa.VerificationError, match="must produce a pointer"):
        ssa.verify_program(program)
