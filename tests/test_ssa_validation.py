from dataclasses import replace

import pytest

from ninetoothed.frontend.python import LoweringError, from_source
from ninetoothed.ir import TensorSpec, ssa


def _tensor(name, shape, dtype="float32"):
    return TensorSpec(ndim=len(shape), shape=tuple(shape), dtype=dtype, name=name)


@pytest.mark.parametrize("indexed_store", (False, True))
def test_verifier_rejects_undefined_operand(indexed_store):
    value = ssa.Value(name="x", type=ssa.Type(kind="tensor", dtype="float32"))
    operation = (
        ssa.Operation(
            opcode="mem.store", operands=("x", "x"), attrs={"indices": ("missing",)}
        )
        if indexed_store
        else ssa.Operation(opcode="arith.add", operands=("missing",))
    )
    program = ssa.Program(
        kind="invalid",
        inputs=(value,),
        blocks=(ssa.Block(operations=(operation,)),),
    )

    with pytest.raises(ssa.VerificationError, match="undefined values: missing"):
        ssa.verify_program(program)


def test_verifier_rejects_duplicate_result_definition():
    value = ssa.Value(name="%0", type=ssa.Type(kind="scalar", dtype="float32"))
    program = ssa.Program(
        kind="invalid",
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(opcode="arith.constant", results=(value,)),
                    ssa.Operation(opcode="arith.constant", results=(value,)),
                )
            ),
        ),
    )

    with pytest.raises(ssa.VerificationError, match="Duplicate SSA definition"):
        ssa.verify_program(program)


def test_verifier_accepts_scalar_string_store_index():
    value = ssa.Value(name="x", type=ssa.Type(kind="tensor", dtype="float32"))
    index = ssa.Value(name="%i", type=ssa.Type(kind="index"))
    store = ssa.Operation(
        opcode="mem.store", operands=("x", "x"), attrs={"indices": "%i"}
    )
    program = ssa.Program(
        kind="indexed_store",
        inputs=(value, index),
        blocks=(ssa.Block(operations=(store,)),),
    )
    assert ssa.verify_program(program) is program


def test_unknown_python_helper_fails_closed_with_source_location():
    source = """
def application(x, out):
    out = unavailable_helper(x)
"""

    with pytest.raises(LoweringError, match=r"unavailable_helper.*line 3"):
        from_source(source, (_tensor("x", ("n",)), _tensor("out", ("n",))))


def test_strict_frontend_rejects_invalid_reduction_axis():
    source = """
def application(x, out):
    out = sum(x, axis=1)
"""

    with pytest.raises(LoweringError, match="outside tensor rank"):
        from_source(
            source,
            (_tensor("x", ("n",)), _tensor("out", ("n",))),
            strict=True,
        )


def test_static_broadcast_mismatch_fails_closed():
    source = """
def application(x, y, out):
    out = x + y
"""

    with pytest.raises(LoweringError, match="Cannot broadcast dimensions"):
        from_source(
            source,
            (
                _tensor("x", ("3",)),
                _tensor("y", ("4",)),
                _tensor("out", ("4",)),
            ),
            strict=True,
        )


def test_cast_updates_ssa_result_dtype():
    program = from_source(
        """
def application(x, out):
    out = x.to(float16)
""",
        (_tensor("x", ("n",)), _tensor("out", ("n",), "float16")),
    )
    cast = next(
        operation
        for operation in program.blocks[0].operations
        if operation.opcode == "tensor.cast"
    )
    assert cast.results[0].type.dtype == "float16"


def test_batched_matmul_type_preserves_broadcast_batch_domain():
    program = from_source(
        """
def application(a, b, out):
    out = a @ b
""",
        (
            _tensor("a", ("batch", "m", "k")),
            _tensor("b", ("1", "k", "n")),
            _tensor("out", ("batch", "m", "n")),
        ),
        strict=True,
    )
    matmul = next(
        operation
        for operation in program.blocks[0].operations
        if operation.opcode == "linalg.matmul"
    )
    assert matmul.results[0].type.shape == ("batch", "m", "n")


def _loop_program(
    *, initial_dtype="float32", argument_dtype="float32", result_dtype="float32"
):
    bound = ssa.Value(name="bound", type=ssa.Type(kind="index"))
    initial = ssa.Value(
        name="initial", type=ssa.Type(kind="scalar", dtype=initial_dtype)
    )
    argument = replace(
        initial, name="%acc", type=replace(initial.type, dtype=argument_dtype)
    )
    result = replace(
        initial, name="%result", type=replace(initial.type, dtype=result_dtype)
    )
    body = ssa.Block(
        args=(replace(bound, name="%iv"), argument),
        operations=(ssa.Operation(opcode="scf.yield", operands=(argument.name,)),),
    )
    loop = ssa.Operation(
        opcode="scf.for",
        operands=("bound", "bound", "bound", "initial"),
        results=(result,),
        attrs={
            "induction": "%iv",
            "iter_args": ({"initial": "initial", "block_arg": "%acc"},),
        },
        regions=(body,),
    )

    return ssa.Program(
        kind="loop",
        inputs=(bound, initial),
        outputs=(result,),
        blocks=(ssa.Block(operations=(loop,)),),
    )


@pytest.mark.parametrize(
    "dtypes, valid",
    (
        ((None, "float32", "float32"), True),
        (("float32", None, "int32"), False),
    ),
)
def test_verifier_checks_partially_known_loop_types(dtypes, valid):
    program = _loop_program(
        initial_dtype=dtypes[0], argument_dtype=dtypes[1], result_dtype=dtypes[2]
    )

    if valid:
        assert ssa.verify_program(program) is program
    else:
        with pytest.raises(ssa.VerificationError, match="Type mismatch"):
            ssa.verify_program(program)


@pytest.mark.parametrize(
    "type_, valid",
    (
        (ssa.Type(kind="scalar", dtype="index"), True),
        (ssa.Type(kind="scalar", dtype="float32"), False),
        (ssa.Type(kind="tensor", shape=("4",), dtype="index"), False),
    ),
)
def test_verifier_checks_loop_bound_types(type_, valid):
    lower = ssa.Value(name="lower", type=ssa.Type(kind="scalar"))
    bound = ssa.Value(name="bound", type=type_)
    step = ssa.Value(name="step", type=ssa.Type(kind="index"))
    loop = ssa.Operation(
        opcode="scf.for",
        operands=(lower.name, bound.name, step.name),
        regions=(
            ssa.Block(
                args=(ssa.Value(name="%iv", type=ssa.Type(kind="index")),),
                operations=(ssa.Operation(opcode="scf.yield"),),
            ),
        ),
    )
    program = ssa.Program(
        kind="loop",
        inputs=(lower, bound, step),
        blocks=(ssa.Block(operations=(loop,)),),
    )

    if valid:
        assert ssa.verify_program(program) is program
    else:
        with pytest.raises(ssa.VerificationError, match="integer scalar bounds"):
            ssa.verify_program(program)


@pytest.mark.parametrize(
    "fault, message",
    (
        ("parent_result", "undefined values"),
        ("escaped_value", "Undefined SSA output"),
        ("early_yield", "must terminate"),
        ("bounds", "three bounds"),
        ("yield_type", "Type mismatch"),
        ("bindings", "loop-carried bindings"),
        ("induction", "inconsistent induction"),
    ),
)
def test_verifier_checks_region_contracts(fault, message):
    program = _loop_program()
    loop = program.blocks[0].operations[0]
    body = loop.regions[0]
    yielded = body.operations[-1]

    if fault == "parent_result":
        body = replace(
            body, operations=(replace(yielded, operands=(loop.results[0].name,)),)
        )
    elif fault == "escaped_value":
        program = replace(program, outputs=(body.args[1],))
    elif fault == "early_yield":
        body = replace(body, operations=(yielded, yielded))
    elif fault == "bounds":
        loop = replace(loop, operands=loop.operands[1:])
    elif fault == "yield_type":
        body = replace(
            body, operations=(replace(yielded, operands=(body.args[0].name,)),)
        )
    elif fault == "bindings":
        loop = replace(loop, attrs={"induction": "%iv"})
    elif fault == "induction":
        loop = replace(loop, attrs={"iter_args": loop.attrs["iter_args"]})
        body = replace(body, args=(replace(body.args[0], name="%i"), body.args[1]))

    loop = replace(loop, regions=(body,))
    program = replace(program, blocks=(ssa.Block(operations=(loop,)),))

    with pytest.raises(ssa.VerificationError, match=message):
        ssa.verify_program(program)


def test_verifier_rejects_if_without_regions():
    condition = ssa.Value(name="condition", type=ssa.Type(kind="scalar", dtype="bool"))
    operation = ssa.Operation(opcode="scf.if", operands=(condition.name,))
    program = ssa.Program(
        kind="if", inputs=(condition,), blocks=(ssa.Block(operations=(operation,)),)
    )

    with pytest.raises(ssa.VerificationError, match="one or two regions"):
        ssa.verify_program(program)


@pytest.mark.parametrize(
    "duplicate, message", ((False, "Type mismatch"), (True, "Duplicate SSA output"))
)
def test_verifier_checks_output_declarations(duplicate, message):
    output = ssa.Value(name="x", type=ssa.Type(kind="scalar", dtype="float32"))
    outputs = (
        (output, output)
        if duplicate
        else (replace(output, type=replace(output.type, dtype="int32")),)
    )
    program = ssa.Program(
        kind="outputs", inputs=(output,), outputs=outputs, blocks=(ssa.Block(),)
    )

    with pytest.raises(ssa.VerificationError, match=message):
        ssa.verify_program(program)


@pytest.mark.parametrize(
    "opcode, operands, message",
    (
        ("mem.store", ("x",), "requires 2 operands"),
        ("mem.load", ("x",), "Invalid memory target"),
    ),
)
def test_verifier_checks_memory_contracts(opcode, operands, message):
    value = ssa.Value(name="x", type=ssa.Type(kind="tensor", dtype="float32"))
    result = replace(value, name="%loaded")
    operation = ssa.Operation(
        opcode=opcode,
        operands=operands,
        results=(result,) if opcode == "mem.load" else (),
    )
    program = ssa.Program(
        kind="invalid", inputs=(value,), blocks=(ssa.Block(operations=(operation,)),)
    )

    with pytest.raises(ssa.VerificationError, match=message):
        ssa.verify_program(program)
