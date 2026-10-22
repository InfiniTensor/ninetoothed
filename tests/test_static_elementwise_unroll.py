"""Small recurrences retain typed carries, effects and arithmetic ordering."""

from ninetoothed.compiler.unroll import unroll_static_elementwise_loops
from ninetoothed.ir import ssa


def _program(*, lower=0, upper=4, step=1, dtype="float32"):
    tensor = ssa.Type(kind="tensor", shape=("256",), dtype=dtype)
    index = ssa.Type(kind="index")
    values = tuple(ssa.Value(name=name, type=tensor) for name in ("x", "y", "out"))
    bounds = tuple(
        ssa.Operation(
            opcode="arith.constant",
            results=(ssa.Value(name=name, type=index),),
            attrs={"value": value},
        )
        for name, value in (("lo", lower), ("hi", upper), ("step", step))
    )
    body = (
        ssa.Operation(
            opcode="arith.constant",
            results=(
                ssa.Value(name="factor", type=ssa.Type(kind="scalar", dtype="float32")),
            ),
            attrs={"value": 0.25},
        ),
        ssa.Operation(
            opcode="arith.mul",
            operands=("carry", "factor"),
            results=(ssa.Value(name="scaled", type=tensor),),
        ),
        ssa.Operation(
            opcode="arith.add",
            operands=("scaled", "y"),
            results=(ssa.Value(name="next", type=tensor),),
        ),
        ssa.Operation(opcode="scf.yield", operands=("next",)),
    )
    region = ssa.Block(
        name="loop",
        args=(ssa.Value(name="i", type=index), ssa.Value(name="carry", type=tensor)),
        operations=body,
    )
    loop = ssa.Operation(
        opcode="scf.for",
        operands=("lo", "hi", "step", "x"),
        results=(ssa.Value(name="result", type=tensor),),
        regions=(region,),
        attrs={
            "induction": "i",
            "iter_args": ({"name": "value", "initial": "x", "block_arg": "carry"},),
            "python_target": "i",
        },
    )
    store = ssa.Operation(
        opcode="mem.store",
        operands=("result", "out"),
        attrs={"target": "out", "target_shape": ("256",)},
    )

    return ssa.Program(
        kind="unnamed_recurrence",
        inputs=values,
        outputs=(values[-1],),
        blocks=(ssa.Block(operations=(*bounds, loop, store)),),
    )


def _evaluate(program, x, y):
    environment = {"x": x, "y": y}

    for operation in program.blocks[0].operations:
        if operation.opcode == "arith.constant":
            value = operation.attrs["value"]
        elif operation.opcode == "arith.mul":
            value = (
                environment[operation.operands[0]] * environment[operation.operands[1]]
            )
        elif operation.opcode == "arith.add":
            value = (
                environment[operation.operands[0]] + environment[operation.operands[1]]
            )
        elif operation.opcode == "mem.store":
            return environment[operation.operands[0]]
        else:
            raise AssertionError(operation.opcode)

        environment[operation.results[0].name] = value


def test_unroll_preserves_recurrence_order():
    iterations = 4
    program, expanded = unroll_static_elementwise_loops(_program(upper=iterations))
    ssa.verify_program(program)
    assert expanded == (("result", iterations),)
    assert [
        op.opcode
        for op in program.blocks[0].operations
        if op.opcode in {"arith.mul", "arith.add"}
    ] == ["arith.mul", "arith.add"] * iterations
    x, y = (3.0, -7.0)
    expected = 3.0

    for _ in range(iterations):
        expected = 0.25 * expected + y

    assert _evaluate(program, x, y) == expected
