"""Tests for the generic common-subexpression elimination pass."""

from dataclasses import replace

import pytest

from ninetoothed.compiler.cse import eliminate_common_subexpressions
from ninetoothed.compiler.passes import default_spec
from ninetoothed.ir import ssa


def _value(name, shape=(), dtype="float32", kind="tensor"):
    return ssa.Value(name=name, type=ssa.Type(kind=kind, shape=shape, dtype=dtype))


def _program(operations, outputs=()):
    block = ssa.Block(
        name="entry",
        args=(),
        operations=tuple(operations),
    )
    return ssa.Program(
        kind="test",
        inputs=(_value("x", ("n",)),),
        outputs=tuple(outputs),
        blocks=(block,),
    )


def test_removes_duplicate_pure_operations():
    program = _program(
        [
            ssa.Operation(
                opcode="math.exp",
                operands=("x",),
                results=(_value("%1", ("n",)),),
                attrs={"callee": "ntl.exp"},
            ),
            ssa.Operation(
                opcode="math.exp",
                operands=("x",),
                results=(_value("%2", ("n",)),),
                attrs={"callee": "ntl.exp"},
            ),
            ssa.Operation(
                opcode="mem.store",
                operands=("%2", "x"),
                attrs={"target": "x"},
            ),
        ]
    )

    result = ssa.verify_program(eliminate_common_subexpressions(program))

    ops = result.blocks[0].operations
    assert len(ops) == 2
    assert ops[0].results[0].name == "%1"
    assert ops[1].operands[0] == "%1"


def test_attrs_and_result_types_participate_in_key():
    program = _program(
        [
            ssa.Operation(
                opcode="tensor.cast",
                operands=("x",),
                results=(_value("%1", ("n",), dtype="float32"),),
                attrs={"dtype": "float32"},
            ),
            ssa.Operation(
                opcode="tensor.cast",
                operands=("x",),
                results=(_value("%2", ("n",), dtype="float16"),),
                attrs={"dtype": "float16"},
            ),
            ssa.Operation(
                opcode="mem.store",
                operands=("%1", "x"),
                attrs={"target": "x"},
            ),
        ]
    )

    result = eliminate_common_subexpressions(program)

    assert len(result.blocks[0].operations) == 3


def test_renames_uses_and_outputs():
    program = _program(
        [
            ssa.Operation(
                opcode="tensor.view",
                operands=("x",),
                results=(_value("%1", ("n",)),),
                attrs={"subscript": "(:)", "source": False},
            ),
            ssa.Operation(
                opcode="tensor.view",
                operands=("x",),
                results=(_value("%2", ("n",)),),
                attrs={"subscript": "(:)", "source": False},
            ),
            ssa.Operation(
                opcode="mem.store",
                operands=("%2", "x"),
                attrs={"target": "x"},
            ),
        ],
        outputs=(_value("%2", ("n",)),),
    )

    result = ssa.verify_program(eliminate_common_subexpressions(program))

    assert len(result.blocks[0].operations) == 2
    assert result.outputs[0].name == "%1"
    assert result.blocks[0].operations[-1].operands[0] == "%1"


def test_preserves_effectful_operations():
    program = _program(
        [
            ssa.Operation(
                opcode="math.exp",
                operands=("x",),
                results=(_value("%1", ("n",)),),
                attrs={"callee": "ntl.exp"},
            ),
            ssa.Operation(
                opcode="math.exp",
                operands=("x",),
                results=(_value("%2", ("n",)),),
                attrs={"callee": "ntl.exp"},
            ),
            ssa.Operation(
                opcode="mem.store",
                operands=("%1", "x"),
                attrs={"target": "x"},
            ),
            ssa.Operation(
                opcode="mem.store",
                operands=("%2", "x"),
                attrs={"target": "x"},
            ),
        ]
    )

    result = ssa.verify_program(eliminate_common_subexpressions(program))

    assert len(result.blocks[0].operations) == 3
    assert [op.opcode for op in result.blocks[0].operations] == [
        "math.exp",
        "mem.store",
        "mem.store",
    ]


def test_pass_is_registered_in_default_pipeline():
    names = default_spec("triton").passes

    assert "ssa.eliminate_common_subexpressions" in names
    assert names.index("ssa.eliminate_common_subexpressions") < names.index(
        "ssa.analyze_effects"
    )


@pytest.mark.parametrize("backend", ["triton", "cuda", "tilelang", "ascendc", "bangc"])
def test_pass_supports_all_backends(backend):
    assert "ssa.eliminate_common_subexpressions" in default_spec(backend).passes
