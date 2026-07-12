import pytest

from ninetoothed.frontend.python import LoweringError, from_source
from ninetoothed.ir import TensorSpec, ssa


def _tensor(name, shape, dtype="float32"):
    return TensorSpec(ndim=len(shape), shape=tuple(shape), dtype=dtype, name=name)


def test_verifier_rejects_undefined_operand():
    program = ssa.Program(
        kind="invalid",
        blocks=(
            ssa.Block(
                operations=(ssa.Operation(opcode="arith.add", operands=("missing",)),)
            ),
        ),
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
