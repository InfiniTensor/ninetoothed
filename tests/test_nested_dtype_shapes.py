import pytest

from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec


@pytest.mark.parametrize("constructor", ("zeros", "empty", "full"))
@pytest.mark.parametrize(
    ("shape", "expected"),
    (
        ("x.dtype.shape", ("17",)),
        ("(x.dtype.shape[0],)", ("17",)),
        ("x.dtype.dtype.shape", ("5",)),
        ("x[0].dtype.shape", ("5",)),
    ),
)
def test_constructors_resolve_nested_tensor_dtype_shapes(constructor, shape, expected):
    value = ", 1" if constructor == "full" else ""
    source = f"""
def application(x, out):
    acc = ntl.{constructor}({shape}{value}, dtype=ntl.float32)
    out = ntl.sum(acc)
"""
    program = from_source(
        source,
        (
            TensorSpec(
                ndim=1,
                shape=("n",),
                dtype="float32",
                name="x",
                attrs={"dtype_shapes": (("n",), ("17",), ("5",))},
            ),
            TensorSpec(ndim=0, dtype="float32", name="out"),
        ),
        strict=True,
    )
    operations = program.blocks[0].operations
    initializer = next(
        operation
        for operation in operations
        if operation.opcode in {"tensor.zeros", "tensor.full"}
    )
    assert initializer.results[0].type.shape == expected
    assert all(operation.opcode != "symbol.attr" for operation in operations)
