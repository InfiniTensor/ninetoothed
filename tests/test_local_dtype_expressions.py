import ast

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends import emit as emit_kernel
from ninetoothed.compiler import lower, make
from ninetoothed.frontend.errors import LoweringError
from ninetoothed.frontend.python import from_application, from_source
from ninetoothed.ir import Kernel, TensorSpec
from tests.utils import get_available_devices


def _kernel(source, dtype="float16"):
    tensors = (
        TensorSpec(ndim=1, shape=(128,), dtype=dtype, name="x"),
        TensorSpec(ndim=1, shape=(128,), dtype=dtype, name="out"),
    )
    program = from_source(source, tensors, strict=True)

    return Kernel(
        kernel_name="local_dtype_application",
        source=source,
        source_language="ninetoothed-python",
        entrypoint="application",
        tensors=tensors,
        ssa=program,
    )


def _cast_dtypes(program):
    return tuple(
        operation.results[0].type.dtype
        for operation in program.blocks[0].operations
        if operation.opcode == "tensor.cast"
    )


@pytest.mark.parametrize("name", ("dtype", "chosen_dtype"))
@pytest.mark.parametrize("expression", ("ntl.cast(x, {name})", "x.to({name})"))
def test_local_dtype_alias_is_independent_of_variable_name(name, expression):
    kernel = _kernel(
        f"def application(x, out):\n    {name} = ntl.float32\n"
        f"    out = {expression.format(name=name)}\n"
    )
    assert _cast_dtypes(kernel.ssa) == ("float32",)

    for backend, expected in (
        ("triton", "tl.float32"),
        ("cuda", "static_cast<float>"),
        ("tilelang", 'T.Cast("float32"'),
    ):
        assert expected in emit_kernel(kernel, backend).primary_source


def test_dtype_alias_chain_retains_value_before_reassignment():
    kernel = _kernel(
        """
def application(x, out):
    dtype = ntl.float32
    saved_dtype = dtype
    chained_dtype = saved_dtype
    dtype = ntl.float16
    wide = ntl.cast(x, chained_dtype)
    narrow = x.to(dtype)
    out = wide + narrow
"""
    )
    assert _cast_dtypes(kernel.ssa) == ("float32", "float16")
    source = emit_kernel(kernel, "triton").primary_source
    assert "tl.float32" in source
    assert "tl.float16" in source


@pytest.mark.parametrize("constructor", ("zeros", "empty", "full"))
def test_constructor_accepts_local_dtype_alias(constructor):
    value = ", 1" if constructor == "full" else ""
    kernel = _kernel(
        f"def application(x, out):\n    dtype = ntl.float32\n"
        f"    out = ntl.{constructor}(x.shape{value}, dtype=dtype)\n"
    )
    initializer = next(
        operation
        for operation in kernel.ssa.blocks[0].operations
        if operation.opcode in {"tensor.zeros", "tensor.full"}
    )
    assert initializer.results[0].type.dtype == "float32"
    assert "tl.float32" in emit_kernel(kernel, "triton").primary_source


@pytest.mark.parametrize(
    ("dtype", "expected"),
    (("float16", "float32"), ("bfloat16", "bfloat16"), ("float32", "float32")),
)
def test_conditional_dtype_alias_resolves_nested_tensor_element_dtype(dtype, expected):
    kernel = _kernel(
        """
def application(x, out):
    dtype = out.dtype.dtype
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32
    out = ntl.cast(x, exp_dtype)
""",
        dtype=dtype,
    )
    assert _cast_dtypes(kernel.ssa) == (expected,)
    assert f"tl.{expected}" in emit_kernel(kernel, "triton").primary_source


def test_local_dtype_alias_preserves_unspecialized_tensor_dtype():
    kernel = _kernel(
        """
def application(x, out):
    dtype = out.dtype.dtype
    chosen_dtype = dtype
    out = ntl.cast(x, chosen_dtype)
""",
        dtype=None,
    )
    assert _cast_dtypes(kernel.ssa) == (None,)
    source = emit_kernel(kernel, "triton").primary_source
    assert "out.dtype.element_ty" in source
    assert "tl.chosen_dtype" not in source
    ast.parse(source)


def test_runtime_data_cannot_select_dtype():
    with pytest.raises(
        LoweringError, match="Dtype selection cannot depend on runtime data"
    ):
        _kernel(
            """
def application(x, out):
    dtype = ntl.float16 if x[0] > 0 else ntl.float32
    out = ntl.cast(x, dtype)
"""
        )


def test_type_only_branch_reassigns_unspecialized_dtype():
    kernel = _kernel(
        """
def application(x, out):
    dtype = out.dtype
    if dtype == ntl.float16 and not dtype == ntl.float32:
        dtype = ntl.float32
    out = ntl.cast(x, dtype)
""",
        dtype=None,
    )
    assert _cast_dtypes(kernel.ssa) == (None,)
    source = emit_kernel(kernel, "triton").primary_source
    assert "out.dtype.element_ty" in source
    assert "tl.float32" in source
    ast.parse(source)


def _exp_with_dtype(x, dtype):
    exp_dtype = dtype if dtype != ntl.float16 else ntl.float32

    return ntl.cast(ntl.exp(ntl.cast(x, exp_dtype)), dtype)


def _helper_application(x, out):
    dtype = out.dtype
    numerator = _exp_with_dtype(x, dtype)
    denominator = ntl.cast(0, dtype)
    out = numerator + denominator  # noqa: F841


def _two_helper_calls(x, out):
    dtype = ntl.float32
    wide = _exp_with_dtype(x, dtype)
    narrow = _exp_with_dtype(x, ntl.float16)
    out = wide + narrow  # noqa: F841


def _arrangement(x, out):
    return x.tile((128,)), out.tile((128,))


def test_inlined_helper_dtype_assignments_are_scoped_to_each_call():
    program = from_application(
        _two_helper_calls,
        (
            TensorSpec(ndim=1, shape=(128,), dtype="float16", name="x"),
            TensorSpec(ndim=1, shape=(128,), dtype="float32", name="out"),
        ),
        strict=True,
    )
    assert _cast_dtypes(program) == ("float32", "float32", "float32", "float16")


def test_public_lower_accepts_helper_with_unspecialized_conditional_dtype():
    artifact = lower(
        _arrangement,
        _helper_application,
        (Tensor(1), Tensor(1)),
        backend="triton",
    )
    source = artifact.primary_source
    assert "out.dtype.element_ty" in source
    assert "tl.float16" in source
    assert "tl.float32" in source
    assert "_exp_with_dtype" not in source
    assert "tl.__nt_inline" not in source
    ast.parse(source)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_helper_conditional_dtype_runtime(device, dtype):
    kernel = make(
        _arrangement,
        _helper_application,
        (Tensor(1), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    x = torch.linspace(-4, 4, 257, dtype=dtype, device=device)
    out = torch.empty_like(x)
    kernel(x, out)
    torch.testing.assert_close(out, x.float().exp().to(dtype))
