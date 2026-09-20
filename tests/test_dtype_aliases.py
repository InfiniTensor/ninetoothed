import ctypes
from types import SimpleNamespace

import pytest
import torch

import ninetoothed
from ninetoothed.backends import emit as emit_kernel
from ninetoothed.backends.materializers import cuda as cuda_materializer
from ninetoothed.backends.materializers import triton as triton_materializer
from ninetoothed.compiler.runtime import _validate_dtype_contract
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec

_INTEGER_DTYPES = (
    (ninetoothed.int8, "int8", ctypes.c_int8, -(2**7)),
    (ninetoothed.int16, "int16", ctypes.c_int16, -(2**15)),
    (ninetoothed.int32, "int32", ctypes.c_int32, -(2**31)),
    (ninetoothed.int64, "int64", ctypes.c_int64, -(2**63) + 1),
    (ninetoothed.uint8, "uint8", ctypes.c_uint8, 2**8 - 1),
    (ninetoothed.uint16, "uint16", ctypes.c_uint16, 2**16 - 1),
    (ninetoothed.uint32, "uint32", ctypes.c_uint32, 2**32 - 1),
    (ninetoothed.uint64, "uint64", ctypes.c_uint64, 2**64 - 1),
)


def _kernel(source, tensors):
    program = from_source(source, tensors, kind="integer_alias_application")
    assert program is not None

    return Kernel(
        kernel_name="integer_alias_application",
        source=source,
        source_language="ninetoothed-python",
        entrypoint="integer_alias_application",
        tensors=tensors,
        ssa=program,
    )


@pytest.mark.parametrize("alias,dtype,ctype,value", _INTEGER_DTYPES)
@pytest.mark.parametrize("expression", ("(x + divisor) / divisor", "divisor + x"))
def test_float_tensor_arithmetic_accepts_public_integer_scalars(
    alias, dtype, ctype, value, expression
):
    del ctype, value

    source = (
        f"def integer_alias_application(x, divisor, out):\n    out = {expression}\n"
    )
    kernels = []

    for scalar_dtype in (alias, dtype):
        kernels.append(
            _kernel(
                source,
                (
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                    TensorSpec(ndim=0, dtype=scalar_dtype, name="divisor"),
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
                ),
            )
        )

    arithmetic = [
        operation
        for operation in kernels[0].ssa.blocks[0].operations
        if operation.opcode.startswith("arith.")
    ]
    assert arithmetic
    assert all(operation.results[0].type.dtype == "float32" for operation in arithmetic)

    for backend in ("triton", "cuda", "tilelang"):
        alias_source, canonical_source = (
            emit_kernel(kernel, backend).primary_source for kernel in kernels
        )
        assert alias_source == canonical_source


@pytest.mark.parametrize("alias,dtype,ctype,value", _INTEGER_DTYPES)
def test_integer_casts_accept_public_aliases(alias, dtype, ctype, value):
    del ctype, value

    kernel = _kernel(
        f"def integer_alias_application(x, out):\n    out = x.to({alias!r})\n",
        (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype=alias, name="out"),
        ),
    )
    cast = next(
        operation
        for operation in kernel.ssa.blocks[0].operations
        if operation.opcode == "tensor.cast"
    )
    assert cast.results[0].type.dtype == dtype
    expected = {
        "triton": f"tl.{dtype}",
        "cuda": f"static_cast<{dtype}_t>",
        "tilelang": f'T.Cast("{dtype}"',
    }

    for backend, fragment in expected.items():
        assert fragment in emit_kernel(kernel, backend).primary_source


@pytest.mark.parametrize("alias,dtype,ctype,value", _INTEGER_DTYPES)
def test_triton_aot_integer_arguments_preserve_width_and_signedness(
    alias, dtype, ctype, value
):
    del dtype

    tensors = (
        TensorSpec(ndim=1, shape=("n",), dtype=alias, name="x"),
        TensorSpec(ndim=0, dtype=alias, name="scalar"),
    )
    scalar_binding = SimpleNamespace(kind="scalar", source="scalar")
    compilation = SimpleNamespace(
        kernel=SimpleNamespace(tensors=tensors),
        launch_plan=SimpleNamespace(tuning_candidates=()),
        launch_abi=SimpleNamespace(
            kernel_args=(SimpleNamespace(kind="tensor", source="x"), scalar_binding)
        ),
        artifact=SimpleNamespace(metadata={"program_mode": {"scalar": True}}),
    )
    assert triton_materializer._compile_signature(compilation) == f"*{alias},{alias},1"
    packed = triton_materializer._triton_aot_scalar(value, alias)
    argument_type = triton_materializer._triton_aot_ctype(
        scalar_binding, {tensor.name: tensor for tensor in tensors}
    )
    assert type(packed) is argument_type is ctype
    assert packed.value == value
    cuda_packed = cuda_materializer._cuda_scalar(value, alias)
    assert type(cuda_packed) is ctype
    assert cuda_packed.value == value


@pytest.mark.parametrize("alias,dtype,ctype,value", _INTEGER_DTYPES)
def test_runtime_accepts_matching_integer_dtype_aliases(alias, dtype, ctype, value):
    del ctype, value

    spec = TensorSpec(ndim=1, shape=(1,), dtype=alias, name="x")
    _validate_dtype_contract(spec, torch.empty(1, dtype=getattr(torch, dtype)))

    with pytest.raises(TypeError, match=f"expected {dtype}"):
        _validate_dtype_contract(spec, torch.empty(1, dtype=torch.float32))
