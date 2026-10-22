"""Source-level contract tests for the BangC backend."""

import pytest

from ninetoothed.backends import emit
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


def _kernel(source, tensors, *, name="application", kind="application", options=None):
    return Kernel(
        kernel_name=name,
        source=None,
        tensors=tensors,
        ssa=from_source(source, tensors, kind=kind),
        compiler_options=options or {},
    )


def _elementwise_source(expression, *, output_dtype="float32", options=None):
    tensors = (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
        TensorSpec(ndim=1, shape=("n",), dtype=output_dtype, name="out"),
    )

    kernel = _kernel(
        "import ninetoothed.language as ntl\n"
        "def application(x,y,out):\n    out = " + expression + "\n",
        tensors,
        kind="application",
        options={"backend_options": options or {}},
    )

    return emit(kernel, "bangc").primary_source


def test_float_to_int32_has_vector_conversion_and_defined_invalid_values():
    source = _elementwise_source("x.to(ntl.int32)", output_dtype="int32")

    assert "__bang_float2int32_tz(nt_dag_buf_1, nt_dag_buf_0, nt_aligned, 0)" in source
    assert "nt_value >= -2147483648.0f && nt_value < 2147483648.0f" in source
    assert "nt_dag_buf_1[nt_j] = (-2147483647 - 1)" in source
    assert "if (!(nt_cast_domain < 2147483648.0f))" in source
    assert "nt_cast_domain = nt_max(nt_cast_domain, nt_wrk[nt_j])" in source
    assert "#define nt_max(a, b)" in source
    assert "struct NtInt32Cast<T, true>" in source


def test_fused_mixed_math_uses_generic_vector_dag():
    source = _elementwise_source("x * ntl.sqrt(y * y + 0.01)")

    assert "nt_dag_buf_" in source
    assert "__bang_mul(" in source
    assert "__bang_sqrt(" in source
    assert "__bang_add_scalar(" in source


def test_comparison_and_select_share_a_typed_dag():
    source = _elementwise_source("ntl.where(x>0,x,0.1*x)")

    assert "__bang_gt(" in source
    assert "__bang_bxor(" in source
    assert "__nram__ bool nt_dag_buf_" not in source
    assert "__bang_mul_scalar(" in source


@pytest.mark.parametrize("operator", ("exp", "sin"))
def test_strict_vector_math_checks_domain_and_retains_scalar_outlier_helper(operator):
    source = _elementwise_source(f"ntl.{operator}(x)")
    assert "nt_domain" in source
    assert "nt_value >= -" in source
    assert "__bang_active_exphp(" not in source

    if operator in {"sin", "cos"}:
        assert f"nt_{operator}(nt_dag_buf_" in source
    elif operator != "sigmoid":
        assert f"{operator}f(nt_dag_buf_" in source
