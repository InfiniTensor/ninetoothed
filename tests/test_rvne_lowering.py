import shutil
import subprocess

import pytest

from ninetoothed.backends.emitters.rvne import RvneTarget, emit
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


def _kernel(source, tensors, name="rvne_test"):
    program = from_source(source, tensors, kind=name)
    assert program is not None

    return Kernel(
        kernel_name=name,
        source=source,
        source_language="ninetoothed-python",
        entrypoint=name,
        tensors=tensors,
        ssa=program,
    )


def _compile_and_run(tmp_path, source, main):
    compiler = shutil.which("clang++") or shutil.which("g++")

    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")

    path = tmp_path / "kernel.cpp"
    executable = tmp_path / "kernel_test"
    path.write_text(source + "\n" + main, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++11", "-O2", "-fwrapv", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_rvne_serial_abi_and_integer_arithmetic(tmp_path):
    kernel = _kernel(
        "def application(x, scale, out):\n"
        "    acc = x * scale\n"
        "    for i in range(3):\n"
        "        acc = acc + x\n"
        "    out = ntl.where(acc > 0, acc, -acc)\n",
        (
            TensorSpec(ndim=1, shape=("n",), dtype="int32", name="x"),
            TensorSpec(ndim=0, dtype="int32", name="scale"),
            TensorSpec(ndim=1, shape=("n",), dtype="int32", name="out"),
        ),
    )
    artifact = emit(kernel)
    source = artifact.primary_source
    assert artifact.primary_source_name == "rvne_test.cpp"
    assert artifact.metadata["variables"] == ("x", "scale")
    assert artifact.metadata["outputs"] == ("out",)
    assert artifact.metadata["shape_params"] == ("n",)
    assert 'extern "C" int launch_rvne_test(' in source
    assert "const int32_t* x" in source
    assert "int32_t scale" in source
    assert "int64_t n" in source
    assert "cuda" not in source.lower()
    assert "blockIdx" not in source
    _compile_and_run(
        tmp_path,
        source,
        """int main() {
    int32_t x[] = {-3, 0, 5};
    int32_t out[3] = {};
    if (launch_rvne_test(x, 2, out, 3)) return 1;
    return out[0] != 15 || out[1] != 0 || out[2] != 25;
}
""",
    )


@pytest.mark.parametrize("operator,expected", [("sum", -9), ("min", -8), ("max", 7)])
def test_rvne_signed_integer_reductions(tmp_path, operator, expected):
    kernel = _kernel(
        f"def application(x, out):\n    out = ntl.{operator}(x)\n",
        (
            TensorSpec(ndim=1, shape=("5",), dtype="int32", name="x"),
            TensorSpec(ndim=1, shape=("1",), dtype="int32", name="out"),
        ),
    )
    _compile_and_run(
        tmp_path,
        emit(kernel).primary_source,
        f"""int main() {{
    int32_t x[] = {{-8, 7, -3, 0, -5}};
    int32_t out[1] = {{}};
    if (launch_rvne_test(x, out)) return 1;
    return out[0] != {expected};
}}
""",
    )


def test_rvne_matmul_reduction(tmp_path):
    kernel = _kernel(
        "def application(x, y, out):\n    out = ntl.dot(x, y)\n",
        (
            TensorSpec(ndim=2, shape=("2", "3"), dtype="int32", name="x"),
            TensorSpec(ndim=2, shape=("3", "2"), dtype="int32", name="y"),
            TensorSpec(ndim=2, shape=("2", "2"), dtype="int32", name="out"),
        ),
    )
    _compile_and_run(
        tmp_path,
        emit(kernel).primary_source,
        """int main() {
    int32_t x[] = {1, 0, 1, 0, 1, 1};
    int32_t y[] = {-8, 7, 3, -4, 2, -1};
    int32_t out[4] = {};
    int32_t expected[] = {-6, 6, 5, -5};
    if (launch_rvne_test(x, y, out)) return 1;
    for (int i = 0; i < 4; ++i) if (out[i] != expected[i]) return 2;
    return 0;
}
""",
    )


def test_rvne_floor_index_and_mask(tmp_path):
    kernel = _kernel(
        "def application(x, out):\n    out = x\n",
        (
            TensorSpec(
                ndim=1,
                shape=("5",),
                dtype="int32",
                name="x",
                attrs={
                    "source_shape": ("2",),
                    "view_linear_offset": "floor((index - 1)/2)",
                    "view_mask": "(floor((index - 1)/2) >= 0) & (floor((index - 1)/2) < 2)",
                },
            ),
            TensorSpec(ndim=1, shape=("5",), dtype="int32", name="out"),
        ),
    )
    source = emit(kernel).primary_source
    assert "nt_floor_div((index - 1), 2)" in source
    _compile_and_run(
        tmp_path,
        source,
        """int main() {
    int32_t x[] = {11, 22};
    int32_t out[5] = {};
    int32_t expected[] = {0, 11, 11, 22, 22};
    if (launch_rvne_test(x, out)) return 1;
    for (int i = 0; i < 5; ++i) if (out[i] != expected[i]) return 2;
    return 0;
}
""",
    )


@pytest.mark.parametrize("dtype", ["float16", "float64", "int8"])
def test_rvne_rejects_unsupported_dtypes(dtype):
    kernel = _kernel(
        "def application(x, out):\n    out = x + x\n",
        tuple(
            TensorSpec(ndim=1, shape=("4",), dtype=dtype, name=name)
            for name in ("x", "out")
        ),
    )

    with pytest.raises((TypeError, ValueError), match="dtype"):
        emit(kernel)


@pytest.mark.parametrize(
    "expression", ["ntl.exp(x)", "x / x", "x // x", "x % x", "x ** 2"]
)
def test_rvne_rejects_unsupported_arithmetic(expression):
    kernel = _kernel(
        f"def application(x, out):\n    out = {expression}\n",
        tuple(
            TensorSpec(ndim=1, shape=("4",), dtype="float32", name=name)
            for name in ("x", "out")
        ),
    )

    with pytest.raises(ValueError, match="does not support SSA operation"):
        emit(kernel)


def test_rvne_preserves_floor_and_mod_for_negative_indices():
    target = RvneTarget()
    assert target.render_index_expr("floor((i - 1)/2)") == "nt_floor_div((i - 1), 2)"
    assert target.render_index_expr("Mod(i - 1, 32)") == "nt_floor_mod((i - 1), 32)"

    with pytest.raises(ValueError, match="index expression"):
        target.render_index_expr("unknown(index)")


def test_rvne_decreasing_loop_and_integer_wrap(tmp_path):
    kernel = _kernel(
        "def application(x, out):\n"
        "    acc = x\n"
        "    for i in range(3, 0, -1):\n"
        "        acc = acc + x\n"
        "    out = acc\n",
        tuple(
            TensorSpec(ndim=1, shape=("2",), dtype="int32", name=name)
            for name in ("x", "out")
        ),
    )
    _compile_and_run(
        tmp_path,
        emit(kernel).primary_source,
        """int main() {
    int32_t x[] = {2147483647, -3};
    int32_t out[2] = {};
    if (launch_rvne_test(x, out)) return 1;
    return out[0] != -4 || out[1] != -12;
}
""",
    )


@pytest.mark.parametrize("reduce", [False, True])
def test_rvne_bool_invert(tmp_path, reduce):
    expression = "ntl.max(~x)" if reduce else "~x"
    kernel = _kernel(
        f"def application(x, out):\n    out = {expression}\n",
        (
            TensorSpec(ndim=1, shape=("2",), dtype="bool", name="x"),
            TensorSpec(
                ndim=1, shape=("1" if reduce else "2",), dtype="bool", name="out"
            ),
        ),
    )
    main = """int main() {
    bool x[] = {false, true};
    bool out[2] = {};
    if (launch_rvne_test(x, out)) return 1;
    return out[0] != true || out[1] != false;
}
"""

    if reduce:
        main = """int main() {
    bool x[] = {true, true};
    bool out[1] = {};
    if (launch_rvne_test(x, out)) return 1;
    return out[0] != false;
}
"""

    _compile_and_run(tmp_path, emit(kernel).primary_source, main)


@pytest.mark.parametrize("operator", ["maximum", "minimum"])
@pytest.mark.parametrize("scalar_first", [False, True])
def test_rvne_mixed_float_extrema_preserve_nan(tmp_path, operator, scalar_first):
    arguments = "0, x" if scalar_first else "x, 0"
    kernel = _kernel(
        f"def application(x, out):\n    out = ntl.{operator}({arguments})\n",
        tuple(
            TensorSpec(ndim=1, shape=("4",), dtype="float32", name=name)
            for name in ("x", "out")
        ),
    )
    expected = "0, 0, 2" if operator == "maximum" else "-1, 0, 0"
    _compile_and_run(
        tmp_path,
        emit(kernel).primary_source,
        f"""int main() {{
    float x[] = {{NAN, -1, 0, 2}};
    float out[4] = {{}};
    float expected[] = {{{expected}}};
    if (launch_rvne_test(x, out)) return 1;
    if (!isnan(out[0])) return 2;
    for (int i = 1; i < 4; ++i) if (out[i] != expected[i - 1]) return 3;
    return 0;
}}
""",
    )


@pytest.mark.parametrize("operator", [">>", "<<"])
@pytest.mark.parametrize("dtype,bits", [("int32", 32), ("int64", 64)])
def test_rvne_signed_shift_extremes(tmp_path, operator, dtype, bits):
    kernel = _kernel(
        f"def application(x, out):\n    out = x {operator} {bits - 1}\n",
        tuple(
            TensorSpec(ndim=1, shape=("4",), dtype=dtype, name=name)
            for name in ("x", "out")
        ),
    )
    minimum = f"INT{bits}_MIN"
    expected = (
        "-1, -1, 0, 0" if operator == ">>" else f"0, {minimum}, {minimum}, {minimum}"
    )
    _compile_and_run(
        tmp_path,
        emit(kernel).primary_source,
        f"""int main() {{
    {dtype}_t x[] = {{{minimum}, -1, 1, INT{bits}_MAX}};
    {dtype}_t out[4] = {{}};
    {dtype}_t expected[] = {{{expected}}};
    if (launch_rvne_test(x, out)) return 1;
    for (int i = 0; i < 4; ++i) if (out[i] != expected[i]) return 2;
    return 0;
}}
""",
    )


@pytest.mark.parametrize("count", ["-1", "32", "33", "count"])
@pytest.mark.parametrize("operator", [">>", "<<"])
def test_rvne_rejects_invalid_or_dynamic_shifts(count, operator):
    kernel = _kernel(
        f"def application(x, count, out):\n    out = x {operator} {count}\n",
        (
            TensorSpec(ndim=1, shape=("4",), dtype="uint32", name="x"),
            TensorSpec(ndim=0, dtype="int32", name="count"),
            TensorSpec(ndim=1, shape=("4",), dtype="uint32", name="out"),
        ),
    )

    with pytest.raises(ValueError, match="constant count within its bit width"):
        emit(kernel)
