import re
import shutil
import subprocess

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import CompileRequest, compile_kernel


def arrangement(x, scale, out):
    return x.tile((1, 32)), scale, out.tile((1, 32))


def application(x, scale, out):
    mean = ntl.sum(x, axis=1) / scale
    out = x - mean[:, None]  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("constexpr", [False, True])
def test_scalar_in_row_expression_is_not_loaded_as_pointer(constexpr):
    scalar = Tensor(
        0, dtype="float32", constexpr=constexpr, value=32.0 if constexpr else None
    )
    kernel = ninetoothed.make(
        arrangement, application, (Tensor(2), scalar, Tensor(2)), max_num_configs=1
    )
    x = torch.randn((3, 32), device="cuda")
    out = torch.full_like(x, -12345)
    kernel(x, 32.0, out)
    torch.testing.assert_close(out, x - x.mean(1, keepdim=True))


def _sum_rows(x, out):
    accumulator = ntl.zeros(out.shape, dtype=ntl.int32)

    for row in range(x.shape[0]):
        accumulator += x[row]

    out = accumulator  # noqa: F841


def _row_arrangement(x, out, layout):
    if layout == "plain":
        return x, out

    if layout == "nested":
        x = x.tile((1, 7)).tile((3, 1))
        x.dtype = x.dtype.squeeze(1)
    else:
        x = x.tile((3, 7))

    return x, out.tile((1, 7))


@pytest.mark.parametrize("backend", ("rvne", "cuda", "triton"))
@pytest.mark.parametrize("layout", ("plain", "tiled", "nested"))
def test_extracted_row_keeps_its_index_across_dtype_levels(tmp_path, backend, layout):
    compilation = compile_kernel(
        CompileRequest(
            arrangement=lambda x, out: _row_arrangement(x, out, layout),
            application=_sum_rows,
            tensors=(
                Tensor(shape=(3, 7), dtype="int32"),
                Tensor(shape=(1, 7), dtype="int32"),
            ),
            backend=backend,
            kernel_name="sum_rows",
        )
    )
    source = compilation.artifact.primary_source
    loop = re.search(r"for\s+(?:\(int64_t\s+)?(v\w+_i)\b", source)

    assert loop is not None

    address_lines = (
        line
        for line in source.splitlines()
        if "tl.load(x +" in line or "x[" in line or re.search(r"nt_idx_\d+\s*=", line)
    )
    assert any(loop.group(1) in line for line in address_lines)

    if backend != "rvne":
        return

    compiler = shutil.which("clang++") or shutil.which("g++")

    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")

    shapes = {"x": (3, 7), "out": (1, 7)}
    strides = {"x": (7, 1), "out": (7, 1)}
    arguments = []

    for binding in compilation.launch_abi.kernel_args:
        if binding.kind == "tensor":
            arguments.append(binding.source)
        elif binding.kind == "shape":
            arguments.append(str(shapes[binding.source][binding.dim]))
        elif binding.kind == "stride":
            arguments.append(str(strides[binding.source][binding.dim]))
        else:
            pytest.fail(f"Unexpected binding in row-read regression: {binding.kind}.")

    main = f"""
int main() {{
    int32_t x[21];
    int32_t out[7] = {{}};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 7; ++col)
            x[row * 7 + col] = row * 100 + col;
    if ({compilation.artifact.entrypoint}({", ".join(arguments)})) return 1;
    for (int col = 0; col < 7; ++col)
        if (out[col] != 300 + 3 * col) return 2;
    return 0;
}}
"""
    source_path = tmp_path / "sum_rows.cpp"
    executable = tmp_path / "sum_rows"
    source_path.write_text(source + main, encoding="utf-8")
    subprocess.run(
        [compiler, "-O2", "-fwrapv", str(source_path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
