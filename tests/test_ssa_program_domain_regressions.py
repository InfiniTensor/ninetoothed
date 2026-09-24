"""Address and target-semantics regressions that do not require a GPU."""

import ast
import functools
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from ninetoothed import Tensor
from ninetoothed.backends.emitters.cuda import CudaTarget
from ninetoothed.backends.emitters.ssa import _floor_divmod_expr
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.ir import ssa
from tests.test_interpreter_gpu import _jit_function, _numeric_source_expression


def _debug_arrangement(x, out):
    arranged_out = out.tile((1, -1, -1))
    arranged_out.dtype = arranged_out.dtype.squeeze((0,))

    return x.tile((2, 2)), arranged_out


def _copy(x, out):
    out = x  # noqa: F841


def _jagged_expand_arrangement(dst, src, *, jagged_dim):
    tile = (1,) + (32,) * (dst.ndim - 1)
    expanded = src.expand(
        tuple(-1 if dim != jagged_dim else dst.shape[dim] for dim in range(src.ndim))
    )

    return dst.tile(tile), expanded.tile(tile)


def _jagged_copy(dst, src):
    dst = src  # noqa: F841


def _compile(arrangement, application, tensors):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="triton",
            kernel_name="program_domain_regression",
            max_num_configs=1,
        )
    )


def test_flattened_debug_output_preserves_each_source_tile():
    compilation = _compile(
        _debug_arrangement,
        _copy,
        (
            Tensor(2, name="x", dtype="int32", other=-1),
            Tensor(3, name="out", dtype="int32"),
        ),
    )
    function = _jit_function(compilation.artifact.primary_source)
    loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and any(
            isinstance(child, ast.Name) and child.id == "x"
            for child in ast.walk(node.args[0])
        )
    ]
    assert loads
    inputs = {
        "x": np.arange(12, dtype=np.int32).reshape(3, 4),
        "out": np.empty((4, 2, 2), dtype=np.int32),
    }
    symbols = {"index": np.arange(16, dtype=np.int64), "mask": np.ones(16, dtype=bool)}

    for binding in compilation.launch_abi.kernel_args:
        if binding.source not in inputs:
            continue

        value = inputs[binding.source]

        if binding.kind == "shape":
            symbols[binding.name] = value.shape[binding.dim]
        elif binding.kind == "stride":
            symbols[binding.name] = value.strides[binding.dim] // value.itemsize
        elif binding.kind == "tensor":
            symbols[binding.name] = 0

    expected = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, -1, -1, 10, 11, -1, -1])

    for load in loads:
        addresses = _numeric_source_expression(load.args[0], symbols)
        mask_node = next(item.value for item in load.keywords if item.arg == "mask")
        mask = np.asarray(_numeric_source_expression(mask_node, symbols), dtype=bool)
        np.testing.assert_array_equal(mask, expected != -1)
        np.testing.assert_array_equal(addresses[mask], expected[mask])


@pytest.mark.parametrize("rank", (3, 4))
@pytest.mark.parametrize("jagged_dim", (1, 2))
def test_expanded_dense_input_does_not_introduce_unbound_jagged_extent(
    rank, jagged_dim
):
    compilation = _compile(
        functools.partial(_jagged_expand_arrangement, jagged_dim=jagged_dim),
        _jagged_copy,
        (Tensor(rank, name="dst", jagged_dim=jagged_dim), Tensor(rank, name="src")),
    )
    function = _jit_function(compilation.artifact.primary_source)
    undefined_extents = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id.endswith("_seq_len")
        and not node.id.endswith("_max_seq_len")
    }
    assert not undefined_extents


def test_cuda_signed_division_source_matches_python_integer_semantics(tmp_path):
    compiler = shutil.which("c++")

    if compiler is None:
        pytest.skip("C++ compiler is required to evaluate generated CUDA scalar syntax")

    context = SimpleNamespace(target=CudaTarget())
    expressions = []

    for operator in ("floordiv", "mod"):
        operation = ssa.Operation(
            opcode=f"arith.{operator}",
            operands=("x", "y"),
            results=(
                ssa.Value(name="%result", type=ssa.Type(kind="scalar", dtype="int32")),
            ),
        )
        expressions.append(_floor_divmod_expr(operator, operation, ("x", "y"), context))

    source = tmp_path / "signed_division.cpp"
    source.write_text(
        "#include <cstdint>\n#include <iostream>\nint main() {\n"
        "std::int32_t x, y;\nwhile (std::cin >> x >> y) {\n"
        f"std::cout << ({expressions[0]}) << ' ' << ({expressions[1]}) << '\\n';\n"
        "}\n}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "signed_division"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)], check=True
    )
    pairs = [
        (x, y)
        for x in (-2147483648, -7, -6, -1, 0, 1, 6, 7, 2147483647)
        for y in (-2147483648, -7, -3, -1, 1, 3, 7, 2147483647)
        if not (x == -2147483648 and y == -1)
    ]
    run = subprocess.run(
        [str(executable)],
        input="".join(f"{x} {y}\n" for x, y in pairs),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [tuple(map(int, line.split())) for line in run.stdout.splitlines()]
    expected = [(x // y, x % y) for x, y in pairs]
    assert actual == expected
