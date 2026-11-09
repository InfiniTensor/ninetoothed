import os
from pathlib import Path

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends.rvne_toolchain import find_rvne_toolchain
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.compiler.runtime import materialize


def _identity(*tensors):
    return tensors


def _add(x, y, out):
    out = x + y  # noqa: F841


def _matmul(x, y, out):
    out = ntl.dot(x, y)  # noqa: F841


def _sum(x, out):
    out = ntl.sum(x)  # noqa: F841


def _sum_rows(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _min(x, out):
    out = ntl.min(x)  # noqa: F841


def _min_rows(x, out):
    out = ntl.min(x, axis=1)  # noqa: F841


def _max(x, out):
    out = ntl.max(x)  # noqa: F841


def _max_rows(x, out):
    out = ntl.max(x, axis=1)  # noqa: F841


@pytest.fixture(scope="module")
def rvne_sdk():
    if not os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN"):
        pytest.skip("RVNE SDK is not configured")

    return find_rvne_toolchain()


def _run_qemu(application, arrays, output_dir, toolchain):
    compilation = compile_kernel(
        CompileRequest(
            arrangement=_identity,
            application=application,
            tensors=tuple(Tensor(shape=a.shape, dtype=a.dtype.name) for a in arrays),
            backend="rvne",
            caller="numpy",
            backend_options={"toolchain_root": str(toolchain.root)},
        )
    )
    handle = materialize(compilation, output_dir=output_dir, mode="aot")
    built = handle._built_artifact
    source = Path(built.source_path)
    executable = Path(built.binary_path)
    assert source.is_file() and source.suffix == ".cpp"
    assert executable.is_file() and executable.suffix == ".elf"
    header = executable.read_bytes()[:20]
    assert header[:6] == b"\x7fELF\x02\x01"
    assert int.from_bytes(header[18:20], "little") == 243
    assert handle(*arrays) is arrays[-1]


def _assert_result(actual, expected):
    if actual.dtype == np.dtype("float32"):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ("int32", "float32"))
@pytest.mark.parametrize("shape", ((7,), (3, 5)), ids=("vector", "matrix"))
def test_rvne_add_qemu(tmp_path, rvne_sdk, dtype, shape):
    count = int(np.prod(shape))
    x = (np.arange(count, dtype=np.int32) % 9 - 4).reshape(shape).astype(dtype)
    y = (np.arange(count, dtype=np.int32) % 7 - 3).reshape(shape).astype(dtype)

    if dtype == "float32":
        x /= np.float32(3.0)
        y /= np.float32(7.0)

    out = np.full(shape, -123, dtype=dtype)
    expected = x + y
    _run_qemu(_add, (x, y, out), tmp_path, rvne_sdk)
    _assert_result(out, expected)


@pytest.mark.parametrize("dtype", ("int32", "float32"))
@pytest.mark.parametrize("dimensions", ((2, 3, 2), (3, 5, 7)), ids=("2x3x2", "3x5x7"))
def test_rvne_matmul_qemu(tmp_path, rvne_sdk, dtype, dimensions):
    rows, inner, columns = dimensions
    x = (
        (np.arange(rows * inner, dtype=np.int32) % 7 - 3)
        .reshape(rows, inner)
        .astype(dtype)
    )
    y = (
        (np.arange(inner * columns, dtype=np.int32) % 9 - 4)
        .reshape(inner, columns)
        .astype(dtype)
    )

    if dtype == "float32":
        x /= np.float32(3.0)
        y /= np.float32(7.0)

    out = np.full((rows, columns), -123, dtype=dtype)
    expected = np.matmul(x, y)
    _run_qemu(_matmul, (x, y, out), tmp_path, rvne_sdk)
    _assert_result(out, expected)


@pytest.mark.parametrize("dtype", ("int32", "float32"))
@pytest.mark.parametrize("operator", ("sum", "min", "max"))
@pytest.mark.parametrize("axis", (None, 1), ids=("full", "rows"))
def test_rvne_reduction_qemu(tmp_path, rvne_sdk, dtype, operator, axis):
    x = np.array(
        [[-8, -7, -3, -2, -5], [8, 7, 3, 2, 5], [-4, 0, 7, -6, 0]],
        dtype=dtype,
    )

    if dtype == "float32":
        x /= np.float32(3.0)

    if axis is None:
        x = x.reshape(-1)

    applications = {
        ("sum", None): _sum,
        ("sum", 1): _sum_rows,
        ("min", None): _min,
        ("min", 1): _min_rows,
        ("max", None): _max,
        ("max", 1): _max_rows,
    }

    if operator == "sum":
        expected = np.sum(x, axis=axis, dtype=dtype)
    else:
        expected = getattr(np, operator)(x, axis=axis)

    expected = np.asarray(expected, dtype=dtype).reshape(-1)
    out = np.full(expected.shape, -123, dtype=dtype)
    _run_qemu(applications[operator, axis], (x, out), tmp_path, rvne_sdk)
    _assert_result(out, expected)
