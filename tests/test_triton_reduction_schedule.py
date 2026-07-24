import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, make
from tests.utils import get_available_devices

WIDTH = Symbol("WIDTH", constexpr=True)
HEIGHT = Symbol("HEIGHT", constexpr=True)


def _row_arrangement(x, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), out.tile((1, WIDTH))


def _row_reduced_arrangement(x, out, WIDTH=WIDTH):
    return x.tile((1, WIDTH)), out.tile((1,))


def _column_arrangement(x, out, HEIGHT=HEIGHT):
    return x.tile((HEIGHT, 1)), out.tile((HEIGHT, 1))


def _middle_axis_arrangement(x, out):
    return x.tile((2, 3, 5)), out.tile((2, 5))


def _square_arrangement(x, out):
    return x.tile((4, 4)), out.tile((4, 4))


def _separate_reduction_arrangement(x, rows, columns):
    return x.tile((4, 8)), rows.tile((4,)), columns.tile((8,))


def _row_normalize(x, out):
    maximum = ntl.max(x, axis=1)
    numerator = ntl.exp(x - maximum[:, None])
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def _row_layernorm(x, out):
    width = x.shape[1]
    mean = ntl.sum(x, axis=1) / width
    mean_square = ntl.sum(x * x, axis=1) / width
    variance = mean_square - mean * mean
    out = (x - mean[:, None]) * ntl.rsqrt(variance[:, None] + 1e-5)  # noqa: F841


def _row_min(x, out):
    out = ntl.min(x, axis=1)  # noqa: F841


def _column_max_broadcast(x, out):
    out = x + ntl.max(x, axis=0)[None, :]  # noqa: F841


def _middle_axis_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _incompatible_broadcast(x, out):
    out = x + ntl.sum(x, axis=1)[None, :]  # noqa: F841


def _separate_reduction_outputs(x, rows, columns):
    rows = ntl.sum(x, axis=1)  # noqa: F841
    columns = ntl.max(x, axis=0)  # noqa: F841


def _request():
    return CompileRequest(
        arrangement=_row_arrangement,
        application=_row_normalize,
        tensors=(Tensor(2), Tensor(2)),
        backend="triton",
        tensor_dtypes={"x": "float32", "out": "float32"},
    )


def test_reduction_domain_selects_triton_row_vector_schedule():
    compilation = DEFAULT_COMPILER.compile(_request())
    metadata = compilation.artifact.metadata["ssa_metadata"]
    domains = metadata["analysis"]["reduction_domains"]

    assert len(domains) == 2
    assert {domain["operator"] for domain in domains} == {"max", "sum"}
    assert all(domain["axis"] == 1 for domain in domains)
    assert all(domain["parallel_shape"] == ("1",) for domain in domains)
    assert metadata["schedule"]["reduction"]["mode"] == "row-vector"
    assert tuple(
        candidate["num_warps"]
        for candidate in compilation.launch_plan.tuning_candidates
    ) == (4, 8, 1)
    assert "tl.max(" in compilation.artifact.primary_source
    assert "tl.sum(" in compilation.artifact.primary_source
    assert "for v" not in compilation.artifact.primary_source

    fallback = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_square_arrangement,
            application=_incompatible_broadcast,
            tensors=(Tensor(2), Tensor(2)),
            backend="triton",
            tensor_dtypes={"x": "float32", "out": "float32"},
        )
    )
    assert (
        fallback.artifact.metadata["ssa_metadata"]["schedule"]["reduction"]["mode"]
        == "scalar-fallback"
    )

    with pytest.raises(ValueError, match="separate kernels"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_separate_reduction_arrangement,
                application=_separate_reduction_outputs,
                tensors=(Tensor(2), Tensor(1), Tensor(1)),
                backend="triton",
                tensor_dtypes={
                    "x": "float32",
                    "rows": "float32",
                    "columns": "float32",
                },
            )
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_triton_row_vector_reduction_runtime(device):
    normalize = make(
        _row_arrangement,
        _row_normalize,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    layernorm = make(
        _row_arrangement,
        _row_layernorm,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    reduce_min = make(
        _row_reduced_arrangement,
        _row_min,
        (Tensor(2), Tensor(1)),
        backend="triton",
        max_num_configs=1,
    )
    column_max = make(
        _column_arrangement,
        _column_max_broadcast,
        (Tensor(2), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )
    middle_sum = make(
        _middle_axis_arrangement,
        _middle_axis_sum,
        (Tensor(3), Tensor(2)),
        backend="triton",
        max_num_configs=1,
    )

    width = 127
    base = torch.randn((37, width * 2), device=device)
    x = base[:, ::2]
    normalized = torch.empty_like(x)
    layernorm_output = torch.empty_like(x)
    minimum = torch.empty((x.shape[0],), device=device)

    normalize(x, normalized, WIDTH=width)
    layernorm(x, layernorm_output, WIDTH=width)
    reduce_min(x, minimum, WIDTH=width)

    torch.testing.assert_close(
        normalized,
        torch.softmax(x, dim=1),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        layernorm_output,
        torch.nn.functional.layer_norm(x, (width,)),
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(minimum, x.min(dim=1).values)

    x = torch.randn((193, 41), device=device)
    output = torch.empty_like(x)
    column_max(x, output, HEIGHT=x.shape[0])
    torch.testing.assert_close(output, x + x.max(dim=0).values, rtol=1e-5, atol=1e-6)

    x = torch.randn((2, 3, 5), device=device)
    output = torch.empty((2, 5), device=device)
    middle_sum(x, output)
    torch.testing.assert_close(output, x.sum(dim=1))
