from dataclasses import replace
from functools import partial

import pytest
import sympy
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    autotune_row_reductions,
)
from ninetoothed.compiler.passes import default_spec
from tests.utils import get_available_devices

PASS = "ssa.group_reduction_rows"


def _arrangement(x, bias, out, *, rows, block_width, bias_axis=1):
    x = x.tile((rows, block_width))

    if bias_axis == 0:
        bias = bias.tile((rows,)).unsqueeze(1).expand((-1, x.shape[1]))
    else:
        bias = bias.tile((block_width,)).expand((x.shape[0], -1))

    return x, bias, out.tile((rows, block_width))


def _application(x, bias, out):
    value = x.to(ntl.float32)
    numerator = ntl.exp(value - ntl.max(value, axis=1)[:, None])
    normalized = numerator / ntl.sum(numerator, axis=1)[:, None]
    out = normalized + ntl.min(value, axis=1)[:, None]  # noqa: F841


def _row_bias_application(x, bias, out):
    out = x + ntl.sum(x, axis=1)[:, None] + bias[:, None]  # noqa: F841


def _column_application(x, bias, out):
    out = x + ntl.sum(x, axis=0) + bias  # noqa: F841


def _input_bias_application(x, bias, out):
    out = x + ntl.sum(x + bias, axis=1)[:, None]  # noqa: F841


def _atomic_bias_application(x, bias, out):
    out = x + ntl.sum(x, axis=1)[:, None]  # noqa: F841
    ntl.atomic_add(bias.source.data_ptr() + 1, 1.0)


def _request(
    *,
    rows=4,
    width=16,
    dtype="float32",
    enabled=True,
    backend="triton",
    physical_rows=None,
    bias_axis=1,
):
    passes = default_spec(backend).passes
    insertion = passes.index("ssa.select_schedule") + 1

    return CompileRequest(
        arrangement=partial(
            _arrangement,
            rows=rows,
            block_width=1 << (width - 1).bit_length(),
            bias_axis=bias_axis,
        ),
        application=_application if bias_axis == 1 else _row_bias_application,
        tensors=(
            Tensor(shape=(5, width), dtype=dtype),
            Tensor(shape=(width if bias_axis == 1 else 5,), dtype=dtype),
            Tensor(shape=(5, width), dtype=dtype),
        ),
        backend=backend,
        pipeline=(*passes[:insertion], PASS, *passes[insertion:]) if enabled else None,
        pass_options={PASS: {"rows": physical_rows}}
        if physical_rows is not None
        else None,
        num_warps=4,
        max_num_configs=1,
    )


@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
def test_row_reduction_selection_preserves_fallback(backend):
    assert PASS not in default_spec(backend).passes
    single = DEFAULT_COMPILER.compile(_request(rows=1, backend=backend))
    assert not single.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]

    for physical_rows in (1, 2, 4):
        compilation = DEFAULT_COMPILER.compile(
            _request(
                backend=backend,
                physical_rows=physical_rows,
                width=512 if backend == "tilelang" else 16,
            )
        )
        group = compilation.artifact.metadata["ssa_schedule"]["row_group"]
        assert group["enabled"] and group["rows"] == physical_rows
        assert tuple(
            int(sympy.sympify(x.render())) for x in compilation.launch_plan.grid
        ) == (8 // physical_rows,)

    mismatched = DEFAULT_COMPILER.compile(
        replace(
            _request(width=4, backend=backend),
            application=_input_bias_application,
            tensors=(
                Tensor(shape=(5, 4), dtype="float32"),
                Tensor(shape=(3,), dtype="float32"),
                Tensor(shape=(5, 4), dtype="float32"),
            ),
        )
    )
    assert not mismatched.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]

    if backend == "triton":
        limited = DEFAULT_COMPILER.compile(
            replace(
                _request(backend=backend),
                pass_options={"ssa.triton.block_reductions": {"max_block_elements": 8}},
            )
        )
        assert not limited.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]


@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
@pytest.mark.parametrize("device", get_available_devices())
def test_row_reduction_runtime(device, backend):
    if backend == "tilelang":
        pytest.importorskip("tilelang")

    for dtype, width, configurations in (
        (torch.float32, 16, ((False, None),)),
        (torch.float32, 13, ((True, None), (True, 2))),
        (torch.float16, 13, ((True, None),)),
        (torch.bfloat16, 13, ((True, None),)),
    ):
        x = torch.randn(5, width * 2, dtype=dtype, device=device)[:, ::2]
        bias = torch.randn(width * 2, dtype=dtype, device=device)[::2]
        output = torch.empty_strided(
            (5, width), (width * 3, 2), dtype=dtype, device=device
        )
        expected = (
            torch.softmax(x.double(), dim=1) + x.double().amin(dim=1, keepdim=True)
        ).to(dtype)

        for enabled, physical_rows in configurations:
            compilation = DEFAULT_COMPILER.compile(
                _request(
                    width=width,
                    dtype=str(dtype).removeprefix("torch."),
                    enabled=enabled,
                    backend=backend,
                    physical_rows=physical_rows,
                )
            )

            if enabled:
                assert compilation.artifact.metadata["ssa_schedule"]["row_group"][
                    "enabled"
                ]
                assert PASS in compilation.pass_trace

            output.fill_(float("nan"))
            DEFAULT_COMPILER.materialize(compilation)(x, bias, output)
            torch.testing.assert_close(output, expected)

    x = torch.randn(5, width, device=device)
    bias = torch.randn(5, device=device)
    output = torch.empty_like(x)
    request = _request(width=width, backend=backend, physical_rows=2, bias_axis=0)
    compilation = DEFAULT_COMPILER.compile(request)
    assert compilation.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]
    DEFAULT_COMPILER.materialize(compilation)(x, bias, output)
    torch.testing.assert_close(output, x + x.sum(1, keepdim=True) + bias[:, None])

    x = torch.randn(5, 4, device=device)
    bias = torch.randn(4, device=device)
    output = torch.empty_like(x)
    column_sums = torch.cat(
        [tile.sum(0, keepdim=True).expand_as(tile) for tile in x.split(4)]
    )
    request = replace(
        _request(width=4, backend=backend, physical_rows=2),
        application=_column_application,
    )
    compilation = DEFAULT_COMPILER.compile(request)
    assert compilation.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]
    DEFAULT_COMPILER.materialize(compilation)(x, bias, output)
    torch.testing.assert_close(output, x + column_sums + bias)


def _shape_application(x, bias, out):
    value = x.to(ntl.float32)
    out = ntl.sum(value, axis=1)[:, None] + value / x.shape[0]  # noqa: F841


def _read_output_application(x, bias, out):
    out = ntl.sum(x, axis=1)[:, None] + out + bias  # noqa: F841


@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
@pytest.mark.parametrize("device", get_available_devices())
def test_row_tuning_preserves_logical_tile(device, backend, monkeypatch):
    if backend == "tilelang":
        pytest.importorskip("tilelang")

    x = torch.randn(5, 13, device=device)
    bias = torch.randn(13, device=device)
    output = torch.empty_like(x)
    request = replace(
        _request(rows=8, width=13, backend=backend),
        application=_shape_application,
        backend=None,
        pipeline=None,
    )
    monkeypatch.setenv("NINETOOTHED_BACKEND", backend)
    tuner = autotune_row_reductions(request, (x, bias, output))
    torch.testing.assert_close(output, x.sum(1, keepdim=True) + x / 8)
    assert {1, 8} <= set(tuner.search["rows"].values())
    x.mul_(2)
    tuner(x, bias, output)
    torch.testing.assert_close(output, x.sum(1, keepdim=True) + x / 8)

    for application in (_read_output_application, _atomic_bias_application):
        before = tuple(value.clone() for value in (x, bias, output))

        with pytest.raises(ValueError, match="write-only"):
            autotune_row_reductions(
                replace(request, application=application), (x, bias, output)
            )

        for value, saved in zip((x, bias, output), before):
            torch.testing.assert_close(value, saved)
