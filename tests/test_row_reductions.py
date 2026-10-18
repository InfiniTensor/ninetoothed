from dataclasses import replace
from functools import partial

import pytest
import sympy
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends.triton_reductions import TritonBlockReductions
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    autotune_row_reductions,
)
from ninetoothed.compiler.passes import Context, default_pipeline, default_spec
from ninetoothed.compiler.row_reductions import GroupReductionRows
from tests.utils import get_available_devices

PASS = "ssa.group_reduction_rows"


def _arrangement(x, bias, out, *, rows, block_width):
    x = x.tile((rows, block_width))
    bias = bias.tile((block_width,)).expand((x.shape[0], -1))

    return x, bias, out.tile((rows, block_width))


def _application(x, bias, out):
    value = x.to(ntl.float32)
    numerator = ntl.exp(value - ntl.max(value, axis=1)[:, None])
    normalized = numerator / ntl.sum(numerator, axis=1)[:, None]
    out = normalized + ntl.min(value, axis=1)[:, None] + bias  # noqa: F841


def _request(
    *,
    rows=4,
    width=16,
    dtype="float32",
    enabled=True,
    backend="triton",
    physical_rows=None,
):
    passes = default_spec(backend).passes
    insertion = passes.index("ssa.select_schedule") + 1

    return CompileRequest(
        arrangement=partial(
            _arrangement, rows=rows, block_width=1 << (width - 1).bit_length()
        ),
        application=_application,
        tensors=(
            Tensor(shape=(5, width), dtype=dtype),
            Tensor(shape=(width,), dtype=dtype),
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
    transform = GroupReductionRows()
    assert PASS not in default_spec(backend).passes

    for rows in (1, 4):
        compilation = DEFAULT_COMPILER.compile(_request(rows=rows, backend=backend))
        schedule = compilation.artifact.metadata["ssa_schedule"]
        assert schedule["row_group"]["enabled"] == (rows > 1)

        if rows > 1:
            assert tuple(
                int(sympy.sympify(x.render())) for x in compilation.launch_plan.grid
            ) == (2,)

        context = Context(
            backend=compilation.target.backend,
            compiler_options={},
            kernel_metadata={},
            tensors=compilation.kernel.tensors,
        )
        program = default_pipeline(backend).run(compilation.kernel.ssa, context)
        selected = transform.run(program, context)
        assert transform.run(selected, context) == selected

        if backend == "triton":
            assert (
                not TritonBlockReductions()
                .run(program, context)
                .metadata["schedule"]["block_reductions"]["enabled"]
            )

    if backend == "triton":
        limited = DEFAULT_COMPILER.compile(
            replace(
                _request(backend=backend),
                pass_options={"ssa.triton.block_reductions": {"max_block_elements": 8}},
            )
        )
        assert not limited.artifact.metadata["ssa_schedule"]["row_group"]["enabled"]

    for physical_rows in (1, 2, 4):
        compilation = DEFAULT_COMPILER.compile(
            _request(backend=backend, physical_rows=physical_rows)
        )
        assert (
            compilation.artifact.metadata["ssa_schedule"]["row_group"]["rows"]
            == physical_rows
        )
        assert tuple(
            int(sympy.sympify(x.render())) for x in compilation.launch_plan.grid
        ) == (8 // physical_rows,)


@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
@pytest.mark.parametrize("device", get_available_devices())
def test_row_reduction_runtime(device, backend):
    if backend == "tilelang":
        pytest.importorskip("tilelang")

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for width in (4, 13):
            x = torch.randn(5, width * 2, dtype=dtype, device=device)[:, ::2]
            x[0].abs_().add_(1)
            x[1].abs_().neg_().sub_(1)
            bias = torch.randn(width * 2, dtype=dtype, device=device)[::2]
            output = torch.empty_strided(
                (5, width), (width * 3, 2), dtype=dtype, device=device
            )
            expected = (
                torch.softmax(x.double(), dim=1)
                + x.double().amin(dim=1, keepdim=True)
                + bias.double()
            ).to(dtype)

            for enabled, physical_rows in ((False, None), (True, None), (True, 2)):
                compilation = DEFAULT_COMPILER.compile(
                    _request(
                        width=width,
                        dtype=str(dtype).split(".")[-1],
                        enabled=enabled,
                        backend=backend,
                        physical_rows=physical_rows,
                    )
                )

                if enabled:
                    schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
                    assert schedule["row_group"]["enabled"]
                    assert PASS in compilation.pass_trace

                DEFAULT_COMPILER.materialize(compilation)(x, bias, output)
                torch.testing.assert_close(output, expected)


def _shape_application(x, bias, out):
    value = x.to(ntl.float32)
    out = ntl.sum(value, axis=1)[:, None] + value / x.shape[0] + bias  # noqa: F841


@pytest.mark.parametrize("backend", ("triton", "cuda", "tilelang"))
@pytest.mark.parametrize("device", get_available_devices())
def test_row_tuning_preserves_logical_tile(device, backend):
    if backend == "tilelang":
        pytest.importorskip("tilelang")

    x = torch.randn(5, 13, device=device)
    bias = torch.randn(13, device=device)
    output = torch.empty_like(x)
    request = replace(
        _request(rows=8, width=13, backend=backend), application=_shape_application
    )
    tuner = autotune_row_reductions(request, (x, bias, output))
    torch.testing.assert_close(output, x.sum(1, keepdim=True) + x / 8 + bias)
    assert {1, 8} <= set(tuner.search["rows"].values())
    x.mul_(2)
    tuner(x, bias, output)
    torch.testing.assert_close(output, x.sum(1, keepdim=True) + x / 8 + bias)
