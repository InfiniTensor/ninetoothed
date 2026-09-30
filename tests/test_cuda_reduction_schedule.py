import re
from pathlib import Path

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor, bfloat16, float16, float32
from ninetoothed.backends.core import Target
from ninetoothed.backends.cuda import CudaOptimizeSchedule
from ninetoothed.compiler import (
    DEFAULT_COMPILER,
    CompileRequest,
    load_built_artifact,
    make,
)
from ninetoothed.compiler.passes import Context
from tests import test_triton_reduction_schedule as reduction
from tests.utils import requires_backend

pytestmark = requires_backend("cuda")


def _row_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def _row_atomic_arrangement(x, out, total, WIDTH=reduction.WIDTH):
    x = x.tile((1, WIDTH))

    return x, out.tile((1,)), total.tile((-1,)).expand(x.shape)


def _row_sum_with_atomic(x, out, total):
    out = ntl.sum(x, axis=1)  # noqa: F841
    ntl.atomic_add(total.source.data_ptr(), 1.0)


def _candidate_names(extent):
    candidates = CudaOptimizeSchedule().schedule_candidates(
        {},
        {
            "granularity": "parallel-reduction",
            "reduction": {"mode": "row-vector", "extent": extent},
        },
        Context(backend=Target.CUDA, compiler_options={}, kernel_metadata={}),
    )

    return tuple(candidate.name for candidate in candidates)


def test_cuda_cooperative_reduction_schedule_and_source():
    assert _candidate_names(31)[0] == "cooperative-reduction-32"
    assert _candidate_names(127)[0] == "cooperative-reduction-128"
    assert _candidate_names(781)[0] == "cooperative-reduction-256"

    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=reduction._row_arrangement,
            application=reduction._row_normalize,
            tensors=(Tensor(2), Tensor(2)),
            backend="cuda",
            max_num_configs=1,
            pass_options={
                "ssa.cuda.optimize_schedule": {"candidate": "cooperative-reduction-128"}
            },
        )
    )
    source = compilation.artifact.primary_source
    schedule = compilation.artifact.metadata["ssa_schedule"]

    assert schedule["cuda_cooperative_reduction"] is True
    assert schedule["threads"] == 128
    assert tuple(value.render() for value in compilation.launch_plan.block) == ("128",)
    assert "__shfl_down_sync" in source
    assert "__shared__ float" in source
    assert "__syncthreads()" in source
    assert "static_cast<int64_t>(threadIdx.x)" in source
    assert "+= static_cast<int64_t>(blockDim.x)" in source
    assert "blocks =" in source
    assert "(WIDTH + threads - 1) / threads" not in source

    assert not re.search(r"\bv\d+_elem\b", source)

    fallback = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=reduction._row_arrangement,
            application=reduction._row_normalize,
            tensors=(Tensor(2), Tensor(2)),
            backend="cuda",
            max_num_configs=1,
            pass_options={"ssa.cuda.optimize_schedule": {"schedule": {"threads": 48}}},
        )
    )
    fallback_schedule = fallback.artifact.metadata["ssa_schedule"]
    assert fallback_schedule["cuda_cooperative_reduction"] is True
    assert fallback_schedule["threads"] == 48
    assert "__shfl_down_sync" not in fallback.artifact.primary_source

    effect_fallback = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_row_atomic_arrangement,
            application=_row_sum_with_atomic,
            tensors=(Tensor(2), Tensor(1), Tensor(1)),
            backend="cuda",
            max_num_configs=1,
        )
    )
    assert (
        effect_fallback.artifact.metadata["ssa_schedule"]["cuda_cooperative_reduction"]
        is True
    )
    assert "__shfl_down_sync" not in effect_fallback.artifact.primary_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_cooperative_reduction_runtime():
    device = "cuda"
    reduce_min = make(
        reduction._row_reduced_arrangement,
        reduction._row_min,
        (Tensor(2, dtype=float32), Tensor(1, dtype=float32)),
        backend="cuda",
        max_num_configs=1,
    )
    normalize = make(
        reduction._row_arrangement,
        reduction._row_normalize,
        (Tensor(2, dtype=float16), Tensor(2, dtype=float16)),
        backend="cuda",
        max_num_configs=1,
    )
    layernorm = make(
        reduction._row_arrangement,
        reduction._row_layernorm,
        (Tensor(2, dtype=bfloat16), Tensor(2, dtype=bfloat16)),
        backend="cuda",
        max_num_configs=1,
    )
    reduce_sum = make(
        reduction._row_reduced_arrangement,
        _row_sum,
        (Tensor(2, dtype=float32), Tensor(1, dtype=float32)),
        backend="cuda",
        max_num_configs=1,
    )
    column_max = make(
        reduction._column_arrangement,
        reduction._column_max_broadcast,
        (Tensor(2, dtype=float32), Tensor(2, dtype=float32)),
        backend="cuda",
        max_num_configs=1,
    )

    base = torch.randn((37, 254), device=device)
    x = base[:, ::2]
    output = torch.empty((37,), device=device)
    reduce_min(x, output, WIDTH=127)
    torch.testing.assert_close(output, x.min(dim=1).values)

    x = torch.randn((41, 781), device=device, dtype=torch.float16)
    output = torch.empty_like(x)
    normalize(x, output, WIDTH=781)
    torch.testing.assert_close(output, torch.softmax(x, dim=1), rtol=2e-3, atol=2e-3)

    x = torch.randn((29, 1127), device=device, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    layernorm(x, output, WIDTH=1127)
    torch.testing.assert_close(
        output,
        torch.nn.functional.layer_norm(x, (1127,)),
        rtol=2e-2,
        atol=2e-2,
    )

    x = torch.randn((17, 4097), device=device)
    output = torch.empty((17,), device=device)
    reduce_sum(x, output, WIDTH=4097)
    torch.testing.assert_close(output, x.sum(dim=1), rtol=1e-4, atol=1e-4)

    x = torch.randn((193, 41), device=device)
    output = torch.empty_like(x)
    column_max(x, output, HEIGHT=193)
    torch.testing.assert_close(output, x + x.max(dim=0).values)

    x = torch.empty((0, 127), device=device, dtype=torch.float16)
    output = torch.empty_like(x)
    normalize(x, output, WIDTH=127)
    assert output.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_cooperative_reduction_aot_reload(tmp_path):
    device = "cuda"
    tensors = (
        Tensor(shape=(37, 127), dtype=float32),
        Tensor(shape=(37,), dtype=float32),
    )
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=reduction._fixed_row_reduced_arrangement,
            application=reduction._row_min,
            tensors=tensors,
            backend="cuda",
            kernel_name="cuda_cooperative_row_min",
            max_num_configs=1,
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )
    x = torch.randn((37, 127), device=device)

    for launch in (handle, load_built_artifact(handle._built_artifact)):
        output = torch.empty((37,), device=device)
        launch(x, output)
        torch.testing.assert_close(output, x.min(dim=1).values)

    assert Path(handle._built_artifact.binary_path).is_file()
    assert "__shfl_down_sync" in compilation.artifact.primary_source
