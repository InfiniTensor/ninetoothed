"""Collective dot candidates retain the source region and access permissions."""

import functools

import pytest

from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from tests.test_ascendc_dot_plan import _matmul
from tests.test_matmul import arrangement


def _compile(
    application=_matmul,
    *,
    mode="fused",
    arch="sm_89",
    num_warps=None,
    candidate="wmma-32x32",
):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
            ),
            application=application,
            tensors=(
                Tensor(shape=(64, 64), dtype="float16", other=0),
                Tensor(shape=(64, 64), dtype="float16", other=0),
                Tensor(shape=(64, 64), dtype="float32", other=0),
            ),
            backend="cuda",
            backend_options={"arch": arch, "cuda_wmma_accumulation": mode},
            num_warps=num_warps,
            pass_options=(
                {"ssa.cuda.optimize_schedule": {"candidate": candidate}}
                if candidate is not None
                else None
            ),
        )
    )


@pytest.mark.parametrize("mode", ("fused", "ordered"))
def test_collective_precedes_output_iteration_and_owns_disjoint_fragments(mode):
    artifact = _compile(mode=mode).artifact

    source = artifact.primary_source

    assert source.index("cp.async.wait_group 0") < source.index(
        "wmma::store_matrix_sync"
    )
    assert source.index("wmma::store_matrix_sync") < source.index("nt_cta_item = 0")
    assert "__syncthreads" not in source.split("nt_cta_item = 0", 1)[1]
    assert ("__fadd_rn" in source) is (mode == "ordered")
    assert artifact.metadata["cuda_cta_dot"]["shared_memory_bytes"] == 5632
    assert "_dense_wmma && reinterpret_cast<uintptr_t>" in source
    assert "cp.async.cg.shared.global" in source
    assert "__float2half_rn" in source
    assert "_copy = threadIdx.x" in source
    assert artifact.metadata["cuda_wmma_accumulation"][
        "source_dot_add_groups_preserved"
    ] is (mode == "ordered")
