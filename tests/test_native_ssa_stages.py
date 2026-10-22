"""One typed SSA partition and ordinary launch ABI across native targets."""

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.backends.materializers.staged import (
    stage_compilations,
)
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.ir import ssa


def _whole_matrices(x, y, out):
    return tuple(t.tile(tuple(t.shape)) for t in (x, y, out))


def _attention_block(x, y, out):
    score = ntl.dot(x, ntl.trans(y))
    probabilities = ntl.exp(score - ntl.max(score, axis=1)[:, None])
    probabilities = probabilities / ntl.sum(probabilities, axis=1)[:, None]
    out = ntl.dot(probabilities.to(y.dtype), y).to(out.dtype)


def _compile(*, enabled=None, backend="cuda", pipeline=None, backend_options=None):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_whole_matrices,
            application=_attention_block,
            tensors=tuple(
                Tensor(shape=(32, 32), dtype="float16", other=0) for _ in range(3)
            ),
            backend=backend,
            backend_options=backend_options,
            pipeline=pipeline,
            pass_options=None
            if enabled is None
            else {"ssa.plan_materialization": {"enabled": enabled}},
        )
    )


@pytest.mark.parametrize("backend", ("cuda", "bangc", "ascendc"))
def test_native_stages_keep_cut_types_sources_and_ordinary_abi(backend):
    compilation = _compile(backend=backend, enabled=True)
    plan, stages = stage_compilations(compilation)
    assert len(stages) == 3
    assert tuple(tensor.dtype for tensor in plan.scratch) == ("float32", "float16")
    originals = {spec.name: spec for spec in compilation.kernel.tensors}

    if backend == "cuda":
        assert [
            stage.artifact.primary_source.count("wmma::mma_sync(") for stage in stages
        ] == [1, 0, 1]
        assert (
            stages[1].artifact.metadata["ssa_schedule"]["reduction"]["mode"]
            == "row-vector"
        )

    for stage in stages:
        ssa.verify_program(stage.kernel.ssa)
        assert stage.artifact.backend.value == backend
        assert stage.artifact.metadata["launch_plan"]
        assert not stage.artifact.metadata["ssa_metadata"]["materialization"]["enabled"]
        assert set(binding.kind for binding in stage.launch_abi.kernel_args) <= {
            "tensor",
            "stride",
        }

        for spec in stage.kernel.tensors:
            if spec.name in originals:
                assert spec == originals[spec.name]

    assert not any(
        op.opcode.startswith("reduce.")
        for op in stages[-1].kernel.ssa.blocks[0].operations
    )
