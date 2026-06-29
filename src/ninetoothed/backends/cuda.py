"""CUDA backend implementation.

CUDA code generation shares the unified SSA lowering path with the other
backends.  Backend-specific logic is limited to target-language spelling of
SSA operations, buffers, loops, and scalar intrinsics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.base import (
    Backend,
    BackendArtifact,
    BackendCapability,
    BackendName,
    BackendOptions,
)
from ninetoothed.backends.ssa_unified import lower_unified_ssa_artifact
from ninetoothed.ir import KernelIR

if TYPE_CHECKING:
    from ninetoothed.ssa_passes import SSAPassRegistry


class CudaBackend(Backend):
    name = BackendName.CUDA
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; CUDA source is emitted from fine-grained SSA.",
            "No kernel-name or operator-specific dispatch is used in backend lowering.",
        ),
    )

    def lower(
        self, kernel: KernelIR, options: BackendOptions | None = None
    ) -> BackendArtifact:
        return lower_unified_ssa_artifact(kernel, self.name)


def _generic_linear_or_reduction_policy(
    schedule: Mapping[str, Any],
) -> Mapping[str, Any]:
    if schedule.get("granularity") == "parallel-reduction":
        return {
            "passes": ("tree-reduction",),
            "lowering": "ssa-reduction-scf-loop",
        }
    return {
        "passes": ("coalesced-linear-indexing",),
        "lowering": "ssa-operation-linear-emission",
    }


def register_ssa_passes(registry: "SSAPassRegistry") -> None:
    from ninetoothed.ssa_passes import OptimizeSchedulePass

    class CudaOptimizeSchedulePass(OptimizeSchedulePass):
        name = "ssa.cuda.optimize_schedule"
        supported_backends = (BackendName.CUDA,)

        def optimization_policy(
            self,
            backend: BackendName,
            analysis: Mapping[str, Any],
            schedule: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del backend, analysis
            if schedule.get("granularity") == "blocked-linalg":
                return {
                    "passes": ("block-tiling",),
                    "lowering": "thread-block-matmul",
                    "tile": {"block_m": 16, "block_n": 16, "block_k": 8},
                }
            return _generic_linear_or_reduction_policy(schedule)

    registry.register(CudaOptimizeSchedulePass, tags=("optimization", "cuda"))
