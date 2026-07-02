"""CUDA backend implementation.

CUDA code generation shares the unified SSA lowering path with the other
backends.  Backend-specific logic is limited to target-language spelling of
SSA operations, buffers, loops, and scalar intrinsics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Options,
    Target,
)
from ninetoothed.backends.emitters.ssa import emit
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class CudaBackend(Backend):
    name = Target.CUDA
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; CUDA source is emitted from fine-grained SSA.",
            "No kernel-name or operator-specific dispatch is used in backend lowering.",
        ),
    )

    def emit(self, kernel: Kernel, options: Options | None = None) -> Artifact:
        return emit(kernel, self.name)


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


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.compiler.passes import (
        Context,
        LowerIntrinsics,
        LowerMemoryScopes,
        OptimizeSchedule,
    )

    class CudaOptimizeSchedule(OptimizeSchedule):
        name = "ssa.cuda.optimize_schedule"
        supported_backends = (Target.CUDA,)

        def optimization_policy(
            self,
            backend: Target,
            analysis: Mapping[str, Any],
            schedule: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del backend, analysis

            if schedule.get("granularity") == "blocked-linalg":
                return {
                    "passes": ("block-tiling",),
                    "lowering": "thread-block-matmul",
                    "schedule": {"tile": {"block_m": 16, "block_n": 16, "block_k": 8}},
                }
            return _generic_linear_or_reduction_policy(schedule)

    registry.register(CudaOptimizeSchedule, tags=("optimization", "cuda"))

    class CudaLowerMemoryScopesPass(LowerMemoryScopes):
        name = "ssa.cuda.lower_memory_scopes"
        supported_backends = (Target.CUDA,)

        def memory_scopes(self, context: Context) -> Mapping[str, str]:
            del context

            return {
                "register": "thread-local",
                "shared": "__shared__",
                "global": "__global__ pointer",
            }

    class CudaLowerIntrinsicsPass(LowerIntrinsics):
        name = "ssa.cuda.lower_intrinsics"
        supported_backends = (Target.CUDA,)

        def intrinsics(self, context: Context) -> Mapping[str, str]:
            del context

            return {
                "dot": "thread loop or mma.sync candidate",
                "exp": "__expf/expf",
                "program_id": "blockIdx/threadIdx",
                "load_store": "pointer load/store",
            }

    registry.register(
        CudaLowerMemoryScopesPass, tags=("target-lowering", "memory", "cuda")
    )
    registry.register(
        CudaLowerIntrinsicsPass, tags=("target-lowering", "intrinsics", "cuda")
    )
