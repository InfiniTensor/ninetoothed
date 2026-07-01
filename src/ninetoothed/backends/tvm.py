"""TVM backend implementation.

TVMScript generation is handled by the unified SSA emitter.  The backend class
is deliberately small so that future optimizations enter through SSA passes,
not through kernel-specialized lowerer branches.
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


class TvmBackend(Backend):
    name = BackendName.TVM
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TVMScript is emitted from fine-grained SSA.",
            "No operator-specific emitters are used by the backend entrypoint.",
        ),
    )

    def lower(
        self, kernel: KernelIR, options: BackendOptions | None = None
    ) -> BackendArtifact:
        return lower_unified_ssa_artifact(kernel, self.name)


def _scheduled_tir_loop_policy(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "passes": ("block-tiling", "loop-reorder"),
        "lowering": "scheduled-tir-loops",
        "schedule": {"tile": dict(schedule.get("tile", {}))},
    }


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
    from ninetoothed.ssa_passes import (
        BackendIntrinsicsLoweringPass,
        BackendMemoryScopesLoweringPass,
        BackendScheduleOptimizationPass,
        SSAPassContext,
    )

    class TvmOptimizeSchedulePass(BackendScheduleOptimizationPass):
        name = "ssa.tvm.optimize_schedule"
        supported_backends = (BackendName.TVM,)

        def optimization_policy(
            self,
            backend: BackendName,
            analysis: Mapping[str, Any],
            schedule: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del backend, analysis

            if schedule.get("granularity") == "blocked-linalg":
                return _scheduled_tir_loop_policy(schedule)
            return _generic_linear_or_reduction_policy(schedule)

    registry.register(TvmOptimizeSchedulePass, tags=("optimization", "tvm"))

    class TvmLowerMemoryScopesPass(BackendMemoryScopesLoweringPass):
        name = "ssa.tvm.lower_memory_scopes"
        supported_backends = (BackendName.TVM,)

        def memory_scopes(self, context: SSAPassContext) -> Mapping[str, str]:
            del context

            return {
                "register": "local",
                "shared": "shared",
                "global": "global",
            }

    class TvmLowerIntrinsicsPass(BackendIntrinsicsLoweringPass):
        name = "ssa.tvm.lower_intrinsics"
        supported_backends = (BackendName.TVM,)

        def intrinsics(self, context: SSAPassContext) -> Mapping[str, str]:
            del context

            return {
                "dot": "TIR loop or tensorize candidate",
                "exp": "T.exp",
                "program_id": "T.thread_binding",
                "load_store": "T.match_buffer/T.BufferStore",
            }

    registry.register(
        TvmLowerMemoryScopesPass, tags=("target-lowering", "memory", "tvm")
    )
    registry.register(
        TvmLowerIntrinsicsPass, tags=("target-lowering", "intrinsics", "tvm")
    )
