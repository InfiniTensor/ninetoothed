"""TileLang backend implementation.

TileLang lowering is driven by the same SSA operation stream as Triton, CUDA,
and TVM.  This file intentionally contains no kernel-specialized emitters.
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


class TileLangBackend(Backend):
    name = BackendName.TILELANG
    capability = BackendCapability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TileLang source is emitted from SSA operations.",
            "Backend-specific behavior is expressed as operation-level rendering only.",
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
        "tile": dict(schedule.get("tile", {})),
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
    from ninetoothed.ssa_passes import OptimizeSchedulePass

    class TileLangOptimizeSchedulePass(OptimizeSchedulePass):
        name = "ssa.tilelang.optimize_schedule"
        supported_backends = (BackendName.TILELANG,)

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

    registry.register(TileLangOptimizeSchedulePass, tags=("optimization", "tilelang"))
