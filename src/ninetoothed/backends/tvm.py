"""TVM backend implementation.

TVMScript generation is handled by the unified SSA emitter.  The backend class
is deliberately small so that future optimizations enter through SSA passes,
not through kernel-specialized lowerer branches.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Options,
    Target,
)
from ninetoothed.backends.emitters.tvm import emit
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TvmBackend(Backend):
    name = Target.TVM
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TVMScript is emitted from fine-grained SSA.",
            "No operator-specific emitters are used by the backend entrypoint.",
        ),
    )

    def emit(self, kernel: Kernel, options: Options | None = None) -> Artifact:
        return emit(kernel, options)


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


class TvmOptimizeSchedule(OptimizeSchedule):
    name = "ssa.tvm.optimize_schedule"
    supported_backends = (Target.TVM,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del analysis, context

        if schedule.get("granularity") != "blocked-linalg":
            return ()
        return (
            ScheduleCandidate(
                name="dlight-default",
                schedule={"tile": {"block_m": 64, "block_n": 64, "block_k": 32}},
                tags=("default", "dlight"),
            ),
            ScheduleCandidate(
                name="dlight-small",
                schedule={"tile": {"block_m": 32, "block_n": 32, "block_k": 32}},
                tags=("small-problem", "dlight"),
            ),
            ScheduleCandidate(
                name="dlight-wide",
                schedule={"tile": {"block_m": 128, "block_n": 64, "block_k": 32}},
                tags=("throughput", "dlight"),
            ),
        )

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis

        if schedule.get("granularity") == "blocked-linalg":
            return _scheduled_tir_loop_policy(schedule)
        return _generic_linear_or_reduction_policy(schedule)


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TVM,
        optimize_schedule=TvmOptimizeSchedule,
    )
