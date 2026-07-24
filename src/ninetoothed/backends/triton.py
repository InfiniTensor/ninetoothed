"""Triton backend implementation.

The backend lowerer is intentionally SSA-first.  It does not classify kernels
by operator family before emitting code; it delegates to the unified SSA
emitter, whose dispatch unit is a single SSA operation.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.triton import emit
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TritonBackend(Backend):
    name = Target.TRITON
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; Triton source is generated from SSA operations.",
            "No source passthrough or kernel-specialized fallback is used.",
        ),
    )

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)


class TritonOptimizeSchedule(OptimizeSchedule):
    name = "ssa.triton.optimize_schedule"
    supported_backends = (Target.TRITON,)

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
                name="balanced",
                schedule={
                    "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                    "num_warps": 4,
                    "num_stages": 3,
                },
                tags=("default", "tensor-core"),
            ),
            ScheduleCandidate(
                name="small",
                schedule={
                    "tile": {"block_m": 16, "block_n": 16, "block_k": 32},
                    "num_warps": 4,
                    "num_stages": 2,
                },
                constraints={"max_m": 128, "max_n": 128, "max_k": 128},
                tags=("small-problem",),
            ),
            ScheduleCandidate(
                name="wide",
                schedule={
                    "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                    "num_warps": 8,
                    "num_stages": 3,
                },
                tags=("throughput",),
            ),
        )

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend
        granularity = str(schedule.get("granularity", "elementwise-grid"))

        if granularity == "exp-reduction-dot-region":
            return {
                "schedule": {"num_warps": 4, "num_stages": 2},
            }

        if granularity == "blocked-linalg":
            preserve_linalg = bool(
                analysis.get("dot_supports_low_precision_intrinsic", False)
            )

            return {
                "preserve_linalg": preserve_linalg,
            }

        if granularity == "parallel-reduction":
            reduction = schedule.get("reduction", {})

            if reduction.get("mode") == "row-vector":
                return {"schedule": {"num_warps": (4, 8, 1), "num_stages": 1}}
            return {"schedule": {"num_warps": 4}}
        return {"schedule": {"num_warps": 4}}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TRITON,
        optimize_schedule=TritonOptimizeSchedule,
    )
