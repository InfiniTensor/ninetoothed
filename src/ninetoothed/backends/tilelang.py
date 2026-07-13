"""TileLang backend implementation.

TileLang lowering is driven by the same SSA operation stream as Triton and
CUDA.  This file intentionally contains no kernel-specialized emitters.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.tilelang import emit
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TileLangBackend(Backend):
    name = Target.TILELANG
    supported_options = frozenset({"max_threads_per_block"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; TileLang source is emitted from SSA operations.",
            "Backend-specific behavior is expressed as operation-level rendering only.",
        ),
    )

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class TileLangOptimizeSchedule(OptimizeSchedule):
    name = "ssa.tilelang.optimize_schedule"
    supported_backends = (Target.TILELANG,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del context

        if schedule.get("granularity") != "blocked-linalg" or not analysis.get(
            "dot_supports_low_precision_intrinsic", False
        ):
            return ()
        return (
            ScheduleCandidate(
                name="balanced",
                schedule={
                    "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                    "threads": 128,
                    "num_stages": 2,
                },
                tags=("default",),
            ),
            ScheduleCandidate(
                name="small",
                schedule={
                    "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                    "threads": 128,
                    "num_stages": 2,
                },
                tags=("small-problem",),
            ),
            ScheduleCandidate(
                name="wide",
                schedule={
                    "tile": {"block_m": 128, "block_n": 64, "block_k": 32},
                    "threads": 256,
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

        if schedule.get("granularity") == "blocked-linalg":
            return {
                "preserve_linalg": bool(
                    analysis.get("dot_supports_low_precision_intrinsic", False)
                )
            }

        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TILELANG,
        optimize_schedule=TileLangOptimizeSchedule,
    )
