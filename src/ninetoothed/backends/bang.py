"""BANG backend implementation.

BANG lowering is driven by the same SSA operation stream as Triton,
CUDA, and TileLang.  This file intentionally contains no
kernel-specialized emitters.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.bang import emit
from ninetoothed.backends.toolchain import normalize_bang_arch
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class BangBackend(Backend):
    name = Target.BANG
    supported_options = frozenset({"arch"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; BANG C++ source is emitted from SSA operations.",
            "Correctness-first flat lowering: one BANG block task serially "
            "processes the scheduled thread count as consecutive elements; "
            "NRAM staging and vector intrinsics are future schedule work.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))
        normalized["arch"] = normalize_bang_arch(normalized.get("arch", "native"))

        return normalized

    def emit(self, kernel: Kernel):
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class BangOptimizeSchedule(OptimizeSchedule):
    name = "ssa.bang.optimize_schedule"
    supported_backends = (Target.BANG,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del analysis, schedule, context

        return ()

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis, schedule

        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.BANG,
        optimize_schedule=BangOptimizeSchedule,
    )
