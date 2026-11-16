"""AscendC baseline: native serial SSA task execution."""

from ninetoothed.backends.core import Backend, Capability, Target
from ninetoothed.backends.emitters.ascendc import emit
from ninetoothed.backends.toolchain import normalize_ascendc_arch
from ninetoothed.compiler.passes import (
    OptimizeSchedule,
    ScheduleCandidate,
    _pass_options,
)


class AscendCBackend(Backend):
    name = Target.ASCENDC
    supported_options = frozenset({"arch"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Scalar SSA tasks compiled by local CCEC and launched on the current NPU stream; one scalar executor; atomic updates are rejected.",
        ),
    )

    def normalize_options(self, options):
        result = dict(super().normalize_options(options))
        result["arch"] = normalize_ascendc_arch(result.get("arch", "native"))

        return result

    def emit(self, kernel):
        return emit(kernel)


class AscendCOptimizeSchedule(OptimizeSchedule):
    name = "ssa.ascendc.optimize_schedule"
    supported_backends = (Target.ASCENDC,)

    def schedule_candidates(self, analysis, schedule, context):
        del analysis, schedule
        requested = _pass_options(context, self.name, "ssa.optimize_schedule").get(
            "candidate"
        )

        if requested not in {"vector-1024", "vector-4096"}:
            return ()

        return tuple(
            ScheduleCandidate(
                name=f"vector-{elements}",
                schedule={"ascendc_vector_chunk": elements},
                tags=("vector", "explicit"),
            )
            for elements in (1024, 4096)
        )

    def optimization_policy(self, backend, analysis, schedule):
        return {}


def register_ssa_passes(registry):
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry, backend=Target.ASCENDC, optimize_schedule=AscendCOptimizeSchedule
    )
