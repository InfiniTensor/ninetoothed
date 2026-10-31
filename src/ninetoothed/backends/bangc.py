"""BangC baseline: native serial SSA task execution."""

from ninetoothed.backends.core import Backend, Capability, Target
from ninetoothed.backends.emitters.bangc import emit
from ninetoothed.backends.toolchain import normalize_bangc_arch
from ninetoothed.compiler.passes import OptimizeSchedule


class BangCBackend(Backend):
    name = Target.BANGC
    supported_options = frozenset({"arch"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Scalar SSA tasks compiled by local CNCC and launched on the current MLU queue; software 64-bit data arithmetic; atomic updates serialized across the full domain.",
        ),
    )

    def normalize_options(self, options):
        result = dict(super().normalize_options(options))
        result["arch"] = normalize_bangc_arch(result.get("arch", "native"))

        return result

    def emit(self, kernel):
        return emit(kernel)


class BangCOptimizeSchedule(OptimizeSchedule):
    name = "ssa.bangc.optimize_schedule"
    supported_backends = (Target.BANGC,)

    def optimization_policy(self, backend, analysis, schedule):
        return {}


def register_ssa_passes(registry):
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry, backend=Target.BANGC, optimize_schedule=BangCOptimizeSchedule
    )
