from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import Artifact, Backend, Capability, Target
from ninetoothed.backends.emitters.ascend import AscendEmitter
from ninetoothed.compiler.passes import Context, OptimizeSchedule, ScheduleCandidate
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class AscendOptimizeSchedule(OptimizeSchedule):
    name = "ssa.ascend.optimize_schedule"
    supported_backends = (Target.ASCEND,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        return ()

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.ASCEND,
        optimize_schedule=AscendOptimizeSchedule,
    )


class AscendBackend(Backend):
    name = Target.ASCEND
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=("Python/Triton source for the Ascend Triton runtime; JIT and AOT.",),
    )

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        return kernel

    def emit(self, kernel: Kernel) -> Artifact:
        emitter = AscendEmitter()
        return emitter.emit(kernel)
