"""AscendC backend implementation.

AscendC code generation shares the unified SSA lowering path with the other
backends.  The target model maps one AI-core block per output chunk on
Ascend 910B devices: kernels are ``__global__ __aicore__`` task programs
with GM_ADDR device pointers, and artifacts are compiled with the CANN
``ccec`` compiler into shared libraries exposing the standard C-ABI
launcher bound to ``aclrtStream`` streams.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.ascendc import emit
from ninetoothed.backends.toolchain import normalize_ascendc_arch
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class AscendCBackend(Backend):
    name = Target.ASCENDC
    supported_options = frozenset(
        {
            "arch",
            "task_chunk",
        }
    )
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; AscendC source is emitted from fine-grained SSA.",
            "Kernels use the AscendC block model (one block per output chunk) "
            "with a ccec-compiled C launcher bound to aclrtStream streams.",
            "Row-vector reductions use one block per parallel slice with an "
            "in-block serial loop (cooperative-reduction schedule).",
            "Scalar transcendental functions are software-implemented in the "
            "generated kernels because AscendC exposes no scalar built-ins "
            "beyond sqrt/max/min.",
            "Known limitations: scalar atomics are unavailable and the "
            "runtime does not guarantee exclusive block execution, so "
            "kernels containing mem.atomic_add are rejected at emission; "
            "reduction store targets must keep the dimensionality of the "
            "reduction domain; bfloat16 and float8 dtypes are rejected in "
            "the scalar lowering; pure elementwise kernels are not yet "
            "staged through LocalTensor/TPipe pipelines.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))
        arch = normalize_ascendc_arch(normalized.get("arch", "native"))
        normalized["arch"] = arch

        if "task_chunk" in normalized:
            chunk = int(normalized["task_chunk"])

            if chunk <= 0:
                raise ValueError(
                    "The AscendC `task_chunk` backend option must be positive."
                )

            normalized["task_chunk"] = chunk

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class AscendCOptimizeSchedule(OptimizeSchedule):
    name = "ssa.ascendc.optimize_schedule"
    supported_backends = (Target.ASCENDC,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        reduction = schedule.get("reduction", {})

        if (
            schedule.get("granularity") == "parallel-reduction"
            and isinstance(reduction, Mapping)
            and reduction.get("mode") == "row-vector"
        ):
            # One AscendC block per parallel slice; the full extent is
            # reduced serially inside the block.  Multi-tile reduction
            # axes keep per-tile semantics, matching the tiled program
            # model of the other backends.
            return (
                ScheduleCandidate(
                    name="cooperative-reduction",
                    schedule={"ascendc_cooperative_reduction": True},
                    tags=("cooperative-reduction",),
                ),
            )

        return ()

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend

        if schedule.get("granularity") == "blocked-linalg":
            # Low-precision dots must keep the shared linalg.dot lowering so
            # accumulation happens in float32; decomposed scf.for loops would
            # truncate the accumulator to float16 every iteration.
            preserve_linalg = bool(
                analysis.get("dot_supports_low_precision_intrinsic", False)
            )

            return {"preserve_linalg": preserve_linalg}
        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.ASCENDC,
        optimize_schedule=AscendCOptimizeSchedule,
    )
