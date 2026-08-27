"""BangC backend implementation.

BangC code generation shares the unified SSA lowering path with the other
backends.  The target model maps one BANG task per output chunk: the generic
flat-index domain is rendered as a scalar loop inside ``__mlu_entry__``
kernels, and artifacts are compiled with the Cambricon ``cncc`` toolchain.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.bangc import emit
from ninetoothed.backends.toolchain import normalize_bangc_arch
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class BangCBackend(Backend):
    name = Target.BANGC
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
            "Unified SSA backend; BangC source is emitted from fine-grained SSA.",
            "Kernels use the BANG task model (one task per output chunk) with a "
            "cncc-compiled C launcher bound to cnrtQueue_t streams.",
            "Pure float32 elementwise kernels over contiguous 1-D layouts are "
            "auto-staged into __nram__ buffers with __memcpy and, where the "
            "op maps directly, __bang_* tensor instructions (measured "
            "~1.2 TB/s on MLU590 vs ~0.5 GB/s for the scalar fallback).",
            "Row-vector reductions use one task per parallel slice with an "
            "in-task serial loop (cooperative-reduction schedule).",
            "Known limitations on MLU590: hardware scalar atomics trap at "
            "runtime (every __bang_atomic_add spelling and raw asm), so "
            "kernels containing mem.atomic_add are serialized onto one task "
            "that walks the full flat domain (exact, not fast); 64-bit "
            "immediate constants are truncated on the device scalar path; "
            "fp16 matmul differs from CNNL by at most one ulp because the "
            "reference uses tensor-compute accumulation orders that scalar "
            "code cannot reproduce bit-exactly; multi-domain online "
            "reductions (flash attention with long sequences) fall back to "
            "the O(n^2) scalar lowering and may exceed the kernel execution "
            "timeout.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))
        arch = normalize_bangc_arch(normalized.get("arch", "native"))
        normalized["arch"] = arch

        if "task_chunk" in normalized:
            chunk = int(normalized["task_chunk"])

            if chunk <= 0:
                raise ValueError(
                    "The BangC `task_chunk` backend option must be positive."
                )

            normalized["task_chunk"] = chunk

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class BangCOptimizeSchedule(OptimizeSchedule):
    name = "ssa.bangc.optimize_schedule"
    supported_backends = (Target.BANGC,)

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
            # One BANG task per parallel slice; the full extent is reduced
            # serially inside the task, keeping the work O(n) per output row.
            return (
                ScheduleCandidate(
                    name="cooperative-reduction",
                    schedule={"bangc_cooperative_reduction": True},
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
            # truncate the accumulator to float16/bfloat16 every iteration.
            preserve_linalg = bool(
                analysis.get("dot_supports_low_precision_intrinsic", False)
            )

            return {"preserve_linalg": preserve_linalg}
        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.BANGC,
        optimize_schedule=BangCOptimizeSchedule,
    )
