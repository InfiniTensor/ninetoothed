"""CUDA backend implementation.

CUDA code generation shares the unified SSA lowering path with the other
backends.  Backend-specific logic is limited to target-language spelling of
SSA operations, buffers, loops, and scalar intrinsics.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.cuda import emit
from ninetoothed.backends.toolchain import (
    cuda_compute_capability,
    normalize_cuda_arch,
)
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class CudaBackend(Backend):
    name = Target.CUDA
    supported_options = frozenset(
        {
            "arch",
            "compute_capability",
            "max_shared_memory_bytes",
            "max_threads_per_block",
        }
    )
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; CUDA source is emitted from fine-grained SSA.",
            "No kernel-name or operator-specific dispatch is used in backend lowering.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))
        arch = normalize_cuda_arch(normalized.get("arch", "native"))
        normalized["arch"] = arch

        if "compute_capability" not in normalized:
            capability = cuda_compute_capability(arch)

            if capability is not None:
                normalized["compute_capability"] = capability

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class CudaOptimizeSchedule(OptimizeSchedule):
    name = "ssa.cuda.optimize_schedule"
    supported_backends = (Target.CUDA,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        platform_cuda = context.resolved_target.platform.metadata.get("cuda", {})

        reduction = schedule.get("reduction", {})

        if (
            schedule.get("granularity") == "parallel-reduction"
            and isinstance(reduction, Mapping)
            and reduction.get("mode") == "row-vector"
        ):
            extent = _static_extent(reduction.get("extent"))

            if extent is not None and extent <= 32:
                thread_counts = (32, 128, 256)
            elif extent is not None and extent <= 128:
                thread_counts = (128, 256, 32)
            else:
                thread_counts = (256, 128, 32)

            return tuple(
                ScheduleCandidate(
                    name=f"cooperative-reduction-{threads}",
                    schedule={
                        "cuda_cooperative_reduction": True,
                        "threads": threads,
                    },
                    tags=("cooperative-reduction",),
                )
                for threads in thread_counts
            )

        if (
            schedule.get("granularity") != "blocked-linalg"
            or not analysis.get("dot_supports_low_precision_intrinsic", False)
            or platform_cuda.get("wmma") is False
        ):
            return ()

        mma = {"m": 16, "n": 16, "k": 16}

        return (
            ScheduleCandidate(
                name="wmma-16x16",
                schedule={
                    "tile": {"block_m": 16, "block_n": 16, "block_k": 16},
                    "mma_shape": mma,
                    "threads": 256,
                },
                constraints={
                    "dtypes": ("float16", "bfloat16"),
                    "minimum_compute_capability": "7.0",
                    "shared_memory_bytes": 2048,
                },
                tags=("default", "wmma"),
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
            preserve_linalg = bool(
                analysis.get("dot_supports_low_precision_intrinsic", False)
                and schedule.get("mma_shape") == {"m": 16, "n": 16, "k": 16}
            )

            return {"preserve_linalg": preserve_linalg}
        return {}


def _static_extent(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.CUDA,
        optimize_schedule=CudaOptimizeSchedule,
    )
