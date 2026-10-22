"""CUDA backend implementation.

CUDA code generation shares the unified SSA lowering path with the other
backends.  Backend-specific logic is limited to target-language spelling of
SSA operations, buffers, loops, and scalar intrinsics.
"""

from dataclasses import replace
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
            "cuda_wmma_accumulation",
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

        accumulation = normalized.get("cuda_wmma_accumulation", "fused")

        if not isinstance(accumulation, str):
            raise TypeError("The `cuda_wmma_accumulation` option must be a string.")

        if accumulation not in {"fused", "ordered"}:
            raise ValueError(
                "The `cuda_wmma_accumulation` option must be 'fused' or 'ordered'."
            )

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
    native_execution = True
    name = "ssa.cuda.optimize_schedule"
    supported_backends = (Target.CUDA,)

    def lower_row_group(self, program, schedule, context):
        from ninetoothed.backends.emitters.cuda import TARGET

        if not TARGET.supports_cooperative_reduction(schedule):
            return {"enabled": False, "reason": "requires cooperative row reduction"}
        return {"mapping": "cooperative-loop"}

    def run(self, program, context):
        # The CTA path composes fragments of an existing source reduction.
        # A standalone dot has a different lowering contract and must not
        # advertise an unusable candidate just because its dtype is Half.
        has_dot_loop = any(
            operation.opcode == "scf.for"
            and len(operation.regions) == 1
            and any(
                inner.opcode in {"linalg.dot", "linalg.matmul"}
                for inner in operation.regions[0].operations
            )
            for block in program.blocks
            for operation in block.operations
        )
        context = replace(
            context,
            kernel_metadata={
                **context.kernel_metadata,
                "cuda_cta_has_dot_loop": has_dot_loop,
            },
        )

        return super().run(program, context)

    def select_candidate(self, candidates, context):
        requested = None

        for name in ("*", self.name, "ssa.optimize_schedule"):
            if "candidate" in context.pass_options.get(name, {}):
                requested = context.pass_options[name]["candidate"]

        if requested == "wmma-32x32":
            target_options = context.compiler_options.get("backend_options", {})
            architecture = cuda_compute_capability(target_options.get("arch", "native"))

            if architecture is not None and tuple(map(int, architecture.split("."))) < (
                8,
                0,
            ):
                raise ValueError(
                    "Unknown schedule candidate `wmma-32x32` for a pre-Ampere compile target."
                )

            if not any(candidate.name == requested for candidate in candidates):
                raise ValueError(
                    "Unknown schedule candidate `wmma-32x32` for this graph or target resources."
                )
        return super().select_candidate(candidates, context)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        platform_cuda = context.resolved_target.platform.metadata.get("cuda", {})

        reduction = schedule.get("reduction", {})
        warps = context.compiler_options.get("num_warps")

        if warps is not None and (
            not isinstance(warps, int)
            or isinstance(warps, bool)
            or not 1 <= warps <= 32
        ):
            raise ValueError("CUDA num_warps must be an integer between 1 and 32.")

        if warps is not None and schedule.get("granularity") != "blocked-linalg":
            limit = context.compiler_options.get("backend_options", {}).get(
                "max_threads_per_block", 1024
            )

            if 32 * warps > int(limit):
                raise ValueError(
                    f"CUDA requests {32 * warps} threads, exceeding the target limit {limit}."
                )

        if (
            schedule.get("granularity") == "parallel-reduction"
            and isinstance(reduction, Mapping)
            and reduction.get("mode") == "row-vector"
        ):
            extent = _static_extent(reduction.get("extent"))

            if warps is not None:
                if warps > 8:
                    raise ValueError(
                        "CUDA cooperative reduction supports at most 8 warps."
                    )

                thread_counts = (32 * warps,)
            elif extent is not None and extent <= 32:
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

        if schedule.get("granularity") != "blocked-linalg" and warps is not None:
            return (
                ScheduleCandidate(
                    name=f"explicit-threads-{32 * warps}",
                    schedule={"threads": 32 * warps},
                    tags=("explicit", "launch-geometry"),
                ),
            )

        if (
            schedule.get("granularity") != "blocked-linalg"
            or not analysis.get("dot_supports_low_precision_intrinsic", False)
            or platform_cuda.get("wmma") is False
        ):
            return ()

        mma = {"m": 16, "n": 16, "k": 16}

        candidates = (
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
            ScheduleCandidate(
                name="wmma-32x32",
                schedule={
                    "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                    "mma_shape": mma,
                    "threads": 128,
                    "cuda_cta_dot": True,
                },
                constraints={
                    "dtypes": ("float16",),
                    "minimum_compute_capability": "8.0",
                    "shared_memory_bytes": 5632,
                },
                tags=("wmma", "collective", "explicit"),
            ),
        )

        return tuple(
            candidate
            for candidate in candidates
            if candidate.name == "wmma-16x16"
            or context.kernel_metadata.get("cuda_cta_has_dot_loop", False)
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
