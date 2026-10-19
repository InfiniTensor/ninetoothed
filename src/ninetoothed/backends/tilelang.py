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

    def lower_row_group(self, program, schedule, context):
        return {"mapping": "scalar-tile"}

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del context

        if schedule.get("granularity") == "exp-reduction-dot-region":
            return (
                ScheduleCandidate(
                    name="flash-128x128",
                    schedule={
                        "tile": {"block_m": 128, "block_n": 128},
                        "threads": 256,
                        "num_stages": 2,
                    },
                    tags=("default", "flash"),
                ),
                ScheduleCandidate(
                    name="flash-128x64",
                    schedule={
                        "tile": {"block_m": 128, "block_n": 64},
                        "threads": 256,
                        "num_stages": 2,
                    },
                    tags=("flash",),
                ),
            )

        if schedule.get("granularity") == "elementwise-grid":
            return (
                ScheduleCandidate(
                    name="elementwise-128",
                    schedule={"threads": 128},
                    tags=("default", "elementwise"),
                ),
                ScheduleCandidate(
                    name="elementwise-256",
                    schedule={"threads": 256},
                    tags=("elementwise",),
                ),
            )

        reduction = schedule.get("reduction", {})

        if (
            schedule.get("granularity") == "parallel-reduction"
            and isinstance(reduction, Mapping)
            and reduction.get("mode") == "row-vector"
        ):
            extent = _static_extent(reduction.get("extent"))

            if extent is None or extent < 256:
                # Small reduction extents are better served by the default
                # per-element lowering than a block-cooperative tree reduction.
                return ()

            if extent is not None and extent <= 32:
                thread_counts = (32, 64, 128)
            elif extent is not None and extent <= 128:
                thread_counts = (128, 64, 256)
            elif extent is not None and extent <= 512:
                thread_counts = (128, 256, 512)
            else:
                thread_counts = (128, 256, 512)

            return tuple(
                ScheduleCandidate(
                    name=f"cooperative-reduction-{threads}",
                    schedule={
                        "tilelang_cooperative_reduction": True,
                        "threads": threads,
                    },
                    tags=("cooperative-reduction",),
                )
                for threads in thread_counts
            )

        if schedule.get("granularity") != "blocked-linalg" or not _dot_gemm_supported(
            analysis
        ):
            return ()

        # Wider-than-16-bit operands (fp32, ...) use a smaller pipeline to stay
        # within the device shared-memory budget; 16-bit ones afford 128x128.
        nbytes = max(
            (_dtype_nbytes(dtype) for dtype in analysis.get("dot_input_dtypes", ())),
            default=2,
        )

        if nbytes >= 4:
            default_tile = {"block_m": 64, "block_n": 64, "block_k": 32}
            default_stages = 2
        else:
            default_tile = {"block_m": 128, "block_n": 128, "block_k": 32}
            default_stages = 3

        return (
            ScheduleCandidate(
                name="balanced",
                schedule={
                    "tile": default_tile,
                    "threads": 128,
                    "num_stages": default_stages,
                },
                tags=("default",),
            ),
            ScheduleCandidate(
                name="small",
                schedule={
                    "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                    "threads": 128,
                    "num_stages": 2,
                },
                tags=("small-problem",),
            ),
            ScheduleCandidate(
                name="wide",
                schedule={
                    "tile": {"block_m": 128, "block_n": 256, "block_k": 64},
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
            return {"preserve_linalg": _dot_gemm_supported(analysis)}

        return {}


_GEMM_DOT_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
    }
)


def _dot_gemm_supported(analysis: Mapping[str, Any]) -> bool:
    """Return whether dot inputs map onto TileLang's ``T.gemm`` path."""
    dot_dtypes = set(analysis.get("dot_input_dtypes", ()))

    return bool(dot_dtypes) and dot_dtypes <= _GEMM_DOT_DTYPES


def _dtype_nbytes(dtype: Any) -> int:
    return {
        "float64": 8,
        "int64": 8,
        "uint64": 8,
        "float32": 4,
        "int32": 4,
        "uint32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "uint16": 2,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }.get(str(dtype), 2)


def _static_extent(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TILELANG,
        optimize_schedule=TileLangOptimizeSchedule,
    )
