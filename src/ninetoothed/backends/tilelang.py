"""TileLang backend implementation.

TileLang lowering is driven by the same SSA operation stream as Triton and
CUDA.  This file intentionally contains no kernel-specialized emitters.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.analysis import dot_accumulation_plan
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
    supported_options = frozenset({"max_threads_per_block", "max_shared_memory_bytes"})
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

    def run(self, program, context):
        # Classify dataflow before tile specialization; symbolic sizes are valid.
        analysis = dict(program.metadata.get("analysis", {}))
        analysis["tilelang_dot_accumulation"] = (
            dot_accumulation_plan(program) is not None
        )
        program = replace(
            program, metadata=dict(program.metadata) | {"analysis": analysis}
        )

        return super().run(program, context)

    def lower_row_group(self, program, schedule, context):
        if schedule.get("tilelang_cooperative_reduction"):
            return {
                "enabled": False,
                "reason": "cooperative reduction has no grouped-row lowering",
            }

        return {"mapping": "scalar-tile"}

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        if (
            analysis.get("has_dot")
            and not analysis.get("tilelang_dot_accumulation")
            and set(analysis.get("dot_input_dtypes", ())) <= {"float16", "bfloat16"}
        ):
            return tuple(
                ScheduleCandidate(
                    name=f"materialized-tiles-{threads}",
                    schedule={"tilelang_materialize_tiles": True, "threads": threads},
                    tags=("materialized-tiles",),
                )
                for threads in (128, 256)
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
            requested = None

            for name in ("*", self.name, "ssa.optimize_schedule"):
                requested = context.pass_options.get(name, {}).get(
                    "candidate", requested
                )

            # Keep grouping by default; explicit candidates resolve the conflict.
            if requested is None and schedule.get("row_group", {}).get("enabled"):
                return ()

            extent = _static_extent(reduction.get("extent"))

            if extent is None or extent < 256:
                # Small reduction extents are better served by the default
                # per-element lowering than a block-cooperative tree reduction.
                return ()

            # Without a concrete device limit, stay within CUDA's conservative
            # shared-memory budget. Include every reduction buffer and scratch.
            options = context.compiler_options.get("backend_options", {})
            shared_limit = int(options.get("max_shared_memory_bytes", 48 * 1024))
            scheduled = set(reduction.get("reductions", ()))
            domains = tuple(
                domain
                for domain in analysis.get("reduction_domains", ())
                if domain["result"] in scheduled
            )

            def shared_bytes(threads):
                return sum(
                    (int(domain["extent"]) + threads)
                    * max(4, _dtype_nbytes(domain.get("result_dtype")))
                    for domain in domains
                )

            return tuple(
                ScheduleCandidate(
                    name=f"cooperative-reduction-{threads}",
                    schedule={
                        "tilelang_cooperative_reduction": True,
                        "threads": threads,
                    },
                    constraints={"shared_memory_bytes": shared_bytes(threads)},
                    tags=("cooperative-reduction",),
                )
                for threads in (128, 256, 512)
                if domains
                and extent % threads == 0
                and shared_bytes(threads) <= shared_limit
            )

        if not analysis.get("tilelang_dot_accumulation") or not _dot_gemm_supported(
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

        if schedule.get("tilelang_materialize_tiles"):
            # Unknown runtime dtypes are checked by the post-specialization plan.
            return {"preserve_linalg": True}

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
