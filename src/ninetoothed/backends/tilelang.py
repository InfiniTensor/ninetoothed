"""TileLang backend implementation.

TileLang lowering is driven by the same SSA operation stream as Triton, CUDA,
and TVM.  This file intentionally contains no kernel-specialized emitters.
"""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Options,
    Target,
)
from ninetoothed.backends.emitters.tilelang import emit
from ninetoothed.compiler.passes import (
    Context,
    LowerIntrinsics,
    LowerMemoryScopes,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TileLangBackend(Backend):
    name = Target.TILELANG
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

    def emit(self, kernel: Kernel, options: Options | None = None) -> Artifact:
        return emit(kernel, options)


def _scheduled_tir_loop_policy(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "passes": ("block-tiling", "loop-reorder"),
        "lowering": "scheduled-tir-loops",
        "schedule": {"tile": dict(schedule.get("tile", {}))},
    }


def _generic_linear_or_reduction_policy(
    schedule: Mapping[str, Any],
) -> Mapping[str, Any]:
    if schedule.get("granularity") == "parallel-reduction":
        return {
            "passes": ("tree-reduction",),
            "lowering": "ssa-reduction-scf-loop",
        }
    return {
        "passes": ("coalesced-linear-indexing",),
        "lowering": "ssa-operation-linear-emission",
    }


class TileLangOptimizeSchedule(OptimizeSchedule):
    name = "ssa.tilelang.optimize_schedule"
    supported_backends = (Target.TILELANG,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        del analysis, context

        if schedule.get("granularity") != "blocked-linalg":
            return ()
        return (
            ScheduleCandidate(
                name="balanced",
                schedule={
                    "tile": {"block_m": 64, "block_n": 64, "block_k": 32},
                    "threads": 128,
                    "num_stages": 2,
                },
                tags=("default",),
            ),
            ScheduleCandidate(
                name="small",
                schedule={
                    "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                    "threads": 128,
                    "num_stages": 2,
                },
                tags=("small-problem",),
            ),
            ScheduleCandidate(
                name="wide",
                schedule={
                    "tile": {"block_m": 128, "block_n": 64, "block_k": 32},
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
        del backend, analysis

        if schedule.get("granularity") == "blocked-linalg":
            return _scheduled_tir_loop_policy(schedule)
        return _generic_linear_or_reduction_policy(schedule)


class TileLangLowerMemoryScopesPass(LowerMemoryScopes):
    name = "ssa.tilelang.lower_memory_scopes"
    supported_backends = (Target.TILELANG,)

    def memory_scopes(self, context: Context) -> Mapping[str, str]:
        del context

        return {
            "register": "local.fragment",
            "shared": "shared",
            "global": "global",
        }


class TileLangLowerIntrinsicsPass(LowerIntrinsics):
    name = "ssa.tilelang.lower_intrinsics"
    supported_backends = (Target.TILELANG,)

    def intrinsics(self, context: Context) -> Mapping[str, str]:
        del context

        return {
            "dot": "T.gemm/T.dot candidate",
            "exp": "T.exp",
            "program_id": "T.Kernel + T.get_thread_binding",
            "load_store": "T.match_buffer",
        }


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.TILELANG,
        optimize_schedule=TileLangOptimizeSchedule,
        lower_memory_scopes=TileLangLowerMemoryScopesPass,
        lower_intrinsics=TileLangLowerIntrinsicsPass,
    )
