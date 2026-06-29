"""Backend-specific SSA pass registrations."""

from __future__ import annotations

from typing import Any, Mapping

from ninetoothed.backends.base import BackendName
from ninetoothed.ssa_passes import OptimizeSchedulePass, SSAPassRegistry


class TritonOptimizeSchedulePass(OptimizeSchedulePass):
    name = "ssa.triton.optimize_schedule"
    supported_backends = (BackendName.TRITON,)

    def optimization_policy(
        self,
        backend: BackendName,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis
        granularity = str(schedule.get("granularity", "elementwise-grid"))
        if granularity == "blocked-linalg":
            return {
                "passes": ("linalg-block-tiling", "strict-fp32-dot-selection"),
                "lowering": "tl.dot-blocked-matmul",
                "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                "small_tile": {"block_m": 16, "block_n": 16, "block_k": 32},
                "small_problem_lowering": "vector-kloop-microkernel",
                "small_problem_threshold": {"m": 128, "n": 128, "k": 128},
                "num_warps": 4,
                "num_stages": 3,
                "use_tensor_cores": False,
                "input_precision": "ieee",
            }
        if granularity == "parallel-reduction":
            return {
                "passes": ("coalesced-load", "single-program-tree-reduction"),
                "lowering": "tl.sum/tl.max",
                "block_size": 1024,
                "num_warps": 4,
            }
        return {
            "passes": ("coalesced-vector-blocks", "load-store-combine"),
            "lowering": "tl.load/tl.store-vectorized",
            "block_size": 1024,
            "vector_width": 4,
            "num_warps": 4,
        }


class CudaOptimizeSchedulePass(OptimizeSchedulePass):
    name = "ssa.cuda.optimize_schedule"
    supported_backends = (BackendName.CUDA,)

    def optimization_policy(
        self,
        backend: BackendName,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis
        if schedule.get("granularity") == "blocked-linalg":
            return {
                "passes": ("block-tiling",),
                "lowering": "thread-block-matmul",
                "tile": {"block_m": 16, "block_n": 16, "block_k": 8},
            }
        return _generic_linear_or_reduction_policy(schedule)


class TileLangOptimizeSchedulePass(OptimizeSchedulePass):
    name = "ssa.tilelang.optimize_schedule"
    supported_backends = (BackendName.TILELANG,)

    def optimization_policy(
        self,
        backend: BackendName,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis
        if schedule.get("granularity") == "blocked-linalg":
            return _scheduled_tir_loop_policy(schedule)
        return _generic_linear_or_reduction_policy(schedule)


class TvmOptimizeSchedulePass(OptimizeSchedulePass):
    name = "ssa.tvm.optimize_schedule"
    supported_backends = (BackendName.TVM,)

    def optimization_policy(
        self,
        backend: BackendName,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis
        if schedule.get("granularity") == "blocked-linalg":
            return _scheduled_tir_loop_policy(schedule)
        return _generic_linear_or_reduction_policy(schedule)


def _scheduled_tir_loop_policy(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "passes": ("block-tiling", "loop-reorder"),
        "lowering": "scheduled-tir-loops",
        "tile": dict(schedule.get("tile", {})),
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


def register_backend_specific_ssa_passes(registry: SSAPassRegistry) -> None:
    registry.register(TritonOptimizeSchedulePass, tags=("optimization", "triton"))
    registry.register(CudaOptimizeSchedulePass, tags=("optimization", "cuda"))
    registry.register(TileLangOptimizeSchedulePass, tags=("optimization", "tilelang"))
    registry.register(TvmOptimizeSchedulePass, tags=("optimization", "tvm"))
