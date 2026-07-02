"""Triton backend implementation.

The backend lowerer is intentionally SSA-first.  It does not classify kernels
by operator family before emitting code; it delegates to the unified SSA
emitter, whose dispatch unit is a single SSA operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Options,
    Target,
)
from ninetoothed.backends.emitters.ssa import emit
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class TritonBackend(Backend):
    name = Target.TRITON
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend; Triton source is generated from SSA operations.",
            "No source passthrough or kernel-specialized fallback is used.",
        ),
    )

    def emit(self, kernel: Kernel, options: Options | None = None) -> Artifact:
        return emit(kernel, self.name)


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.compiler.passes import (
        Context,
        LowerIntrinsics,
        LowerMemoryScopes,
        OptimizeSchedule,
    )

    class TritonOptimizeSchedule(OptimizeSchedule):
        name = "ssa.triton.optimize_schedule"
        supported_backends = (Target.TRITON,)

        def optimization_policy(
            self,
            backend: Target,
            analysis: Mapping[str, Any],
            schedule: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del backend, analysis
            granularity = str(schedule.get("granularity", "elementwise-grid"))

            if granularity == "blocked-linalg":
                return {
                    "passes": ("linalg-block-tiling", "strict-fp32-dot-selection"),
                    "lowering": "tl.dot-blocked-matmul",
                    "schedule": {
                        "tile": {"block_m": 32, "block_n": 32, "block_k": 32},
                        "num_warps": 4,
                        "num_stages": 3,
                    },
                    "small_tile": {"block_m": 16, "block_n": 16, "block_k": 32},
                    "small_problem_lowering": "vector-kloop-microkernel",
                    "small_problem_threshold": {"m": 128, "n": 128, "k": 128},
                    "use_tensor_cores": False,
                    "input_precision": "ieee",
                }

            if granularity == "parallel-reduction":
                return {
                    "passes": ("coalesced-load", "single-program-tree-reduction"),
                    "lowering": "tl.sum/tl.max",
                    "block_size": 1024,
                    "schedule": {"num_warps": 4},
                }
            return {
                "passes": ("coalesced-vector-blocks", "load-store-combine"),
                "lowering": "tl.load/tl.store-vectorized",
                "block_size": 1024,
                "schedule": {"vector_width": 4, "num_warps": 4},
            }

    class TritonLowerMemoryScopesPass(LowerMemoryScopes):
        name = "ssa.triton.lower_memory_scopes"
        supported_backends = (Target.TRITON,)

        def memory_scopes(self, context: Context) -> Mapping[str, str]:
            del context

            return {
                "register": "tl.scalar/tl.tensor",
                "shared": "tl.dot-managed-smem",
                "global": "pointer",
            }

    class TritonLowerIntrinsicsPass(LowerIntrinsics):
        name = "ssa.triton.lower_intrinsics"
        supported_backends = (Target.TRITON,)

        def intrinsics(self, context: Context) -> Mapping[str, str]:
            del context

            return {
                "dot": "tl.dot",
                "exp": "tl.exp/tl.exp2",
                "program_id": "tl.program_id",
                "load_store": "tl.load/tl.store",
            }

    registry.register(TritonOptimizeSchedule, tags=("optimization", "triton"))
    registry.register(
        TritonLowerMemoryScopesPass, tags=("target-lowering", "memory", "triton")
    )
    registry.register(
        TritonLowerIntrinsicsPass, tags=("target-lowering", "intrinsics", "triton")
    )
