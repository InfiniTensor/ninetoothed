"""BangC backend implementation.

BangC code generation shares the unified SSA lowering path with the other
backends. Proven typed dataflow uses NRAM vectors, same-row reductions, DMA
transfers and LT dot tiles; the original indexed SSA remains the fallback.
Artifacts are compiled with the Cambricon ``cncc`` toolchain.
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
    LowerNativeSchedule,
    OptimizeSchedule,
)
from ninetoothed.ir import Kernel

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class BangCBackend(Backend):
    name = Target.BANGC
    supported_options = frozenset(
        {
            "arch",
            "vector_chunk",
            "task_chunk",
            "row_batch_rows",
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
            "Typed pure elementwise and same-row SSA graphs use proved NRAM "
            "vector dataflow, shared intermediates, reductions and broadcasts, "
            "with physical layout and extent guards retaining indexed fallbacks.",
            "Proved affine copies and indexed loads use native DMA transfers; "
            "full FP16 dot tiles use LT primitives and preserve the original "
            "FP32 partial accumulation groups and typed epilogue.",
            "Known limitations on MLU590: hardware scalar atomics trap at "
            "runtime (every __bang_atomic_add spelling and raw asm), so "
            "kernels containing mem.atomic_add are serialized onto one task "
            "that walks the full flat domain (exact, not fast); cncc 5.1.2 "
            "scalar integer data loads/stores truncate to 48 bits, so int64 "
            "and uint64 numerical data are rejected during code generation "
            "or compilation (including inferred C++ integer types) "
            "(internal shape/index/pointer values and scalar rand arguments "
            "with proven uint32 consumers are retained); "
            "native FP16 dot uses FP32 partial groups which may differ from "
            "CNNL accumulation order; unpartitioned multi-domain reductions "
            "can repeat expensive producers and exceed execution timeouts. "
            "Proved SSA partitions execute through ordinary native "
            "JIT handles while preserving intermediate dtype boundaries.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))
        chunk = normalized.get("vector_chunk")

        if chunk is not None and (type(chunk) is not int or chunk < 128 or chunk % 128):
            raise ValueError(
                "The BangC vector_chunk must be a positive multiple of 128."
            )

        arch = normalize_bangc_arch(normalized.get("arch", "native"))
        normalized["arch"] = arch

        if "task_chunk" in normalized:
            chunk = int(normalized["task_chunk"])

            if chunk <= 0:
                raise ValueError(
                    "The BangC `task_chunk` backend option must be positive."
                )

            if chunk > (1 << 63) - 1:
                raise ValueError(
                    "The BangC `task_chunk` backend option must fit positive int64."
                )

            normalized["task_chunk"] = chunk

        if "row_batch_rows" in normalized:
            rows = normalized["row_batch_rows"]

            if rows != "auto" and (
                not isinstance(rows, int)
                or isinstance(rows, bool)
                or rows not in {8, 16}
            ):
                raise ValueError(
                    "The BangC row_batch_rows option must be auto, 8 or 16."
                )

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class BangCLowerSchedule(LowerNativeSchedule):
    name = "ssa.bangc.lower_schedule"
    supported_backends = (Target.BANGC,)
    reduction_flag = "bangc_cooperative_reduction"


class BangCOptimizeSchedule(OptimizeSchedule):
    native_execution = True
    name = "ssa.bangc.optimize_schedule"
    supported_backends = (Target.BANGC,)

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend

        return {}


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    registry.register(BangCLowerSchedule, tags=("lowering", "correctness"))
    register_pass_bundle(
        registry,
        backend=Target.BANGC,
        optimize_schedule=BangCOptimizeSchedule,
    )
