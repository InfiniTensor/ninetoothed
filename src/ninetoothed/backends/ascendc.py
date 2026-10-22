"""AscendC backend implementation.

AscendC code generation shares the unified SSA lowering path with the other
backends.  The target model maps one AI-core block per output chunk on
Ascend devices: kernels are ``__global__ __aicore__`` task programs
with GM_ADDR device pointers, and artifacts are compiled with the CANN
``ccec`` compiler into shared libraries exposing the standard C-ABI
launcher bound to ``aclrtStream`` streams.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.ascendc import emit
from ninetoothed.backends.toolchain import normalize_ascendc_arch, resolve_ascendc_arch
from ninetoothed.compiler.passes import (
    Context,
    LowerNativeSchedule,
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
            "vector_chunk",
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
            "Proved full-row float32 graphs use private UB vector pipelines "
            "and preserve row-reduction/store ownership. Other reduction "
            "layouts retain the complete indexed fallback domain.",
            "Strict vector math uses bounded-domain and IEEE outlier guards; "
            "other scalar transcendental values use generated software helpers.",
            "Eligible half dots use Cube primitives. Explicit K32 dot groups "
            "keep independent float32 partials and ordered additions through "
            "ordinary native handles; loop reassociation remains disabled.",
            "Implicit staged defaults preserve monolithic AOT source and ABI; "
            "explicit SSA materialization currently requires JIT.",
            "Known limitations: scalar atomics are unavailable and the "
            "runtime does not guarantee exclusive block execution, so "
            "kernels containing mem.atomic_add are rejected at emission; "
            "reduction store targets must keep the dimensionality of the "
            "reduction domain; bfloat16 and float8 dtypes are rejected in "
            "the scalar lowering. Float32 vector graphs, typed float16/int32 "
            "stores, uniform/affine/transpose transfers and indexed batches "
            "use proved geometry, resource and runtime guards. Pure DMA "
            "supports 32-byte output alignment; scalar stores require "
            "exclusive 64-byte cache lines or a complete serial fallback.",
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

        if "vector_chunk" in normalized:
            chunk = normalized["vector_chunk"]

            if (
                not isinstance(chunk, int)
                or isinstance(chunk, bool)
                or chunk < 128
                or chunk > 16256
                or chunk % 128
            ):
                raise ValueError(
                    "The AscendC `vector_chunk` must be a 128-aligned integer "
                    "between 128 and 16256."
                )

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        return specialize_schedule_tiles(kernel)


class AscendCLowerSchedule(LowerNativeSchedule):
    name = "ssa.ascendc.lower_schedule"
    supported_backends = (Target.ASCENDC,)
    reduction_flag = "ascendc_cooperative_reduction"


class AscendCOptimizeSchedule(OptimizeSchedule):
    native_execution = True
    name = "ssa.ascendc.optimize_schedule"
    supported_backends = (Target.ASCENDC,)

    def run(self, program, context):
        from ninetoothed.backends.ascendc_resources import (
            NATIVE_DIVISION_CANDIDATE,
            division_strategy_options,
        )

        options = division_strategy_options(context.pass_options)
        requested_native = options.get("candidate") == NATIVE_DIVISION_CANDIDATE

        if requested_native:
            from ninetoothed.backends.ascendc_resources import (
                native_division_platform_evidence,
            )

            types = {value.name: value.type for value in program.inputs}
            supported = False

            for block in program.blocks:
                for operation in block.operations:
                    types.update(
                        (result.name, result.type) for result in operation.results
                    )

                    if operation.regions or len(operation.results) != 1:
                        continue

                    result = operation.results[0].type

                    if result.kind != "tensor" or result.dtype != "float32":
                        continue

                    if operation.opcode in {"math.reciprocal", "math.sigmoid"}:
                        operand = (
                            types.get(operation.operands[0])
                            if operation.operands
                            else None
                        )
                        supported |= bool(
                            operand
                            and operand.kind == "tensor"
                            and operand.dtype == "float32"
                        )
                    elif (
                        operation.opcode == "arith.div" and len(operation.operands) == 2
                    ):
                        lhs, rhs = (types.get(name) for name in operation.operands)
                        supported |= bool(
                            lhs
                            and rhs
                            and lhs.dtype == rhs.dtype == "float32"
                            and rhs.kind == "tensor"
                        )

            if not supported:
                raise ValueError(
                    "The native division candidate requires a typed float32 vector division."
                )

            if (
                program.metadata.get("schedule", {}).get("granularity")
                == "blocked-linalg"
            ):
                raise ValueError(
                    "The native division candidate requires a proved vector schedule."
                )

            options = context.compiler_options.get("backend_options", {})

            if (
                resolve_ascendc_arch(
                    normalize_ascendc_arch(options.get("arch", "native"))
                )
                != "dav-c220"
            ):
                raise ValueError(
                    "Native division requires the validated dav-c220 platform."
                )

            native_division_platform_evidence()
            program = replace(
                program,
                facts={
                    **program.facts,
                    "analysis": {
                        **program.facts.get("analysis", {}),
                        "ascendc_native_division_eligible": True,
                    },
                },
            )
        return super().run(program, context)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        reduction = schedule.get("reduction", {})
        from ninetoothed.backends.ascendc_resources import (
            NATIVE_DIVISION_CANDIDATE,
            division_strategy_options,
        )

        requested = division_strategy_options(context.pass_options).get("candidate")

        if requested == NATIVE_DIVISION_CANDIDATE:
            if not analysis.get("ascendc_native_division_eligible"):
                raise ValueError(
                    "The native division candidate requires a typed float32 vector division."
                )

            native_schedule = {"ascendc_strict_division_strategy": "native_division"}

            if isinstance(reduction, Mapping) and reduction.get("mode") == "row-vector":
                native_schedule["ascendc_cooperative_reduction"] = True
            return (
                ScheduleCandidate(
                    name="vector-native-division",
                    schedule=native_schedule,
                    tags=("vector", "precision-compatible", "native-division"),
                ),
            )

        return ()

    def select_candidate(self, candidates, context):
        from ninetoothed.backends.ascendc_resources import (
            division_strategy_options,
        )

        requested = division_strategy_options(context.pass_options).get("candidate")

        if requested is not None and not candidates:
            raise ValueError(
                f"Unknown schedule candidate `{requested}` for `{self.name}`. "
                "No proved schedule candidates are available."
            )
        return super().select_candidate(candidates, context)

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend

        if schedule.get("granularity") == "blocked-linalg":
            return {}

        policy = {"strict_validity_summary": True}

        if schedule.get("ascendc_strict_division_strategy") == "native_division":
            policy["strict_division_strategy"] = "native_division"
        return policy


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    registry.register(AscendCLowerSchedule, tags=("lowering", "correctness"))
    register_pass_bundle(
        registry,
        backend=Target.ASCENDC,
        optimize_schedule=AscendCOptimizeSchedule,
    )
