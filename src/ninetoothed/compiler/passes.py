"""SSA optimization and target-lowering pass infrastructure.

The passes in this module operate on fine-grained SSA operations: loops,
arithmetic, reductions, tensor loads/stores, and linalg-style primitives.
They do not introduce whole-operator IR nodes.

Pass execution is intentionally organized like a compiler pipeline:

* hardware-independent passes canonicalize, analyze, and attach generic
  schedule intent to SSA;
* language-specific passes, registered from ``ninetoothed.backends``, select
  schedules that are consumed by target emitters and launch planning.
* platform passes validate capabilities and apply concrete target constraints.

The pass registry is the public control point for default pipelines, custom
pipelines, and deterministic target schedule selection.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ninetoothed.backends.core import Target, normalize_target
from ninetoothed.compiler.canonicalize import canonicalize_dtype_expressions
from ninetoothed.compiler.cse import (
    eliminate_common_subexpressions,
)
from ninetoothed.compiler.layout import LayoutTransfer, analyze_layout_transfer
from ninetoothed.compiler.numeric_bounds import analyze_numeric_bounds
from ninetoothed.compiler.reduction_geometry import prove_unit_parallel_reductions
from ninetoothed.compiler.reductions import analyze_reductions
from ninetoothed.compiler.uniform import uniform_store
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa
from ninetoothed.targets import TargetContext, resolve_target_context

HARDWARE_INDEPENDENT = "hardware_independent"
LANGUAGE_SPECIFIC = "language_specific"
PLATFORM_SPECIFIC = "platform_specific"
BACKEND_SPECIFIC = "backend_specific"


@dataclass(frozen=True, kw_only=True)
class PipelineSpec:
    """Declarative pass pipeline configuration."""

    passes: tuple[str, ...]
    mode: str = "default"
    pass_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class Context:
    """Context shared by SSA passes."""

    backend: Target
    compiler_options: Mapping[str, Any]
    kernel_metadata: Mapping[str, Any]
    tensors: tuple[Any, ...] = ()
    pass_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    pipeline_spec: PipelineSpec | None = None
    target: TargetContext | None = None

    def __post_init__(self) -> None:
        if self.target is None:
            object.__setattr__(self, "target", resolve_target_context(self.backend))

    @property
    def resolved_target(self) -> TargetContext:
        assert self.target is not None

        return self.target


@dataclass(frozen=True, kw_only=True)
class ScheduleCandidate:
    """One legal backend schedule choice for deterministic selection."""

    name: str
    schedule: Mapping[str, Any]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def as_metadata(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "schedule": dict(self.schedule),
            "constraints": dict(self.constraints),
            "tags": self.tags,
        }


class Pass:
    """Base class for semantics-preserving SSA transforms."""

    name = "ssa.pass"
    category = HARDWARE_INDEPENDENT
    phase = "generic"
    supported_backends: tuple[Target, ...] = ()
    option_validator: Callable[..., None] | None = None
    reads: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    changes_ssa: bool = False
    preserves_stage_plan: bool = False

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Descriptor:
    """Registry metadata for one SSA pass."""

    name: str
    category: str
    phase: str
    factory: Callable[[], Pass]
    supported_backends: tuple[Target, ...] = ()
    default_enabled: bool = True
    description: str = ""
    tags: tuple[str, ...] = ()
    option_validator: Callable[..., None] | None = None
    reads: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    changes_ssa: bool = False

    def create(self) -> Pass:
        return self.factory()

    def supports(self, backend: Target | str | None) -> bool:
        if not self.supported_backends:
            return True

        backend_name = normalize_target(backend)

        return backend_name in self.supported_backends


class Registry:
    """Registry that owns pass discovery and pipeline construction."""

    def __init__(self) -> None:
        self._descriptors: dict[str, Descriptor] = {}

    def register(
        self,
        pass_factory: type[Pass] | Callable[[], Pass] | Pass,
        *,
        name: str | None = None,
        category: str | None = None,
        phase: str | None = None,
        supported_backends: Sequence[Target | str] | None = None,
        default_enabled: bool | None = None,
        description: str | None = None,
        tags: Sequence[str] = (),
    ) -> None:
        factory = _normalize_pass_factory(pass_factory)
        probe = factory()
        pass_name = name or probe.name

        if pass_name in self._descriptors:
            raise ValueError(f"SSA pass `{pass_name}` is already registered.")

        backends = (
            tuple(normalize_target(backend) for backend in supported_backends)
            if supported_backends is not None
            else tuple(getattr(probe, "supported_backends", ()))
        )
        doc = description

        if doc is None:
            raw_doc = (probe.__doc__ or "").strip().splitlines()
            doc = raw_doc[0].strip() if raw_doc else ""

        normalized_category = category or probe.category

        if normalized_category == BACKEND_SPECIFIC:
            normalized_category = LANGUAGE_SPECIFIC

        self._descriptors[pass_name] = Descriptor(
            name=pass_name,
            category=normalized_category,
            phase=phase or probe.phase,
            factory=factory,
            supported_backends=backends,
            default_enabled=probe.default_enabled
            if hasattr(probe, "default_enabled") and default_enabled is None
            else bool(True if default_enabled is None else default_enabled),
            description=doc,
            tags=tuple(tags),
            option_validator=probe.option_validator,
            reads=probe.reads,
            provides=probe.provides,
            changes_ssa=probe.changes_ssa,
        )

    def validate_options(self, pass_names, options) -> None:
        fallback = None

        for name, settings in options.items():
            if name in {
                "ssa.plan_reduction_reuse",
                "ssa.prove_numeric_bounds",
                "ssa.vector_resource_policy",
                "ssa.ascendc.plan_serial_queue_storage",
            }:
                raise ValueError(
                    f"SSA pass `{name}` was replaced by shared analysis or target planning."
                )

            descriptor = self._descriptors.get(name)

            if descriptor is None:
                if fallback is None:
                    fallback = _default_registry()

                descriptor = fallback._descriptors.get(name)

            if descriptor is not None and descriptor.option_validator is not None:
                descriptor.option_validator(settings, present=name in pass_names)

    def get(self, name: str) -> Descriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            available = ", ".join(self._descriptors)
            raise KeyError(
                f"Unknown SSA pass `{name}`. Available passes: {available}."
            ) from exc

    def descriptors(
        self,
        *,
        category: str | None = None,
        backend: Target | str | None = None,
    ) -> tuple[Descriptor, ...]:
        result = tuple(self._descriptors.values())

        if category == BACKEND_SPECIFIC:
            category = LANGUAGE_SPECIFIC

        if category is not None:
            result = tuple(
                descriptor for descriptor in result if descriptor.category == category
            )

        if backend is not None:
            result = tuple(
                descriptor for descriptor in result if descriptor.supports(backend)
            )
        return result

    def names(
        self,
        *,
        category: str | None = None,
        backend: Target | str | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            descriptor.name
            for descriptor in self.descriptors(category=category, backend=backend)
        )


class Pipeline:
    """Ordered pass pipeline with a textual trace in SSA metadata."""

    def __init__(
        self,
        passes: tuple[Pass, ...],
        *,
        descriptors: tuple[Descriptor, ...] = (),
        spec: PipelineSpec | None = None,
        registry: Registry | None = None,
    ):
        self.passes = passes
        self.descriptors = descriptors
        self.spec = spec
        self.registry = registry

    def run(
        self,
        program: ssa.Program,
        context: Context,
        *,
        registry: Registry | None = None,
    ) -> ssa.Program:
        _default_registry(registry or self.registry).validate_options(
            {pass_.name for pass_ in self.passes}, context.pass_options
        )
        from ninetoothed.compiler.materialization import retain_bound_plan

        current = ssa.verify_program(retain_bound_plan(program))
        current = _with_metadata(current, native_execution=False)
        current = _with_metadata(current, pipeline_selection=self._pipeline_metadata())

        for pass_ in self.passes:
            previous = current
            current = pass_.run(current, context)

            if (
                current.plans
                and not pass_.preserves_stage_plan
                and (previous.inputs, previous.outputs, previous.blocks)
                != (current.inputs, current.outputs, current.blocks)
            ):
                if previous.metadata.get("materialization", {}).get(
                    "explicitly_enabled"
                ):
                    raise ValueError(
                        f"Pass `{pass_.name}` changed SSA after explicitly requested stage planning. "
                        "Place materialization planning after this transform."
                    )

                current = replace(current, plans={})
                current = _with_metadata(
                    current,
                    materialization={
                        "enabled": False,
                        "selected": False,
                        "reason": "SSA changed after stage planning",
                    },
                )

            if pass_.preserves_stage_plan:
                current = replace(
                    current,
                    plans={
                        name: plan.bind(current) if plan else None
                        for name, plan in current.plans.items()
                    },
                )

            current = refresh_program_facts(current, context)
            ssa.verify_program(current)
            current = _with_metadata(
                current,
                pass_trace=tuple(current.metadata.get("pass_trace", ()))
                + (pass_.name,),
            )
        return current

    def _pipeline_metadata(self) -> Mapping[str, Any]:
        if self.spec is None:
            return {
                "mode": "manual",
                "selected_passes": tuple(pass_.name for pass_ in self.passes),
            }

        categories = {
            HARDWARE_INDEPENDENT: tuple(
                descriptor.name
                for descriptor in self.descriptors
                if descriptor.category == HARDWARE_INDEPENDENT
            ),
            LANGUAGE_SPECIFIC: tuple(
                descriptor.name
                for descriptor in self.descriptors
                if descriptor.category == LANGUAGE_SPECIFIC
            ),
            PLATFORM_SPECIFIC: tuple(
                descriptor.name
                for descriptor in self.descriptors
                if descriptor.category == PLATFORM_SPECIFIC
            ),
        }
        categories[BACKEND_SPECIFIC] = categories[LANGUAGE_SPECIFIC]

        return {
            "mode": self.spec.mode,
            "selected_passes": self.spec.passes,
            "reason": self.spec.reason,
            "categories": categories,
        }


class Canonicalize(Pass):
    """Normalize generic SSA into the canonical dialect used by all backends."""

    name = "ssa.canonicalize"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    changes_ssa = True

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        return _with_metadata(
            canonicalize_dtype_expressions(program),
            dialect="generic-ssa",
            canonical=True,
            coarse_operator_nodes=False,
        )


class UnrollStaticElementwiseLoops(Pass):
    """Expand small typed pure loops in their original arithmetic order."""

    name = "ssa.unroll_static_elementwise"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    changes_ssa = True

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        enabled = _reuse_pass_enabled(context, self.name)
        expanded = ()

        if enabled:
            from ninetoothed.compiler.unroll import unroll_static_elementwise_loops

            program, expanded = unroll_static_elementwise_loops(program)
        return _with_metadata(
            program, static_unroll={"enabled": enabled, "expanded": expanded}
        )


class CommonSubexpressionElimination(Pass):
    """Reuse identical expressions within one effect-free block segment."""

    name = "ssa.cse"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    changes_ssa = True

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        enabled = _reuse_pass_enabled(context, self.name)
        eliminated = ()

        if enabled:
            program, eliminated = eliminate_common_subexpressions(program)

        return _with_metadata(
            program,
            cse={"enabled": enabled, "eliminated": eliminated},
        )


class PlanMaterialization(Pass):
    """Plan storage boundaries for reused expensive tensor producer graphs."""

    name = "ssa.plan_materialization"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"
    default_enabled = True

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        options = _pass_options(context, self.name)
        default = context.backend in {Target.CUDA, Target.BANGC, Target.ASCENDC}

        if set(options) - {"enabled"} or not isinstance(
            options.get("enabled", default), bool
        ):
            raise ValueError(
                f"SSA pass `{self.name}` only accepts a boolean `enabled` option."
            )

        enabled = options.get("enabled", default)

        if enabled and context.backend not in {
            Target.CUDA,
            Target.BANGC,
            Target.ASCENDC,
        }:
            raise ValueError(
                "SSA materialization currently requires a native C-style stage consumer."
            )

        from ninetoothed.compiler.materialization import plan_materialization

        plan = plan_materialization(program) if enabled else None
        program = replace(program, plans={**program.plans, "materialization": plan})

        return _with_metadata(
            program,
            materialization={
                "enabled": enabled,
                "explicitly_enabled": enabled and options.get("enabled") is True,
                "selected": plan is not None,
                **(plan.metadata() if plan else {}),
            },
        )


def _reuse_pass_enabled(context, name):
    options = _pass_options(context, name)

    if set(options) - {"enabled"} or not isinstance(options.get("enabled", True), bool):
        raise ValueError(f"SSA pass `{name}` only accepts a boolean `enabled` option.")
    return options.get("enabled", True)


FACT_NAMES = ("analysis", "numeric_bounds", "reduction_reuse")


def program_facts(program, name):
    """Read a fact only from the SSA snapshot that established it."""
    facts = program.facts

    if facts.get("structure") != (program.inputs, program.outputs, program.blocks):
        return {}
    return facts.get(name, {})


def refresh_program_facts(program, context):
    """Rebuild existing analysis after target shape or layout changes."""
    if "structure" in program.facts and (
        program.facts.get("structure")
        != (program.inputs, program.outputs, program.blocks)
        or program.facts.get("tensors") != context.tensors
        or program.facts.get("target") != context.resolved_target
    ):
        previous = program.metadata.get("analysis", {}).get("layout_transfer")
        refreshed = AnalyzeEffects().run(program, context)

        # Keep the public diagnostic after decomposition removes the view op.
        # Fresh facts remain independent: a diagnostic is never a new proof.
        if (
            isinstance(previous, LayoutTransfer)
            and not previous.schedulable
            and refreshed.metadata["analysis"]["layout_transfer"] is None
        ):
            refreshed = _with_metadata(
                refreshed,
                analysis={
                    **refreshed.metadata["analysis"],
                    "layout_transfer": previous,
                },
            )

        return refreshed
    return program


class AnalyzeEffects(Pass):
    """Collect dataflow facts needed by schedule selection."""

    name = "ssa.analyze_effects"
    category = HARDWARE_INDEPENDENT
    phase = "analysis"
    provides = FACT_NAMES

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        unit_parallel = prove_unit_parallel_reductions(program)
        reuse_enabled = bool(program.metadata.get("cse", {}).get("enabled", False))
        reuse = {
            "enabled": reuse_enabled,
            "results": unit_parallel if reuse_enabled else (),
        }
        program = _with_metadata(program, reduction_reuse=reuse)
        opcodes = tuple(_iter_opcodes(program))
        stores = sum(1 for opcode in opcodes if opcode == "mem.store")
        reductions = sum(1 for opcode in opcodes if opcode.startswith("reduce."))
        loops = sum(1 for opcode in opcodes if opcode == "scf.for")
        dot_input_dtypes = _linalg_input_dtypes(program)
        reduction_analysis = analyze_reductions(
            program, context.tensors, unit_parallel=unit_parallel
        )
        layout_transfer = analyze_layout_transfer(program, context.tensors)

        program = _with_metadata(
            program,
            numeric_bounds=analyze_numeric_bounds(program),
            analysis={
                "operation_count": len(opcodes),
                "store_count": stores,
                "reduction_count": reductions,
                "loop_count": loops,
                "has_dot": "linalg.dot" in opcodes or "linalg.matmul" in opcodes,
                "dot_input_dtypes": dot_input_dtypes,
                "dot_supports_low_precision_intrinsic": bool(dot_input_dtypes)
                and all(dtype in {"float16", "bfloat16"} for dtype in dot_input_dtypes),
                "has_exp_reduction_dot_pattern": _has_exp_reduction_dot_pattern(
                    opcodes
                ),
                "layout_transfer": layout_transfer,
                "uniform_store": uniform_store(program),
                **reduction_analysis,
            },
        )

        return replace(
            program,
            facts={
                "structure": (program.inputs, program.outputs, program.blocks),
                "tensors": context.tensors,
                "target": context.resolved_target,
                **{name: program.metadata[name] for name in FACT_NAMES},
            },
        )


class DecomposeLinalg(Pass):
    """Lower high-level linalg ops into index/extract/store SSA operations."""

    name = "ssa.decompose_linalg"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    changes_ssa = True
    preserves_stage_plan = True

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        optimization = dict(program.metadata.get("optimization", {}))

        # Accumulator precision is part of dot semantics, independent of tensorization.
        dot_dtypes = _linalg_input_dtypes(program)

        if dot_dtypes and all(dtype in {"float16", "bfloat16"} for dtype in dot_dtypes):
            optimization["preserve_linalg"] = True
            program = _with_metadata(program, optimization=optimization)

        if optimization.get("preserve_linalg"):
            return _with_metadata(
                program,
                linalg_decomposed=False,
                linalg_preserved=True,
                coarse_operator_nodes=False,
            )

        value_types = _program_value_types(program)
        blocks = tuple(
            _decompose_linalg_block(block, value_types) for block in program.blocks
        )

        return _replace_program(
            program,
            blocks=blocks,
            metadata=dict(program.metadata)
            | {
                "linalg_decomposed": True,
                "coarse_operator_nodes": False,
            },
        )


class SelectSchedule(Pass):
    """Attach backend-neutral schedule intent."""

    name = "ssa.select_schedule"
    category = HARDWARE_INDEPENDENT
    phase = "schedule_intent"
    reads = ("analysis",)
    provides = ("schedule",)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        analysis = dict(program_facts(program, "analysis"))
        options = _pass_options(context, self.name)
        schedule = {
            "granularity": _schedule_granularity(analysis),
            "indexing": "flat-contiguous",
            "parallelism": "program-blocks",
        }
        reduction = analysis.get("reduction_schedule")
        layout_transfer = analysis.get("layout_transfer")

        if "reduction" in options:
            raise ValueError(
                "Reduction schedule options are compiler-owned and cannot be "
                "overridden through `ssa.select_schedule`."
            )

        schedule = _merge_nested(schedule, options)

        if isinstance(reduction, Mapping):
            schedule["reduction"] = dict(reduction)

        if (
            schedule.get("granularity") == "layout-transfer"
            and isinstance(layout_transfer, LayoutTransfer)
            and layout_transfer.schedulable
        ):
            schedule["layout_transfer"] = layout_transfer

        return _with_metadata(program, schedule=schedule, schedule_analysis=analysis)


class LowerNativeSchedule(Pass):
    """Map a proved reduction domain onto serial native task ownership."""

    category = LANGUAGE_SPECIFIC
    phase = "schedule_lowering"
    reads = ("schedule",)
    provides = ("native_task_domain",)
    reduction_flag: str

    def run(self, program, context):
        del context
        schedule = dict(program.metadata.get("schedule", {}))
        reduction = schedule.get("reduction", {})

        if (
            schedule.get("granularity") == "parallel-reduction"
            and isinstance(reduction, Mapping)
            and reduction.get("mode") == "row-vector"
        ):
            schedule[self.reduction_flag] = True

        return _with_metadata(program, schedule=schedule)


class OptimizeSchedule(Pass):
    """Contract for language-specific schedule optimization passes."""

    name = "ssa.optimize_schedule"
    native_execution = False
    category = LANGUAGE_SPECIFIC
    phase = "optimization"
    reads = ("analysis", "schedule")
    provides = ("target_schedule",)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        analysis = dict(program_facts(program, "analysis"))
        schedule = dict(program.metadata.get("schedule", {}))
        proposed = _deduplicate_candidates(
            tuple(self.schedule_candidates(analysis, schedule, context))
        )
        legal = []
        rejected = []

        for candidate in proposed:
            reason = _candidate_rejection_reason(candidate, analysis, context)

            if reason is None:
                legal.append(candidate)
            else:
                rejected.append(candidate.as_metadata() | {"reason": reason})

        selected = self.select_candidate(tuple(legal), context)

        if selected is not None:
            legal.remove(selected)
            legal.insert(0, selected)

        max_num_configs = context.compiler_options.get("max_num_configs")

        if max_num_configs is not None:
            limit = max(0, int(max_num_configs))
            rejected.extend(
                candidate.as_metadata()
                | {"reason": f"truncated by max_num_configs={limit}"}
                for candidate in legal[limit:]
            )
            legal = legal[:limit]

        candidates = tuple(legal)
        rejected = tuple(rejected)

        if selected is not None:
            schedule = _merge_nested(schedule, selected.schedule)

        optimization = dict(
            self.optimization_policy(context.backend, analysis, schedule)
        )
        optimization_options = dict(
            _pass_options(context, self.name, "ssa.optimize_schedule")
        )
        optimization_options.pop("candidate", None)

        if "reduction" in dict(optimization_options.get("schedule", {})):
            raise ValueError(
                "Reduction schedule options are compiler-owned and cannot be "
                "overridden by an optimization pass."
            )

        optimization = _merge_nested(
            optimization,
            optimization_options,
        )
        optimized_schedule = _merge_nested(
            schedule,
            dict(optimization.get("schedule", {})),
        )
        reduction = analysis.get("reduction_schedule")

        if isinstance(reduction, Mapping):
            optimized_schedule["reduction"] = dict(reduction)

        group = optimized_schedule.get("row_group", {})

        if group.get("enabled"):
            optimized_schedule["row_group"] = dict(group) | dict(
                self.lower_row_group(program, optimized_schedule, context)
            )

        candidate_metadata = tuple(candidate.as_metadata() for candidate in candidates)

        return _replace_program(
            program,
            metadata=dict(program.metadata)
            | {
                "schedule": optimized_schedule,
                "optimization": optimization,
                **({"native_execution": True} if self.native_execution else {}),
                "schedule_candidates": candidate_metadata,
                "rejected_schedule_candidates": rejected,
                "selected_schedule_candidate": (
                    selected.name if selected is not None else None
                ),
            },
        )

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        """Return legal schedule candidates in deterministic fallback order."""
        del analysis, schedule, context

        return ()

    def lower_row_group(self, program, schedule, context):
        """Select a real backend implementation of the common grouping intent."""
        return {"enabled": False, "reason": "backend has no grouped-row lowering"}

    def select_candidate(
        self,
        candidates: tuple[ScheduleCandidate, ...],
        context: Context,
    ) -> ScheduleCandidate | None:
        """Select one legal candidate using explicit or default policy."""
        options = _pass_options(context, self.name, "ssa.optimize_schedule")
        requested = options.get("candidate")

        if not candidates:
            if requested is not None:
                raise ValueError(
                    f"Unknown schedule candidate `{requested}` for `{self.name}`. "
                    "No proved schedule candidates are available."
                )

            return None

        if requested is None:
            return candidates[0]

        for candidate in candidates:
            if candidate.name == requested:
                return candidate

        available = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(
            f"Unknown schedule candidate `{requested}` for `{self.name}`. "
            f"Available candidates: {available}."
        )

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class ValidateTargetCapabilities(Pass):
    """Reject IR capabilities explicitly unsupported by a concrete target."""

    name = "ssa.validate_target_capabilities"
    category = PLATFORM_SPECIFIC
    phase = "target_legality"
    provides = ("target_capabilities",)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        group = program.metadata.get("schedule", {}).get("row_group", {})

        if group.get("enabled") and "mapping" not in group:
            raise ValueError("Row grouping must precede backend schedule optimization.")

        snapshot = (
            program.inputs,
            program.outputs,
            program.blocks,
            context.resolved_target,
        )

        if program.facts.get("target_validation") == snapshot:
            return program

        required = _required_capabilities(program)
        report = context.resolved_target.validate_capabilities(required)

        return replace(
            _with_metadata(
                program, required_capabilities=required, target_capabilities=report
            ),
            facts={**program.facts, "target_validation": snapshot},
        )


def create_default_registry() -> Registry:
    from ninetoothed.compiler.row_reductions import GroupReductionRows

    registry = Registry()
    registry.register(Canonicalize, tags=("generic", "required"))
    registry.register(UnrollStaticElementwiseLoops, tags=("generic", "unroll"))
    registry.register(CommonSubexpressionElimination, tags=("generic", "reuse"))
    registry.register(
        PlanMaterialization, tags=("generic", "materialization", "guarded")
    )
    registry.register(AnalyzeEffects, tags=("generic", "analysis", "required"))
    registry.register(SelectSchedule, tags=("schedule",))
    registry.register(GroupReductionRows, tags=("schedule", "reduction"))
    _register_backend_passes(registry)
    registry.register(DecomposeLinalg, tags=("generic", "linalg", "decomposition"))
    registry.register(
        ValidateTargetCapabilities,
        tags=("target", "capability", "legality", "required"),
    )

    return registry


def _register_backend_passes(registry: Registry) -> None:
    from ninetoothed.backends.registry import register_passes

    register_passes(registry)


def registered(
    *,
    category: str | None = None,
    backend: Target | str | None = None,
    registry: Registry | None = None,
) -> tuple[Descriptor, ...]:
    """Return registered SSA pass descriptors."""
    return _default_registry(registry).descriptors(category=category, backend=backend)


def default_spec(
    backend: Target | str | None,
    *,
    registry: Registry | None = None,
) -> PipelineSpec:
    """Return the default declarative pipeline for a backend."""
    backend_name = normalize_target(backend)
    registry = _default_registry(registry)
    pass_names = _default_pass_names(backend_name, registry)
    _validate_passes(pass_names, backend_name, registry)

    return PipelineSpec(
        passes=pass_names,
        mode="default",
        reason="default backend pipeline",
    )


def default_pipeline(
    backend: Target | str | None,
    *,
    registry: Registry | None = None,
) -> Pipeline:
    """Return the default target-aware SSA lowering pipeline."""
    backend_name = normalize_target(backend)

    return build(
        default_spec(backend_name, registry=registry),
        backend=backend_name,
        registry=registry,
    )


def build(
    spec: PipelineSpec | Sequence[str] | Mapping[str, Any],
    *,
    backend: Target | str | None,
    registry: Registry | None = None,
) -> Pipeline:
    """Build an executable pass pipeline from a declarative spec."""
    backend_name = normalize_target(backend)
    registry = _default_registry(registry)
    normalized = _normalize_pipeline_spec(spec, backend_name, registry)
    descriptors = tuple(registry.get(name) for name in normalized.passes)
    passes = tuple(descriptor.create() for descriptor in descriptors)

    return Pipeline(passes, descriptors=descriptors, spec=normalized, registry=registry)


def lower_for_target(
    program: ssa.Program,
    *,
    backend: Target | str | None,
    platform: str | None = None,
    compute_arch: str | None = None,
    target_context: TargetContext | None = None,
    compiler_options: Mapping[str, Any] | None = None,
    kernel_metadata: Mapping[str, Any] | None = None,
    tensors: tuple[Any, ...] = (),
    pass_pipeline: Pipeline
    | PipelineSpec
    | Sequence[str]
    | Mapping[str, Any]
    | None = None,
    pass_options: Mapping[str, Mapping[str, Any]] | None = None,
    pass_registry: Registry | None = None,
) -> ssa.Program:
    """Run an SSA pass pipeline for a backend."""
    backend_name = normalize_target(backend)
    target_context = target_context or resolve_target_context(
        backend_name,
        platform=platform,
        compute_arch=compute_arch,
    )

    if target_context.backend != backend_name:
        raise ValueError(
            f"Target context backend `{target_context.backend.value}` does not match "
            f"pass backend `{backend_name.value}`."
        )

    registry = _default_registry(
        pass_registry
        or (pass_pipeline.registry if isinstance(pass_pipeline, Pipeline) else None)
    )
    compiler_options = dict(compiler_options or {})
    kernel_metadata = dict(kernel_metadata or {})
    explicit_pass_options = _merge_pass_options(
        compiler_options.get("ssa_pass_options", {}),
        kernel_metadata.get("ssa_pass_options", {}),
        pass_options or {},
    )

    if isinstance(pass_pipeline, Pipeline):
        context = Context(
            backend=backend_name,
            target=target_context,
            compiler_options=compiler_options,
            kernel_metadata=kernel_metadata,
            tensors=tensors,
            pass_options=_merge_pass_options(
                pass_pipeline.spec.pass_options if pass_pipeline.spec else {},
                explicit_pass_options,
            ),
            pipeline_spec=pass_pipeline.spec,
        )

        lowered = _ensure_target_capabilities(
            pass_pipeline.run(program, context, registry=registry), context
        )

        return _with_metadata(
            lowered,
            target_backend=backend_name.value,
            target_platform=target_context.platform.name,
            target_compute_arch=target_context.compute_arch,
            target=target_context.as_metadata(),
            lowering_stage="scheduled-ssa",
        )

    configured_pipeline = (
        pass_pipeline
        or compiler_options.get("ssa_pass_pipeline")
        or kernel_metadata.get("ssa_pass_pipeline")
    )

    if configured_pipeline is None:
        pipeline = default_pipeline(backend_name, registry=registry)
        spec = pipeline.spec
    else:
        pipeline = None
        spec = _normalize_pipeline_spec(
            configured_pipeline,
            backend_name,
            registry,
        )

    if spec is None:
        raise ValueError("A pass pipeline must carry a PipelineSpec.")

    default_pass_options = dict(spec.pass_options)
    merged_pass_options = _merge_pass_options(spec.pass_options, explicit_pass_options)
    spec = PipelineSpec(
        passes=spec.passes,
        mode=spec.mode,
        pass_options=merged_pass_options,
        reason=spec.reason,
    )
    context = Context(
        backend=backend_name,
        target=target_context,
        compiler_options=compiler_options,
        kernel_metadata=kernel_metadata,
        tensors=tensors,
        pass_options=merged_pass_options,
        pipeline_spec=spec,
    )

    if pipeline is None or merged_pass_options != default_pass_options:
        pipeline = build(spec, backend=backend_name, registry=registry)

    lowered = _ensure_target_capabilities(
        pipeline.run(program, context, registry=registry), context
    )

    return _with_metadata(
        lowered,
        target_backend=backend_name.value,
        target_platform=target_context.platform.name,
        target_compute_arch=target_context.compute_arch,
        target=target_context.as_metadata(),
        lowering_stage="scheduled-ssa",
    )


def validate_for_target(
    program: ssa.Program,
    *,
    backend: Target | str,
    target_context: TargetContext,
    compiler_options: Mapping[str, Any],
    kernel_metadata: Mapping[str, Any],
) -> ssa.Program:
    """Revalidate final SSA at a backend boundary without rerunning optimizations."""
    context = Context(
        backend=normalize_target(backend),
        target=target_context,
        compiler_options=compiler_options,
        kernel_metadata=kernel_metadata,
    )

    return _ensure_target_capabilities(program, context)


def _normalize_pass_factory(
    pass_factory: type[Pass] | Callable[[], Pass] | Pass,
) -> Callable[[], Pass]:
    if isinstance(pass_factory, Pass):
        return lambda pass_=pass_factory: pass_

    if isinstance(pass_factory, type) and issubclass(pass_factory, Pass):
        return pass_factory
    return pass_factory


def _normalize_pipeline_spec(
    spec: PipelineSpec | Sequence[str] | Mapping[str, Any] | None,
    backend: Target,
    registry: Registry,
) -> PipelineSpec:
    if spec is None:
        return default_spec(backend, registry=registry)

    if isinstance(spec, PipelineSpec):
        spec = PipelineSpec(
            passes=_ensure_target_capability_pass(spec.passes),
            mode=spec.mode,
            pass_options=spec.pass_options,
            reason=spec.reason,
        )
        _validate_passes(spec.passes, backend, registry)

        return spec

    if isinstance(spec, Mapping):
        passes = spec.get("passes")

        if passes is None:
            passes = _default_pass_names(backend, registry)

        normalized = PipelineSpec(
            passes=_ensure_target_capability_pass(tuple(str(name) for name in passes)),
            mode=str(spec.get("mode", "custom")),
            pass_options=spec.get("pass_options", {}),
            reason=spec.get("reason"),
        )
        _validate_passes(normalized.passes, backend, registry)

        return normalized

    normalized = PipelineSpec(
        passes=_ensure_target_capability_pass(tuple(str(name) for name in spec)),
        mode="custom",
        reason="explicit custom pass sequence",
    )
    _validate_passes(normalized.passes, backend, registry)

    return normalized


def _validate_passes(
    pass_names: Sequence[str],
    backend: Target,
    registry: Registry,
) -> None:
    for name in pass_names:
        descriptor = registry.get(name)

        if not descriptor.supports(backend):
            raise ValueError(
                f"SSA pass `{name}` does not support backend `{backend.value}`."
            )


def _default_pass_names(backend: Target, registry: Registry) -> tuple[str, ...]:
    return tuple(
        descriptor.name
        for descriptor in registry.descriptors(backend=backend)
        if descriptor.default_enabled
    )


def _ensure_target_capability_pass(passes: tuple[str, ...]) -> tuple[str, ...]:
    name = ValidateTargetCapabilities.name

    return tuple(pass_name for pass_name in passes if pass_name != name) + (name,)


def _ensure_target_capabilities(
    program: ssa.Program,
    context: Context,
) -> ssa.Program:
    validated = ValidateTargetCapabilities().run(program, context)
    ssa.verify_program(validated)
    pass_trace = tuple(validated.metadata.get("pass_trace", ()))

    if not pass_trace or pass_trace[-1] != ValidateTargetCapabilities.name:
        pass_trace += (ValidateTargetCapabilities.name,)

    return _with_metadata(
        validated,
        pass_trace=pass_trace,
    )


def _required_capabilities(program: ssa.Program) -> tuple[str, ...]:
    required = set(_capability_values(program.metadata.get("required_capabilities")))

    for type_ in _program_value_types(program).values():
        dtype = str(type_.dtype or "").lower()

        if dtype.startswith("float8") or dtype.startswith("fp8"):
            required.add("dtype.fp8")

    def visit_block(block: ssa.Block) -> None:
        for operation in block.operations:
            required.update(
                _capability_values(operation.attrs.get("required_capabilities"))
            )

            if operation.opcode in {"arith.pow", "math.pow"}:
                required.add("math.pow")

            if operation.opcode == "reduce.min":
                required.add("reduction.min")

            for region in operation.regions:
                visit_block(region)

    for block in program.blocks:
        visit_block(block)
    return tuple(sorted(required))


def _capability_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _deduplicate_candidates(
    candidates: tuple[ScheduleCandidate, ...],
) -> tuple[ScheduleCandidate, ...]:
    unique = []
    seen = set()

    for candidate in candidates:
        key = repr(candidate.as_metadata())

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _candidate_rejection_reason(
    candidate: ScheduleCandidate,
    analysis: Mapping[str, Any],
    context: Context,
) -> str | None:
    constraints = dict(candidate.constraints)
    backend_options = dict(context.compiler_options.get("backend_options", {}))
    required_capability = constraints.get("minimum_compute_capability")
    actual_capability = backend_options.get("compute_capability")

    if required_capability is not None and actual_capability is not None:
        required = tuple(int(part) for part in str(required_capability).split("."))
        actual = tuple(int(part) for part in str(actual_capability).split("."))

        if actual < required:
            return (
                f"requires compute capability {required_capability}, "
                f"target is {actual_capability}"
            )

    dtype_constraint = constraints.get("dtypes")

    if dtype_constraint is not None:
        input_dtypes = set(analysis.get("dot_input_dtypes", ()))

        if input_dtypes and not input_dtypes <= set(dtype_constraint):
            return (
                f"dtypes {sorted(input_dtypes)} are outside {sorted(dtype_constraint)}"
            )

    max_threads = backend_options.get("max_threads_per_block")
    threads = candidate.schedule.get("threads")

    if (
        max_threads is not None
        and threads is not None
        and int(threads) > int(max_threads)
    ):
        return f"uses {threads} threads, target limit is {max_threads}"

    max_shared = backend_options.get("max_shared_memory_bytes")
    shared = candidate.constraints.get(
        "shared_memory_bytes", candidate.schedule.get("shared_memory_bytes")
    )

    if max_shared is not None and shared is not None and int(shared) > int(max_shared):
        return f"uses {shared} shared-memory bytes, target limit is {max_shared}"
    return None


def _merge_pass_options(
    *options: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for option_map in options:
        for pass_name, pass_option in dict(option_map or {}).items():
            if not isinstance(pass_option, Mapping):
                raise ValueError(
                    f"Options for SSA pass `{pass_name}` must be a mapping."
                )

            merged[str(pass_name)] = _merge_nested(
                merged.get(str(pass_name), {}), dict(pass_option)
            )
    return merged


def _pass_options(context: Context, *names: str) -> Mapping[str, Any]:
    merged: Mapping[str, Any] = {}

    for name in ("*", *names):
        merged = _merge_nested(dict(merged), dict(context.pass_options.get(name, {})))
    return merged


def _merge_nested(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(left)

    for key, value in dict(right).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _iter_opcodes(program: ssa.Program) -> tuple[str, ...]:
    opcodes: list[str] = []

    def visit_block(block: ssa.Block) -> None:
        for operation in block.operations:
            opcodes.append(operation.opcode)

            for region in operation.regions:
                visit_block(region)

    for block in program.blocks:
        visit_block(block)
    return tuple(opcodes)


def _linalg_input_dtypes(program: ssa.Program) -> tuple[str, ...]:
    value_types = _program_value_types(program)
    dtypes: list[str] = []

    def visit_block(block: ssa.Block) -> None:
        for operation in block.operations:
            if operation.opcode in {"linalg.dot", "linalg.matmul"}:
                for operand in operation.operands[:2]:
                    dtype = value_types.get(operand, ssa.Type(kind="unknown")).dtype

                    if dtype is not None:
                        dtypes.append(normalize_dtype(dtype))

            for region in operation.regions:
                visit_block(region)

    for block in program.blocks:
        visit_block(block)
    return tuple(dtypes)


def _has_exp_reduction_dot_pattern(opcodes: tuple[str, ...]) -> bool:
    required = {
        "scf.for",
        "linalg.dot",
        "reduce.max",
        "reduce.sum",
    }

    return required.issubset(set(opcodes)) and any(
        opcode in {"math.exp", "math.exp2"} for opcode in opcodes
    )


def _schedule_granularity(analysis: Mapping[str, Any]) -> str:
    layout_transfer = analysis.get("layout_transfer")

    if isinstance(layout_transfer, LayoutTransfer) and layout_transfer.schedulable:
        return "layout-transfer"

    if analysis.get("has_exp_reduction_dot_pattern"):
        return "exp-reduction-dot-region"

    if analysis.get("has_dot"):
        return "blocked-linalg"

    if analysis.get("reduction_count"):
        return "parallel-reduction"
    return "elementwise-grid"


def _map_block(block: ssa.Block, fn) -> ssa.Block:
    return ssa.Block(
        name=block.name,
        args=block.args,
        operations=tuple(
            _map_operation(operation, fn) for operation in block.operations
        ),
    )


def _map_operation(operation: ssa.Operation, fn) -> ssa.Operation:
    mapped_regions = tuple(_map_block(region, fn) for region in operation.regions)
    operation = ssa.Operation(
        opcode=operation.opcode,
        operands=operation.operands,
        results=operation.results,
        attrs=operation.attrs,
        regions=mapped_regions,
    )

    return fn(operation)


def _decompose_linalg_block(
    block: ssa.Block,
    parent_value_types: Mapping[str, ssa.Type],
) -> ssa.Block:
    value_types = dict(parent_value_types)
    value_types.update({arg.name: arg.type for arg in block.args})

    for operation in block.operations:
        value_types.update({result.name: result.type for result in operation.results})

    existing_names = {
        value.name for operation in block.operations for value in operation.results
    } | {arg.name for arg in block.args}
    temp_index = _next_temp_index(existing_names)
    transposes = {
        operation.results[0].name: operation
        for operation in block.operations
        if operation.opcode == "linalg.transpose"
        and len(operation.operands) == 1
        and len(operation.results) == 1
        and not operation.regions
    }
    consumed_transposes = {
        operation.operands[0]
        for operation in block.operations
        if operation.opcode == "mem.store"
        and operation.operands
        and operation.operands[0] in transposes
    }
    matmuls = {
        operation.results[0].name: operation
        for operation in block.operations
        if operation.opcode in {"linalg.matmul", "linalg.dot"}
        and len(operation.operands) == 2
        and len(operation.results) == 1
        and not operation.regions
    }
    consumed_matmuls = {
        operation.operands[0]
        for operation in block.operations
        if operation.opcode == "mem.store"
        and operation.operands
        and operation.operands[0] in matmuls
    }
    operations: list[ssa.Operation] = []

    for operation in block.operations:
        if (
            operation.opcode == "linalg.transpose"
            and operation.results
            and operation.results[0].name in consumed_transposes
        ):
            continue

        if (
            operation.opcode in {"linalg.matmul", "linalg.dot"}
            and operation.results
            and operation.results[0].name in consumed_matmuls
        ):
            continue

        if (
            operation.opcode == "mem.store"
            and operation.operands
            and operation.operands[0] in transposes
        ):
            transpose = transposes[operation.operands[0]]
            source = transpose.operands[0]
            output = operation.operands[1]
            col, temp_index = _fresh_value(
                existing_names, temp_index, ssa.Type(kind="index")
            )
            row, temp_index = _fresh_value(
                existing_names, temp_index, ssa.Type(kind="index")
            )
            value, temp_index = _fresh_value(
                existing_names,
                temp_index,
                ssa.Type(
                    kind="scalar",
                    dtype=transpose.results[0].type.dtype,
                ),
            )
            operations.extend(
                (
                    ssa.Operation(
                        opcode="index.offset",
                        operands=(output,),
                        results=(col,),
                        attrs={"dim": 0, "decomposition": "transpose"},
                    ),
                    ssa.Operation(
                        opcode="index.offset",
                        operands=(output,),
                        results=(row,),
                        attrs={"dim": 1, "decomposition": "transpose"},
                    ),
                    ssa.Operation(
                        opcode="tensor.extract",
                        operands=(source, row.name, col.name),
                        results=(value,),
                        attrs={"decomposition": "transpose", "source": True},
                    ),
                    ssa.Operation(
                        opcode="mem.store",
                        operands=(value.name, output),
                        attrs=dict(operation.attrs)
                        | {
                            "target": operation.attrs.get("target", output),
                            "decomposition": "transpose",
                        },
                    ),
                )
            )
            continue

        if (
            operation.opcode == "mem.store"
            and operation.operands
            and operation.operands[0] in matmuls
        ):
            matmul = matmuls[operation.operands[0]]
            operations.extend(
                _decompose_matmul_store(
                    matmul,
                    operation,
                    value_types,
                    existing_names,
                    temp_index,
                )[0]
            )
            temp_index = _next_temp_index(existing_names)
            continue

        regions = tuple(
            _decompose_linalg_block(region, value_types) for region in operation.regions
        )
        operations.append(
            ssa.Operation(
                opcode=operation.opcode,
                operands=operation.operands,
                results=operation.results,
                attrs=operation.attrs,
                regions=regions,
            )
        )
    return ssa.Block(name=block.name, args=block.args, operations=tuple(operations))


def _decompose_matmul_store(
    matmul: ssa.Operation,
    store: ssa.Operation,
    value_types: Mapping[str, ssa.Type],
    existing_names: set[str],
    temp_index: int,
) -> tuple[tuple[ssa.Operation, ...], int]:
    lhs, rhs = matmul.operands
    output = store.operands[1]
    m, n, k = _infer_matmul_symbols(matmul, lhs, rhs, output, value_types)
    output_type = value_types.get(output, matmul.results[0].type)
    scalar_type = _scalar_type(output_type)

    row, temp_index = _fresh_value(existing_names, temp_index, ssa.Type(kind="index"))
    col, temp_index = _fresh_value(existing_names, temp_index, ssa.Type(kind="index"))
    zero, temp_index = _fresh_value(
        existing_names, temp_index, ssa.Type(kind="scalar", dtype="int64")
    )
    one, temp_index = _fresh_value(
        existing_names, temp_index, ssa.Type(kind="scalar", dtype="int64")
    )
    acc_init, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    kk = ssa.Value(name="%kk", type=ssa.Type(kind="index"))
    acc_iter = ssa.Value(name="%acc_iter", type=scalar_type)
    lhs_value, temp_index = _fresh_value(
        existing_names, temp_index, _scalar_type(value_types.get(lhs, output_type))
    )
    rhs_value, temp_index = _fresh_value(
        existing_names, temp_index, _scalar_type(value_types.get(rhs, output_type))
    )
    product, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    acc_next, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    acc_result, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    loop = ssa.Operation(
        opcode="scf.for",
        operands=(zero.name, k, one.name, acc_init.name),
        results=(acc_result,),
        attrs={
            "induction": kk.name,
            "iter_args": (
                {
                    "name": "acc",
                    "initial": acc_init.name,
                    "block_arg": acc_iter.name,
                },
            ),
            "decomposition": "matmul",
            "m": m,
            "n": n,
            "k": k,
        },
        regions=(
            ssa.Block(
                name="matmul_k",
                args=(kk, acc_iter),
                operations=(
                    ssa.Operation(
                        opcode="tensor.extract",
                        operands=(lhs, row.name, kk.name),
                        results=(lhs_value,),
                        attrs={"decomposition": "matmul", "operand": "lhs"},
                    ),
                    ssa.Operation(
                        opcode="tensor.extract",
                        operands=(rhs, kk.name, col.name),
                        results=(rhs_value,),
                        attrs={"decomposition": "matmul", "operand": "rhs"},
                    ),
                    ssa.Operation(
                        opcode="arith.mul",
                        operands=(lhs_value.name, rhs_value.name),
                        results=(product,),
                        attrs={"decomposition": "matmul"},
                    ),
                    ssa.Operation(
                        opcode="arith.add",
                        operands=(acc_iter.name, product.name),
                        results=(acc_next,),
                        attrs={"decomposition": "matmul"},
                    ),
                    ssa.Operation(opcode="scf.yield", operands=(acc_next.name,)),
                ),
            ),
        ),
    )

    return (
        (
            ssa.Operation(
                opcode="index.offset",
                operands=(output,),
                results=(row,),
                attrs={"dim": 0, "decomposition": "matmul"},
            ),
            ssa.Operation(
                opcode="index.offset",
                operands=(output,),
                results=(col,),
                attrs={"dim": 1, "decomposition": "matmul"},
            ),
            ssa.Operation(
                opcode="arith.constant",
                results=(zero,),
                attrs={"value": 0, "decomposition": "matmul"},
            ),
            ssa.Operation(
                opcode="arith.constant",
                results=(one,),
                attrs={"value": 1, "decomposition": "matmul"},
            ),
            ssa.Operation(
                opcode="arith.constant",
                results=(acc_init,),
                attrs={"value": 0.0, "decomposition": "matmul"},
            ),
            loop,
            ssa.Operation(
                opcode="mem.store",
                operands=(acc_result.name, output),
                attrs=dict(store.attrs)
                | {
                    "target": store.attrs.get("target", output),
                    "decomposition": "matmul",
                },
            ),
        ),
        temp_index,
    )


def _program_value_types(program: ssa.Program) -> dict[str, ssa.Type]:
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }

    for block in program.blocks:
        _collect_block_value_types(block, value_types)
    return value_types


def _collect_block_value_types(
    block: ssa.Block, value_types: dict[str, ssa.Type]
) -> None:
    value_types.update({arg.name: arg.type for arg in block.args})

    for operation in block.operations:
        value_types.update({result.name: result.type for result in operation.results})

        for region in operation.regions:
            _collect_block_value_types(region, value_types)


def _infer_matmul_symbols(
    operation: ssa.Operation,
    lhs: str,
    rhs: str,
    output: str,
    value_types: Mapping[str, ssa.Type],
) -> tuple[str, str, str]:
    lhs_shape = tuple(
        str(dim) for dim in value_types.get(lhs, ssa.Type(kind="tensor")).shape
    )
    rhs_shape = tuple(
        str(dim) for dim in value_types.get(rhs, ssa.Type(kind="tensor")).shape
    )
    output_shape = tuple(
        str(dim) for dim in value_types.get(output, ssa.Type(kind="tensor")).shape
    )
    m = _first_symbol(
        operation.attrs.get("m"),
        _shape_dim(output_shape, 0),
        _shape_dim(lhs_shape, 0),
        "m",
    )
    n = _first_symbol(
        operation.attrs.get("n"),
        _shape_dim(output_shape, 1),
        _shape_dim(rhs_shape, 1),
        "n",
    )
    k = _first_symbol(
        operation.attrs.get("k"),
        _shape_dim(lhs_shape, -1),
        _shape_dim(rhs_shape, 0),
        "k",
    )

    return m, n, k


def _shape_dim(shape: tuple[str, ...], index: int) -> str | None:
    if not shape:
        return None

    try:
        return shape[index]
    except IndexError:
        return None


def _first_symbol(*candidates: object) -> str:
    for candidate in candidates:
        if candidate is None:
            continue

        text = str(candidate)

        if text.isidentifier():
            return text
    return str(candidates[-1])


def _scalar_type(type_: ssa.Type) -> ssa.Type:
    return ssa.Type(kind="scalar", dtype=type_.dtype or "float32")


def _next_temp_index(existing_names: set[str]) -> int:
    index = 0

    while f"%{index}" in existing_names:
        index += 1
    return index


def _fresh_value(
    existing_names: set[str],
    temp_index: int,
    type_: ssa.Type,
) -> tuple[ssa.Value, int]:
    while f"%{temp_index}" in existing_names:
        temp_index += 1

    name = f"%{temp_index}"
    existing_names.add(name)

    return ssa.Value(name=name, type=type_), temp_index + 1


def _with_metadata(program: ssa.Program, **metadata: Any) -> ssa.Program:
    return _replace_program(program, metadata=dict(program.metadata) | metadata)


def _replace_program(
    program: ssa.Program,
    *,
    blocks: tuple[ssa.Block, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ssa.Program:
    return replace(
        program,
        blocks=program.blocks if blocks is None else blocks,
        metadata=program.metadata if metadata is None else dict(metadata),
    )


_DEFAULT_REGISTRY: Registry | None = None


def _default_registry(registry: Registry | None = None) -> Registry:
    """Return the lazily initialized registry without backend import cycles."""
    global _DEFAULT_REGISTRY

    if registry is not None:
        return registry

    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()
    return _DEFAULT_REGISTRY
