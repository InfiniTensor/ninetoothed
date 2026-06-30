"""SSA optimization and target-lowering pass infrastructure.

The passes in this module operate on fine-grained SSA operations: loops,
arithmetic, reductions, tensor loads/stores, and linalg-style primitives.
They do not introduce whole-operator IR nodes.

Pass execution is intentionally organized like a compiler pipeline:

* hardware-independent passes canonicalize, analyze, and attach generic
  schedule intent to SSA;
* backend-specific passes, registered from ``ninetoothed.backends``, attach
  target policies such as tiling, memory scopes, and intrinsic choices.

The pass registry is the public control point for default pipelines, custom
pipelines, and policy-based autotune pipeline selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping

from ninetoothed.backends.base import BackendName, normalize_backend_name
from ninetoothed.ir import (
    SSABlockIR,
    SSAOperationIR,
    SSAProgramIR,
    SSATypeIR,
    SSAValueIR,
)

HARDWARE_INDEPENDENT = "hardware_independent"
HARDWARE_DEPENDENT = "hardware_dependent"
BACKEND_SPECIFIC = "backend_specific"


@dataclass(frozen=True)
class SSAPipelineSpec:
    """Declarative pass pipeline configuration."""

    passes: tuple[str, ...]
    mode: str = "default"
    pass_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    candidate_pipelines: tuple[tuple[str, ...], ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class SSAPassContext:
    """Context shared by SSA passes."""

    backend: BackendName
    compiler_options: Mapping[str, Any]
    kernel_metadata: Mapping[str, Any]
    pass_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    pipeline_spec: SSAPipelineSpec | None = None


class SSAPass:
    """Base class for semantics-preserving SSA transforms."""

    name = "ssa.pass"
    category = HARDWARE_INDEPENDENT
    phase = "generic"
    supported_backends: tuple[BackendName, ...] = ()

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        raise NotImplementedError


@dataclass(frozen=True)
class SSAPassDescriptor:
    """Registry metadata for one SSA pass."""

    name: str
    category: str
    phase: str
    factory: Callable[[], SSAPass]
    supported_backends: tuple[BackendName, ...] = ()
    default_enabled: bool = True
    description: str = ""
    tags: tuple[str, ...] = ()

    def create(self) -> SSAPass:
        return self.factory()

    def supports(self, backend: BackendName | str | None) -> bool:
        if not self.supported_backends:
            return True
        backend_name = normalize_backend_name(backend)
        return backend_name in self.supported_backends


class SSAPassRegistry:
    """Registry that owns pass discovery and pipeline construction."""

    def __init__(self) -> None:
        self._descriptors: dict[str, SSAPassDescriptor] = {}

    def register(
        self,
        pass_factory: type[SSAPass] | Callable[[], SSAPass] | SSAPass,
        *,
        name: str | None = None,
        category: str | None = None,
        phase: str | None = None,
        supported_backends: Sequence[BackendName | str] | None = None,
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
            tuple(normalize_backend_name(backend) for backend in supported_backends)
            if supported_backends is not None
            else tuple(getattr(probe, "supported_backends", ()))
        )
        doc = description
        if doc is None:
            raw_doc = (probe.__doc__ or "").strip().splitlines()
            doc = raw_doc[0].strip() if raw_doc else ""

        self._descriptors[pass_name] = SSAPassDescriptor(
            name=pass_name,
            category=category or probe.category,
            phase=phase or probe.phase,
            factory=factory,
            supported_backends=backends,
            default_enabled=probe.default_enabled
            if hasattr(probe, "default_enabled") and default_enabled is None
            else bool(True if default_enabled is None else default_enabled),
            description=doc,
            tags=tuple(tags),
        )

    def get(self, name: str) -> SSAPassDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:
            available = ", ".join(self._descriptors)
            raise KeyError(
                f"Unknown SSA pass `{name}`. Available passes: {available}"
            ) from exc

    def descriptors(
        self,
        *,
        category: str | None = None,
        backend: BackendName | str | None = None,
    ) -> tuple[SSAPassDescriptor, ...]:
        result = tuple(self._descriptors.values())
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
        backend: BackendName | str | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            descriptor.name
            for descriptor in self.descriptors(category=category, backend=backend)
        )


class SSAPassPipeline:
    """Ordered pass pipeline with a textual trace in SSA metadata."""

    def __init__(
        self,
        passes: tuple[SSAPass, ...],
        *,
        descriptors: tuple[SSAPassDescriptor, ...] = (),
        spec: SSAPipelineSpec | None = None,
    ):
        self.passes = passes
        self.descriptors = descriptors
        self.spec = spec

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        current = _with_metadata(program, pipeline_selection=self._pipeline_metadata())
        for pass_ in self.passes:
            current = pass_.run(current, context)
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
            HARDWARE_DEPENDENT: tuple(
                descriptor.name
                for descriptor in self.descriptors
                if descriptor.category == HARDWARE_DEPENDENT
            ),
            BACKEND_SPECIFIC: tuple(
                descriptor.name
                for descriptor in self.descriptors
                if descriptor.category == BACKEND_SPECIFIC
            ),
        }
        return {
            "mode": self.spec.mode,
            "selected_passes": self.spec.passes,
            "candidate_pipelines": self.spec.candidate_pipelines,
            "reason": self.spec.reason,
            "categories": categories,
        }


class CanonicalizeSSAPass(SSAPass):
    """Normalize generic SSA into the canonical dialect used by all backends."""

    name = "ssa.canonicalize"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        return _with_metadata(
            program,
            dialect="generic-ssa",
            canonical=True,
            coarse_operator_nodes=False,
        )


class AnalyzeSSAEffectsPass(SSAPass):
    """Collect dataflow facts needed by schedule selection."""

    name = "ssa.analyze_effects"
    category = HARDWARE_INDEPENDENT
    phase = "analysis"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        opcodes = tuple(_iter_opcodes(program))
        stores = sum(1 for opcode in opcodes if opcode == "mem.store")
        reductions = sum(1 for opcode in opcodes if opcode.startswith("reduce."))
        loops = sum(1 for opcode in opcodes if opcode == "scf.for")
        return _with_metadata(
            program,
            analysis={
                "operation_count": len(opcodes),
                "store_count": stores,
                "reduction_count": reductions,
                "loop_count": loops,
                "has_dot": "linalg.dot" in opcodes or "linalg.matmul" in opcodes,
                "has_exp_reduction_dot_pattern": _has_exp_reduction_dot_pattern(
                    opcodes
                ),
            },
        )


class DecomposeLinalgPass(SSAPass):
    """Lower high-level linalg ops into index/extract/store SSA operations."""

    name = "ssa.decompose_linalg"
    category = HARDWARE_INDEPENDENT
    phase = "canonicalization"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
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


class SelectSchedulePass(SSAPass):
    """Attach backend-neutral schedule intent."""

    name = "ssa.select_schedule"
    category = HARDWARE_INDEPENDENT
    phase = "schedule_intent"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        analysis = dict(program.metadata.get("analysis", {}))
        options = _pass_options(context, self.name)
        schedule = {
            "granularity": _schedule_granularity(analysis),
            "indexing": "flat-contiguous",
            "parallelism": "program-blocks",
        }
        schedule = _merge_nested(schedule, options)
        return _with_metadata(program, schedule=schedule)


class BackendScheduleOptimizationPass(SSAPass):
    """Contract for backend-specific schedule optimization passes."""

    name = "ssa.optimize_schedule"
    category = BACKEND_SPECIFIC
    phase = "optimization"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        analysis = dict(program.metadata.get("analysis", {}))
        schedule = dict(program.metadata.get("schedule", {}))
        optimization = dict(
            self.optimization_policy(context.backend, analysis, schedule)
        )
        optimization = _merge_nested(
            optimization,
            _pass_options(context, self.name, "ssa.optimize_schedule"),
        )
        optimized_schedule = _merge_nested(
            schedule,
            dict(optimization.get("schedule", {})),
        )

        blocks = tuple(
            _map_block(
                block, lambda op: _annotate_operation(op, optimization=optimization)
            )
            for block in program.blocks
        )
        return _replace_program(
            program,
            blocks=blocks,
            metadata=dict(program.metadata)
            | {
                "schedule": optimized_schedule,
                "optimization": optimization,
            },
        )

    def optimization_policy(
        self,
        backend: BackendName,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class OptimizeSchedulePass(BackendScheduleOptimizationPass):
    """Backward-compatible base name for backend schedule passes."""


class BackendMemoryScopesLoweringPass(SSAPass):
    """Contract for backend-specific memory scope materialization passes."""

    name = "ssa.lower_memory_scopes"
    category = BACKEND_SPECIFIC
    phase = "target_lowering"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        scopes = dict(self.memory_scopes(context))
        return annotate_ssa_operations(
            program,
            attrs={"memory_scope": scopes},
            metadata={"memory_scope": scopes},
        )

    def memory_scopes(self, context: SSAPassContext) -> Mapping[str, str]:
        raise NotImplementedError


class BackendIntrinsicsLoweringPass(SSAPass):
    """Contract for backend-specific intrinsic materialization passes."""

    name = "ssa.lower_intrinsics"
    category = BACKEND_SPECIFIC
    phase = "target_lowering"

    def run(self, program: SSAProgramIR, context: SSAPassContext) -> SSAProgramIR:
        intrinsic = dict(self.intrinsics(context))
        return annotate_ssa_operations(
            program,
            attrs={"backend_intrinsic": intrinsic},
            metadata={
                "target_backend": context.backend.value,
                "backend_intrinsics": intrinsic,
                "lowering_stage": "target-annotated-ssa",
            },
        )

    def intrinsics(self, context: SSAPassContext) -> Mapping[str, str]:
        raise NotImplementedError


def create_default_ssa_pass_registry() -> SSAPassRegistry:
    registry = SSAPassRegistry()
    registry.register(CanonicalizeSSAPass, tags=("generic", "required"))
    registry.register(DecomposeLinalgPass, tags=("generic", "linalg", "decomposition"))
    registry.register(AnalyzeSSAEffectsPass, tags=("generic", "analysis", "required"))
    registry.register(SelectSchedulePass, tags=("schedule",))
    _register_backend_specific_ssa_passes(registry)
    return registry


def _register_backend_specific_ssa_passes(registry: SSAPassRegistry) -> None:
    from ninetoothed.backends.backend_pass_registry import (
        register_backend_specific_ssa_passes,
    )

    register_backend_specific_ssa_passes(registry)


def registered_ssa_passes(
    *,
    category: str | None = None,
    backend: BackendName | str | None = None,
    registry: SSAPassRegistry | None = None,
) -> tuple[SSAPassDescriptor, ...]:
    """Return registered SSA pass descriptors."""

    return (registry or DEFAULT_SSA_PASS_REGISTRY).descriptors(
        category=category, backend=backend
    )


def default_ssa_pipeline_spec(
    backend: BackendName | str | None,
    *,
    registry: SSAPassRegistry | None = None,
) -> SSAPipelineSpec:
    """Return the default declarative pipeline for a backend."""

    backend_name = normalize_backend_name(backend)
    pass_names = _default_pass_names(backend_name)
    _validate_passes(pass_names, backend_name, registry or DEFAULT_SSA_PASS_REGISTRY)
    return SSAPipelineSpec(
        passes=pass_names,
        mode="default",
        reason="default backend pipeline",
    )


def default_ssa_pipeline(backend: BackendName | str | None) -> SSAPassPipeline:
    """Return the default target-aware SSA lowering pipeline."""

    backend_name = normalize_backend_name(backend)
    return build_ssa_pipeline(
        default_ssa_pipeline_spec(backend_name), backend=backend_name
    )


def build_ssa_pipeline(
    spec: SSAPipelineSpec | Sequence[str] | Mapping[str, Any],
    *,
    backend: BackendName | str | None,
    registry: SSAPassRegistry | None = None,
) -> SSAPassPipeline:
    """Build an executable pass pipeline from a declarative spec."""

    backend_name = normalize_backend_name(backend)
    registry = registry or DEFAULT_SSA_PASS_REGISTRY
    normalized = _normalize_pipeline_spec(spec, backend_name, registry)
    descriptors = tuple(registry.get(name) for name in normalized.passes)
    passes = tuple(descriptor.create() for descriptor in descriptors)
    return SSAPassPipeline(passes, descriptors=descriptors, spec=normalized)


def autotune_ssa_pipeline_spec(
    program: SSAProgramIR,
    context: SSAPassContext,
    *,
    registry: SSAPassRegistry | None = None,
    autotune: bool | str | Mapping[str, Any] = True,
) -> SSAPipelineSpec:
    """Select a pipeline using a policy-based autotune planner.

    This does not run kernels. It records candidate pipelines and chooses the
    best static pipeline for the observed SSA shape. Runtime measurement can be
    layered on top by passing explicit ``passes`` and ``pass_options`` later.
    """

    registry = registry or DEFAULT_SSA_PASS_REGISTRY
    default_passes = _default_pass_names(context.backend)
    optimize_pass = _backend_optimize_pass_name(context.backend)
    no_backend_opt = tuple(name for name in default_passes if name != optimize_pass)
    candidates = (default_passes, no_backend_opt)
    opcodes = tuple(_iter_opcodes(program))
    granularity = _schedule_granularity(
        {
            "has_dot": "linalg.dot" in opcodes or "linalg.matmul" in opcodes,
            "has_exp_reduction_dot_pattern": _has_exp_reduction_dot_pattern(opcodes),
            "reduction_count": sum(
                1 for opcode in opcodes if opcode.startswith("reduce.")
            ),
        }
    )
    reason = (
        f"policy-autotune selected `{optimize_pass}` for backend={context.backend.value}, "
        f"ssa_granularity={granularity}"
    )

    pass_options: Mapping[str, Mapping[str, Any]] = {}
    if isinstance(autotune, Mapping):
        if "passes" in autotune:
            selected = tuple(str(name) for name in autotune["passes"])
            _validate_passes(selected, context.backend, registry)
            return SSAPipelineSpec(
                passes=selected,
                mode="autotune",
                pass_options=autotune.get("pass_options", {}),
                candidate_pipelines=candidates + (selected,),
                reason=str(autotune.get("reason", "explicit autotune pass override")),
            )
        pass_options = autotune.get("pass_options", {})

    _validate_passes(default_passes, context.backend, registry)
    return SSAPipelineSpec(
        passes=default_passes,
        mode="autotune",
        pass_options=pass_options,
        candidate_pipelines=candidates,
        reason=reason,
    )


def lower_ssa_for_backend(
    program: SSAProgramIR | None,
    *,
    backend: BackendName | str | None,
    compiler_options: Mapping[str, Any] | None = None,
    kernel_metadata: Mapping[str, Any] | None = None,
    pass_pipeline: SSAPassPipeline
    | SSAPipelineSpec
    | Sequence[str]
    | Mapping[str, Any]
    | None = None,
    pass_options: Mapping[str, Mapping[str, Any]] | None = None,
    autotune: bool | str | Mapping[str, Any] = False,
    pass_registry: SSAPassRegistry | None = None,
) -> SSAProgramIR | None:
    """Run an SSA pass pipeline for a backend."""

    if program is None:
        return None

    backend_name = normalize_backend_name(backend)
    registry = pass_registry or DEFAULT_SSA_PASS_REGISTRY
    compiler_options = dict(compiler_options or {})
    kernel_metadata = dict(kernel_metadata or {})
    explicit_pass_options = _merge_pass_options(
        compiler_options.get("ssa_pass_options", {}),
        kernel_metadata.get("ssa_pass_options", {}),
        pass_options or {},
    )
    base_context = SSAPassContext(
        backend=backend_name,
        compiler_options=compiler_options,
        kernel_metadata=kernel_metadata,
        pass_options=explicit_pass_options,
    )

    if isinstance(pass_pipeline, SSAPassPipeline):
        context = SSAPassContext(
            backend=backend_name,
            compiler_options=compiler_options,
            kernel_metadata=kernel_metadata,
            pass_options=explicit_pass_options,
        )
        return pass_pipeline.run(program, context)

    if pass_pipeline is None and _autotune_enabled(autotune):
        spec = autotune_ssa_pipeline_spec(
            program,
            base_context,
            registry=registry,
            autotune=autotune,
        )
    else:
        spec = _normalize_pipeline_spec(
            pass_pipeline
            or compiler_options.get("ssa_pass_pipeline")
            or kernel_metadata.get("ssa_pass_pipeline")
            or default_ssa_pipeline_spec(backend_name, registry=registry),
            backend_name,
            registry,
        )

    merged_pass_options = _merge_pass_options(spec.pass_options, explicit_pass_options)
    spec = SSAPipelineSpec(
        passes=spec.passes,
        mode=spec.mode,
        pass_options=merged_pass_options,
        candidate_pipelines=spec.candidate_pipelines,
        reason=spec.reason,
    )
    context = SSAPassContext(
        backend=backend_name,
        compiler_options=compiler_options,
        kernel_metadata=kernel_metadata,
        pass_options=merged_pass_options,
        pipeline_spec=spec,
    )
    return build_ssa_pipeline(spec, backend=backend_name, registry=registry).run(
        program, context
    )


def _normalize_pass_factory(
    pass_factory: type[SSAPass] | Callable[[], SSAPass] | SSAPass,
) -> Callable[[], SSAPass]:
    if isinstance(pass_factory, SSAPass):
        return lambda pass_=pass_factory: pass_
    if isinstance(pass_factory, type) and issubclass(pass_factory, SSAPass):
        return pass_factory
    return pass_factory


def _normalize_pipeline_spec(
    spec: SSAPipelineSpec | Sequence[str] | Mapping[str, Any] | None,
    backend: BackendName,
    registry: SSAPassRegistry,
) -> SSAPipelineSpec:
    if spec is None:
        return default_ssa_pipeline_spec(backend, registry=registry)
    if isinstance(spec, SSAPipelineSpec):
        _validate_passes(spec.passes, backend, registry)
        return spec
    if isinstance(spec, Mapping):
        passes = spec.get("passes")
        if passes is None:
            passes = _default_pass_names(backend)
        normalized = SSAPipelineSpec(
            passes=tuple(str(name) for name in passes),
            mode=str(spec.get("mode", "custom")),
            pass_options=spec.get("pass_options", {}),
            candidate_pipelines=tuple(
                tuple(candidate) for candidate in spec.get("candidate_pipelines", ())
            ),
            reason=spec.get("reason"),
        )
        _validate_passes(normalized.passes, backend, registry)
        return normalized
    normalized = SSAPipelineSpec(
        passes=tuple(str(name) for name in spec),
        mode="custom",
        reason="explicit custom pass sequence",
    )
    _validate_passes(normalized.passes, backend, registry)
    return normalized


def _validate_passes(
    pass_names: Sequence[str],
    backend: BackendName,
    registry: SSAPassRegistry,
) -> None:
    for name in pass_names:
        descriptor = registry.get(name)
        if not descriptor.supports(backend):
            raise ValueError(
                f"SSA pass `{name}` does not support backend `{backend.value}`."
            )


def _default_pass_names(backend: BackendName) -> tuple[str, ...]:
    return (
        "ssa.canonicalize",
        "ssa.analyze_effects",
        "ssa.select_schedule",
        _backend_optimize_pass_name(backend),
        "ssa.decompose_linalg",
        _backend_memory_pass_name(backend),
        _backend_intrinsics_pass_name(backend),
    )


def _backend_optimize_pass_name(backend: BackendName) -> str:
    return f"ssa.{backend.value}.optimize_schedule"


def _backend_memory_pass_name(backend: BackendName) -> str:
    return f"ssa.{backend.value}.lower_memory_scopes"


def _backend_intrinsics_pass_name(backend: BackendName) -> str:
    return f"ssa.{backend.value}.lower_intrinsics"


def _autotune_enabled(value: bool | str | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    if isinstance(value, str):
        return value.lower() not in {"", "0", "false", "none", "off"}
    return bool(value)


def _merge_pass_options(
    *options: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for option_map in options:
        for pass_name, pass_option in dict(option_map or {}).items():
            merged[str(pass_name)] = _merge_nested(
                merged.get(str(pass_name), {}), dict(pass_option)
            )
    return merged


def _pass_options(context: SSAPassContext, *names: str) -> Mapping[str, Any]:
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


def _iter_opcodes(program: SSAProgramIR) -> tuple[str, ...]:
    opcodes: list[str] = []

    def visit_block(block: SSABlockIR) -> None:
        for operation in block.operations:
            opcodes.append(operation.opcode)
            for region in operation.regions:
                visit_block(region)

    for block in program.blocks:
        visit_block(block)
    return tuple(opcodes)


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
    if analysis.get("has_exp_reduction_dot_pattern"):
        return "exp-reduction-dot-region"
    if analysis.get("has_dot"):
        return "blocked-linalg"
    if analysis.get("reduction_count"):
        return "parallel-reduction"
    return "elementwise-grid"


def annotate_ssa_operations(
    program: SSAProgramIR,
    *,
    attrs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> SSAProgramIR:
    """Return ``program`` with every operation annotated by ``attrs``."""

    blocks = tuple(
        _map_block(block, lambda op: _annotate_operation(op, **dict(attrs)))
        for block in program.blocks
    )
    return _replace_program(
        program,
        blocks=blocks,
        metadata=dict(program.metadata) | dict(metadata or {}),
    )


def _map_block(block: SSABlockIR, fn) -> SSABlockIR:
    return SSABlockIR(
        name=block.name,
        args=block.args,
        operations=tuple(
            _map_operation(operation, fn) for operation in block.operations
        ),
    )


def _map_operation(operation: SSAOperationIR, fn) -> SSAOperationIR:
    mapped_regions = tuple(_map_block(region, fn) for region in operation.regions)
    operation = SSAOperationIR(
        operation.opcode,
        operands=operation.operands,
        results=operation.results,
        attrs=operation.attrs,
        regions=mapped_regions,
    )
    return fn(operation)


def _decompose_linalg_block(
    block: SSABlockIR,
    parent_value_types: Mapping[str, SSATypeIR],
) -> SSABlockIR:
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
    operations: list[SSAOperationIR] = []
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
                existing_names, temp_index, SSATypeIR("index")
            )
            row, temp_index = _fresh_value(
                existing_names, temp_index, SSATypeIR("index")
            )
            value, temp_index = _fresh_value(
                existing_names,
                temp_index,
                transpose.results[0].type,
            )
            operations.extend(
                (
                    SSAOperationIR(
                        "index.offset",
                        operands=(output,),
                        results=(col,),
                        attrs={"dim": 0, "decomposition": "transpose"},
                    ),
                    SSAOperationIR(
                        "index.offset",
                        operands=(output,),
                        results=(row,),
                        attrs={"dim": 1, "decomposition": "transpose"},
                    ),
                    SSAOperationIR(
                        "tensor.extract",
                        operands=(source, row.name, col.name),
                        results=(value,),
                        attrs={"decomposition": "transpose"},
                    ),
                    SSAOperationIR(
                        "mem.store",
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
            SSAOperationIR(
                operation.opcode,
                operands=operation.operands,
                results=operation.results,
                attrs=operation.attrs,
                regions=regions,
            )
        )
    return SSABlockIR(name=block.name, args=block.args, operations=tuple(operations))


def _decompose_matmul_store(
    matmul: SSAOperationIR,
    store: SSAOperationIR,
    value_types: Mapping[str, SSATypeIR],
    existing_names: set[str],
    temp_index: int,
) -> tuple[tuple[SSAOperationIR, ...], int]:
    lhs, rhs = matmul.operands
    output = store.operands[1]
    m, n, k = _infer_matmul_symbols(matmul, lhs, rhs, output, value_types)
    output_type = value_types.get(output, matmul.results[0].type)
    scalar_type = _scalar_type(output_type)

    row, temp_index = _fresh_value(existing_names, temp_index, SSATypeIR("index"))
    col, temp_index = _fresh_value(existing_names, temp_index, SSATypeIR("index"))
    zero, temp_index = _fresh_value(
        existing_names, temp_index, SSATypeIR("scalar", dtype="int64")
    )
    one, temp_index = _fresh_value(
        existing_names, temp_index, SSATypeIR("scalar", dtype="int64")
    )
    acc_init, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    kk = SSAValueIR("%kk", SSATypeIR("index"))
    acc_iter = SSAValueIR("%acc_iter", scalar_type)
    lhs_value, temp_index = _fresh_value(
        existing_names, temp_index, _scalar_type(value_types.get(lhs, output_type))
    )
    rhs_value, temp_index = _fresh_value(
        existing_names, temp_index, _scalar_type(value_types.get(rhs, output_type))
    )
    product, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    acc_next, temp_index = _fresh_value(existing_names, temp_index, scalar_type)
    acc_result, temp_index = _fresh_value(existing_names, temp_index, scalar_type)

    loop = SSAOperationIR(
        "scf.for",
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
            SSABlockIR(
                name="matmul_k",
                args=(kk, acc_iter),
                operations=(
                    SSAOperationIR(
                        "tensor.extract",
                        operands=(lhs, row.name, kk.name),
                        results=(lhs_value,),
                        attrs={"decomposition": "matmul", "operand": "lhs"},
                    ),
                    SSAOperationIR(
                        "tensor.extract",
                        operands=(rhs, kk.name, col.name),
                        results=(rhs_value,),
                        attrs={"decomposition": "matmul", "operand": "rhs"},
                    ),
                    SSAOperationIR(
                        "arith.mul",
                        operands=(lhs_value.name, rhs_value.name),
                        results=(product,),
                        attrs={"decomposition": "matmul"},
                    ),
                    SSAOperationIR(
                        "arith.add",
                        operands=(acc_iter.name, product.name),
                        results=(acc_next,),
                        attrs={"decomposition": "matmul"},
                    ),
                    SSAOperationIR("scf.yield", operands=(acc_next.name,)),
                ),
            ),
        ),
    )
    return (
        (
            SSAOperationIR(
                "index.offset",
                operands=(output,),
                results=(row,),
                attrs={"dim": 0, "decomposition": "matmul"},
            ),
            SSAOperationIR(
                "index.offset",
                operands=(output,),
                results=(col,),
                attrs={"dim": 1, "decomposition": "matmul"},
            ),
            SSAOperationIR(
                "arith.constant",
                results=(zero,),
                attrs={"value": 0, "decomposition": "matmul"},
            ),
            SSAOperationIR(
                "arith.constant",
                results=(one,),
                attrs={"value": 1, "decomposition": "matmul"},
            ),
            SSAOperationIR(
                "arith.constant",
                results=(acc_init,),
                attrs={"value": 0.0, "decomposition": "matmul"},
            ),
            loop,
            SSAOperationIR(
                "mem.store",
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


def _program_value_types(program: SSAProgramIR) -> dict[str, SSATypeIR]:
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }
    for block in program.blocks:
        _collect_block_value_types(block, value_types)
    return value_types


def _collect_block_value_types(
    block: SSABlockIR, value_types: dict[str, SSATypeIR]
) -> None:
    value_types.update({arg.name: arg.type for arg in block.args})
    for operation in block.operations:
        value_types.update({result.name: result.type for result in operation.results})
        for region in operation.regions:
            _collect_block_value_types(region, value_types)


def _infer_matmul_symbols(
    operation: SSAOperationIR,
    lhs: str,
    rhs: str,
    output: str,
    value_types: Mapping[str, SSATypeIR],
) -> tuple[str, str, str]:
    lhs_shape = tuple(
        str(dim) for dim in value_types.get(lhs, SSATypeIR("tensor")).shape
    )
    rhs_shape = tuple(
        str(dim) for dim in value_types.get(rhs, SSATypeIR("tensor")).shape
    )
    output_shape = tuple(
        str(dim) for dim in value_types.get(output, SSATypeIR("tensor")).shape
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


def _scalar_type(type_: SSATypeIR) -> SSATypeIR:
    return SSATypeIR("scalar", dtype=type_.dtype or "float32")


def _next_temp_index(existing_names: set[str]) -> int:
    index = 0
    while f"%{index}" in existing_names:
        index += 1
    return index


def _fresh_value(
    existing_names: set[str],
    temp_index: int,
    type_: SSATypeIR,
) -> tuple[SSAValueIR, int]:
    while f"%{temp_index}" in existing_names:
        temp_index += 1
    name = f"%{temp_index}"
    existing_names.add(name)
    return SSAValueIR(name, type_), temp_index + 1


def _annotate_operation(operation: SSAOperationIR, **attrs: Any) -> SSAOperationIR:
    return SSAOperationIR(
        operation.opcode,
        operands=operation.operands,
        results=operation.results,
        attrs=dict(operation.attrs) | attrs,
        regions=operation.regions,
    )


def _with_metadata(program: SSAProgramIR, **metadata: Any) -> SSAProgramIR:
    return _replace_program(program, metadata=dict(program.metadata) | metadata)


def _replace_program(
    program: SSAProgramIR,
    *,
    blocks: tuple[SSABlockIR, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SSAProgramIR:
    return SSAProgramIR(
        kind=program.kind,
        inputs=program.inputs,
        outputs=program.outputs,
        blocks=program.blocks if blocks is None else blocks,
        metadata=program.metadata if metadata is None else dict(metadata),
    )


DEFAULT_SSA_PASS_REGISTRY = create_default_ssa_pass_registry()
