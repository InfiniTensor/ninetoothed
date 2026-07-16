"""Unified NineToothed SSA compiler driver."""

import copy
import inspect
import itertools
import math
import os
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ninetoothed.backends import default_registry, emit
from ninetoothed.backends.core import (
    Artifact,
    Target,
    normalize_target,
)
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import LoweringError, from_application
from ninetoothed.ir import IndexExpr, Kernel, LaunchABI, LaunchBinding, LaunchPlan
from ninetoothed.naming import is_meta, remove_prefixes

from .specialization import (
    is_schedule_tile_parameter,
    scheduled_meta_defaults,
    specialize_program,
    specialize_tensor_specs,
)


@dataclass(frozen=True, kw_only=True)
class CompileRequest:
    arrangement: Any | None = None
    application: Any
    tensors: tuple[Any, ...] = ()
    backend: Target | str | None = None
    caller: str = "torch"
    kernel_name: str | None = None
    num_warps: int | tuple[int, ...] | None = None
    num_stages: int | tuple[int, ...] | None = None
    max_num_configs: int | None = None
    pipeline: Any | None = None
    pass_options: Mapping[str, Mapping[str, Any]] | None = None
    backend_options: Mapping[str, Any] | None = None
    tensor_dtypes: Mapping[str, str] | None = None
    specialization_values: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class Compilation:
    request: CompileRequest
    kernel: Kernel
    artifact: Artifact
    launch_plan: LaunchPlan
    pass_trace: tuple[str, ...]

    @property
    def launch_abi(self) -> LaunchABI:
        """Return the public ABI carried by the structured launch plan."""
        return self.launch_plan.abi


class Compiler:
    """Facade for pure SSA compilation and optional runtime materialization."""

    def compile(self, request: CompileRequest) -> Compilation:
        return _compile_kernel(request)

    def materialize(self, request, *, output_dir=None, mode="jit"):
        compilation = (
            request if isinstance(request, Compilation) else self.compile(request)
        )
        from ninetoothed.compiler.runtime import materialize

        return materialize(compilation, output_dir=output_dir, mode=mode)


DEFAULT_COMPILER = Compiler()


def resolve_target(backend: Target | str | None) -> Target:
    if backend is None:
        backend = os.environ.get("NINETOOTHED_BACKEND")
    return normalize_target(backend)


def compile_kernel(request: CompileRequest) -> Compilation:
    """Compile a request with the process-wide default compiler."""
    return DEFAULT_COMPILER.compile(request)


def aot(
    func,
    *,
    backend=None,
    caller="cuda",
    kernel_name=None,
    output_dir,
    num_warps=None,
    num_stages=None,
    pipeline=None,
    pass_options=None,
    **backend_options,
):
    """Compile an annotated application and materialize its backend artifact."""
    return DEFAULT_COMPILER.materialize(
        CompileRequest(
            application=func,
            backend=backend,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            pipeline=pipeline,
            pass_options=pass_options,
            backend_options=backend_options,
        ),
        output_dir=output_dir,
        mode="aot",
    )


def make(
    arrangement,
    application,
    tensors,
    *,
    backend=None,
    caller="torch",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    pipeline=None,
    pass_options=None,
    **backend_options,
):
    """Arrange, compile, and materialize a NineToothed application."""
    mode = "jit" if caller == "torch" else "aot"

    return DEFAULT_COMPILER.materialize(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tuple(tensors),
            backend=backend,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
            pipeline=pipeline,
            pass_options=pass_options,
            backend_options=backend_options,
        ),
        output_dir=output_dir,
        mode=mode,
    )


def lower(
    arrangement,
    application,
    tensors,
    *,
    backend: str | None = None,
    caller: str = "torch",
    kernel_name: str | None = None,
    output_dir: str | Path | None = None,
    num_warps: int | tuple[int, ...] | None = None,
    num_stages: int | tuple[int, ...] | None = None,
    max_num_configs: int | None = None,
    pipeline: Any | None = None,
    pass_options: Mapping[str, Mapping[str, Any]] | None = None,
    write: bool = False,
    **backend_options: Any,
) -> Artifact:
    """Lower a NineToothed kernel without initializing a backend runtime."""
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tuple(tensors),
            backend=backend,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
            pipeline=pipeline,
            pass_options=pass_options,
            backend_options=backend_options,
        )
    )
    artifact = compilation.artifact

    if write:
        if output_dir is None:
            raise ValueError("Output directory is required when `write=True`.")

        artifact.write_to(output_dir)
    return artifact


def _compile_kernel(request: CompileRequest) -> Compilation:
    """Compile an application through SSA and a selected backend, without fallback."""
    application = request.application
    params = tuple(inspect.signature(application).parameters)

    if request.arrangement is None:
        annotations = inspect.get_annotations(application, eval_str=False)

        try:
            arranged = tuple(copy.deepcopy(annotations[name]) for name in params)
        except KeyError as exc:
            raise LoweringError(
                f"Cannot lower `{application.__name__}`: parameter `{exc.args[0]}` "
                "does not have a Tensor annotation."
            ) from exc
    else:
        symbolic_tensors = copy.deepcopy(request.tensors)
        arranged_value = request.arrangement(*symbolic_tensors)
        arranged = (
            arranged_value if isinstance(arranged_value, tuple) else (arranged_value,)
        )

    if len(arranged) != len(params):
        raise LoweringError(
            f"Cannot lower `{application.__name__}`: arrangement returned "
            f"{len(arranged)} values for {len(params)} parameters."
        )

    for name, tensor in zip(params, arranged):
        dtype = (request.tensor_dtypes or {}).get(name)

        if dtype is not None:
            getattr(tensor, "source", tensor).dtype = dtype

    specs = tensor_specs(params, arranged)
    kernel_name = request.kernel_name or application.__name__
    meta_defaults = _meta_defaults(arranged)

    try:
        program = from_application(application, specs, kind=kernel_name, strict=True)
    except LoweringError as exc:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: {exc}."
        ) from exc

    if program is None:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: "
            "source inspection did not produce ssa.Program."
        )

    target = resolve_target(request.backend)
    _validate_tuning_options(target, request)
    specialization_values = dict(request.specialization_values or {})

    if target != Target.TRITON:
        specialization_values = {
            name: value
            for name, value in specialization_values.items()
            if not is_schedule_tile_parameter(name)
        }

    if specialization_values:
        specs = specialize_tensor_specs(specs, specialization_values)
        program = specialize_program(program, specialization_values)

    backend_options = (
        default_registry()
        .get(target)
        .normalize_options(dict(request.backend_options or {}))
    )
    request = replace(request, backend_options=backend_options)
    kernel = Kernel(
        kernel_name=kernel_name,
        source=_source(application),
        source_language="ninetoothed-python",
        entrypoint=kernel_name,
        tensors=specs,
        compiler_options={
            "num_warps": request.num_warps,
            "num_stages": request.num_stages,
            "max_num_configs": request.max_num_configs,
            "ssa_pass_pipeline": request.pipeline,
            "ssa_pass_options": dict(request.pass_options or {}),
            "backend_options": backend_options,
        },
        metadata={
            "caller": request.caller,
            "ssa_ir_source": "application_ast",
            "ssa_tensor_ir_source": "arrangement_views",
            "runtime_shape_params": _runtime_shape_params(specs),
            "generation_py_fallback": False,
            "meta_defaults": meta_defaults,
        },
        ssa=program,
    )
    artifact = emit(kernel, backend=target)
    scheduled_defaults = scheduled_meta_defaults(
        meta_defaults, artifact.metadata.get("ssa_schedule", {})
    )
    launch_abi = _launch_abi(
        params,
        specs,
        artifact,
        arranged,
        meta_defaults=scheduled_defaults,
    )
    launch_plan = _launch_plan(launch_abi, artifact, request, arranged)
    kernel = replace(
        kernel,
        launch_abi=launch_abi,
        launch_plan=launch_plan,
        metadata=dict(kernel.metadata) | {"meta_defaults": scheduled_defaults},
    )
    artifact = replace(
        artifact,
        metadata=dict(artifact.metadata)
        | {
            "launch_abi": _abi_dict(launch_abi),
            "launch_plan": _launch_plan_dict(launch_plan),
            "generation_py_fallback": False,
        },
    )

    return Compilation(
        request=request,
        kernel=kernel,
        artifact=artifact,
        launch_plan=launch_plan,
        pass_trace=tuple(artifact.metadata.get("ssa_pass_trace", ())),
    )


def _launch_abi(
    params,
    specs,
    artifact: Artifact,
    arranged,
    *,
    meta_defaults: Mapping[str, int] | None = None,
) -> LaunchABI:
    by_name = {spec.name: spec for spec in specs}
    auxiliary_bindings = {
        str(binding["name"]): binding
        for binding in artifact.metadata.get("auxiliary_bindings", ())
    }
    meta_defaults = dict(meta_defaults or _meta_defaults(arranged))
    constexpr_values = {
        name: getattr(getattr(tensor, "source", tensor), "value", None)
        for name, tensor in zip(params, arranged)
    }
    bindings = []

    for name in (
        *artifact.metadata.get("variables", ()),
        *artifact.metadata.get("outputs", ()),
    ):
        if name in auxiliary_bindings:
            auxiliary = auxiliary_bindings[name]
            bindings.append(
                LaunchBinding(
                    name=name,
                    kind=str(auxiliary["kind"]),
                    source=str(auxiliary["source"]),
                )
            )
            continue

        spec = by_name[name]
        bindings.append(
            LaunchBinding(
                name=name,
                kind=(
                    "constexpr"
                    if spec.constexpr
                    else "scalar"
                    if spec.ndim == 0
                    else "jagged_values"
                    if spec.jagged_dim is not None
                    else "tensor"
                ),
                source=name,
            )
        )

    shape_params = tuple(artifact.metadata.get("shape_params", ()))

    for name in shape_params:
        binding = _derived_binding(name, specs, constexpr_values)
        bindings.append(
            binding
            or LaunchBinding(
                name=name,
                kind="meta",
                source=remove_prefixes(name),
                value=meta_defaults.get(name),
            )
        )

    return LaunchABI(
        public_args=tuple(params),
        kernel_args=tuple(bindings),
        outputs=tuple(artifact.metadata.get("outputs", ())),
        shape_params=shape_params,
    )


def _launch_plan(
    abi: LaunchABI,
    artifact: Artifact,
    request: CompileRequest,
    arranged,
) -> LaunchPlan:
    metadata = artifact.metadata
    grid = tuple(
        IndexExpr.parse(str(value)) for value in metadata.get("launch_grid", ("1",))
    )
    block = tuple(
        IndexExpr.parse(str(value)) for value in metadata.get("launch_block", ())
    )
    dynamic = tuple(
        binding.name
        for binding in abi.kernel_args
        if binding.kind
        in {
            "shape",
            "stride",
            "meta",
            "jagged_values_numel",
            "jagged_offsets_numel",
            "jagged_max_seq_len",
        }
    )
    specialization = tuple(
        dict.fromkeys(
            (
                *dynamic,
                *tuple((request.specialization_values or {}).keys()),
            )
        )
    )
    candidates = (
        _triton_tuning_candidates(metadata, request, arranged)
        if artifact.backend == Target.TRITON
        else ()
    )

    return LaunchPlan(
        abi=abi,
        grid=grid,
        block=block,
        dynamic_parameters=dynamic,
        specialization_key=specialization,
        tuning_candidates=candidates,
    )


def _validate_tuning_options(target: Target, request: CompileRequest) -> None:
    if request.max_num_configs is not None and request.max_num_configs < 1:
        raise ValueError("The `max_num_configs` value must be at least one.")

    for name, value in (
        ("num_warps", request.num_warps),
        ("num_stages", request.num_stages),
    ):
        if value is None:
            continue

        values = value if isinstance(value, tuple) else (value,)

        if not values or any(int(item) < 1 for item in values):
            raise ValueError(f"The `{name}` value must contain positive integers.")

    if target == Target.TRITON:
        return

    if (
        isinstance(request.num_warps, tuple)
        or isinstance(request.num_stages, tuple)
        or request.max_num_configs not in {None, 1}
    ):
        raise NotImplementedError(
            f"Backend auto-tuning is not supported for `{target.value}` yet; "
            "use scalar num_warps/num_stages and max_num_configs=1."
        )


def _triton_tuning_candidates(
    metadata: Mapping[str, Any],
    request: CompileRequest,
    arranged,
) -> tuple[Mapping[str, Any], ...]:
    schedule = dict(metadata.get("ssa_schedule", {}))
    warps = _configuration_values(
        request.num_warps,
        schedule.get("num_warps"),
        default=4,
    )
    stages = _configuration_values(
        request.num_stages,
        schedule.get("num_stages"),
        default=3,
    )
    candidates = []

    for meta, num_warps, num_stages in itertools.product(
        _meta_parameter_configurations(arranged), warps, stages
    ):
        meta_id = "_".join(
            f"{remove_prefixes(name)}-{value}" for name, value in meta.items()
        )
        candidate = {
            "id": "_".join(
                value
                for value in (
                    meta_id,
                    f"warps-{num_warps}",
                    f"stages-{num_stages}",
                )
                if value
            ),
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

        if meta:
            candidate["meta_parameters"] = meta

        if candidate not in candidates:
            candidates.append(candidate)

    limit = request.max_num_configs

    if limit is not None and len(candidates) > limit:
        candidates = [
            candidates[index * len(candidates) // limit] for index in range(limit)
        ]
    return tuple(candidates)


def _meta_parameter_configurations(arranged) -> tuple[dict[str, int], ...]:
    symbols = {
        str(symbol): symbol
        for tensor in arranged
        for symbol in tensor.names()
        if is_meta(str(symbol))
    }

    if not symbols:
        return ({},)

    names = tuple(sorted(symbols))
    values = tuple(_symbol_values(symbols[name]) for name in names)
    configurations = tuple(
        configuration
        for combination in itertools.product(*values)
        if _meta_configuration_is_legal(
            arranged, configuration := dict(zip(names, combination))
        )
    )

    if not configurations:
        raise ValueError(
            "Failed to generate Triton tuning candidates. Check the lower and "
            "upper bounds of the block-size symbols."
        )
    return configurations


def _symbol_values(symbol) -> tuple[int, ...]:
    values = range(int(symbol.lower_bound), int(symbol.upper_bound) + 1)

    if getattr(symbol, "power_of_two", False):
        return tuple(
            value for value in values if value > 0 and value & (value - 1) == 0
        )
    return tuple(values)


def _meta_configuration_is_legal(arranged, configuration) -> bool:
    """Apply the old generator's conservative innermost-tile size bound."""
    import sympy

    max_num_elements = 2**15

    for tensor in arranged:
        expression = sympy.sympify(str(math.prod(tensor.innermost().shape)))
        specialized = expression.subs(configuration)

        if specialized.free_symbols:
            continue

        num_elements = int(specialized)

        if not 1 <= num_elements <= max_num_elements:
            return False
    return True


def _configuration_values(value, scheduled, *, default: int) -> tuple[int, ...]:
    selected = value if value is not None else scheduled

    if selected is None:
        selected = default

    values = selected if isinstance(selected, tuple) else (selected,)
    normalized = tuple(dict.fromkeys(int(item) for item in values))

    if not normalized or any(item < 1 for item in normalized):
        raise ValueError("Triton launch configurations must contain positive integers.")
    return normalized


def _derived_binding(name: str, specs, constexpr_values):
    for spec in specs:
        attrs = spec.attrs

        if name == attrs.get("jagged_values_numel_param"):
            return LaunchBinding(
                name=name,
                kind="jagged_values_numel",
                source=spec.name,
            )

        if name == attrs.get("jagged_offsets_numel_param"):
            return LaunchBinding(
                name=name,
                kind="jagged_offsets_numel",
                source=spec.name,
            )

        if name == attrs.get("jagged_max_seq_len_param"):
            return LaunchBinding(
                name=name,
                kind="jagged_max_seq_len",
                source=spec.name,
            )

        for dim, symbol in enumerate(attrs.get("source_shape", ())):
            if name == symbol:
                return LaunchBinding(name=name, kind="shape", source=spec.name, dim=dim)

        for dim, symbol in enumerate(attrs.get("source_strides", ())):
            if name == symbol:
                return LaunchBinding(
                    name=name, kind="stride", source=spec.name, dim=dim
                )

        if spec.constexpr and name == spec.name:
            return LaunchBinding(
                name=name,
                kind="constexpr",
                source=spec.name,
                value=constexpr_values.get(spec.name),
            )
    return None


def _meta_defaults(arranged) -> dict[str, int]:
    defaults = {}

    for tensor in arranged:
        for symbol in tensor.names():
            name = str(symbol)

            if not hasattr(symbol, "lower_bound"):
                continue

            lower = int(symbol.lower_bound)
            upper = int(symbol.upper_bound)
            value = min(max(256, lower), upper)

            if getattr(symbol, "power_of_two", False):
                value = 1 << max(0, value.bit_length() - 1)

            defaults[name] = value
    return defaults


def _runtime_shape_params(specs) -> tuple[str, ...]:
    params = []

    for spec in specs:
        attrs = spec.attrs
        values = (
            *attrs.get("source_shape", ()),
            *attrs.get("source_strides", ()),
            attrs.get("jagged_values_numel_param"),
            attrs.get("jagged_offsets_numel_param"),
            attrs.get("jagged_max_seq_len_param"),
        )

        for value in values:
            name = str(value or "")

            if (
                name.isidentifier()
                and "constexpr_prefix" not in name
                and name not in params
            ):
                params.append(name)
    return tuple(params)


def _abi_dict(abi: LaunchABI) -> dict[str, Any]:
    return {
        "public_args": abi.public_args,
        "kernel_args": tuple(binding.__dict__ for binding in abi.kernel_args),
        "outputs": abi.outputs,
        "shape_params": abi.shape_params,
    }


def _launch_plan_dict(plan: LaunchPlan) -> dict[str, Any]:
    return {
        "abi": _abi_dict(plan.abi),
        "grid": tuple(value.render() for value in plan.grid),
        "block": tuple(value.render() for value in plan.block),
        "dynamic_parameters": plan.dynamic_parameters,
        "specialization_key": plan.specialization_key,
        "tuning_candidates": plan.tuning_candidates,
    }


def _source(application) -> str:
    try:
        return textwrap.dedent(inspect.getsource(application))
    except OSError:
        return f"def {application.__name__}(...):\n    pass\n"


__all__ = [
    "Compiler",
    "Compilation",
    "CompileRequest",
    "DEFAULT_COMPILER",
    "aot",
    "compile_kernel",
    "lower",
    "make",
    "resolve_target",
]
