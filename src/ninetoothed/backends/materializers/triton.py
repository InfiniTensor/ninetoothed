"""Triton artifact materialization."""

import ctypes
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, replace
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.emitters.expressions import replace_symbols
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import find_nvcc
from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    artifact_directory,
    cache_lock,
    compilation_cache_key,
    read_manifest,
    write_manifest,
    write_source,
)
from ninetoothed.compiler.layout_runtime import (
    build_layout_transfer_validator,
    memory_spans_overlap,
    tensor_memory_span,
)

_TRITON_AOT_LAUNCHER_SCHEMA = 1
_TRITON_MODULE_PATTERN = re.compile(
    r"^CUmodule ([A-Za-z_][A-Za-z0-9_]*)_mod = NULL;$", re.MULTILINE
)
_TRITON_FUNCTION_PATTERN = re.compile(
    r"^CUfunction ([A-Za-z_][A-Za-z0-9_]*)_func = NULL;$", re.MULTILINE
)
_TRITON_LOADER_PATTERN = re.compile(
    r"^void load_([A-Za-z_][A-Za-z0-9_]*)\(\) \{$", re.MULTILINE
)


def _active_triton_target():
    try:
        from triton.runtime import driver

        return driver.active.get_current_target()
    except (AttributeError, ImportError):
        return None
    except (KeyError, RuntimeError, TypeError, UnicodeDecodeError):
        return _torch_hip_target(driver.active)


def _torch_hip_target(active_driver):
    """Recover target discovery from vendor HIP runtime ABI mismatches."""
    try:
        import torch
        from triton.backends.compiler import GPUTarget

        try:
            from triton import knobs

            override_arch = knobs.runtime.override_arch
        except (AttributeError, ImportError):
            # Triton 3.1 vendor builds predate ``triton.knobs``.  Those builds
            # still expose GPUTarget and report the HIP architecture through
            # PyTorch, so target recovery does not require the newer knob API.
            override_arch = None

        if torch.version.hip is None or not torch.cuda.is_available():
            return None

        device = active_driver.get_current_device()
        properties = torch.cuda.get_device_properties(device)
        architecture = str(
            override_arch or properties.gcnArchName
        ).split(":", 1)[0]
        warp_size = int(properties.warp_size)

        if not architecture.startswith("gfx") or warp_size <= 0:
            return None

        target = GPUTarget("hip", architecture, warp_size)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return None

    def get_current_target(_driver):
        return target

    # Triton calls target discovery again during JIT compilation.  Cache the
    # validated PyTorch result on the active driver so those calls use the same
    # target instead of re-entering the incompatible vendor HIP helper.
    active_driver.get_current_target = types.MethodType(
        get_current_target,
        active_driver,
    )

    return target


def _legalize_triton_num_stages(num_stages):
    stages = int(num_stages)
    target = _active_triton_target()

    if target is None:
        return stages

    backend = str(getattr(target, "backend", "")).lower()
    warp_size = getattr(target, "warp_size", None)

    # CoreX presents itself through Triton's CUDA target but uses a 64-thread
    # warp and currently supports one software-pipeline stage.  A capability
    # check keeps HIP wave64 and conventional CUDA warp32 targets unchanged.
    if backend == "cuda" and warp_size == 64:
        return 1

    return stages


def _legalized_tuning_candidates(compilation):
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    legalized = []
    configurations = set()

    for candidate in candidates:
        normalized = dict(candidate)
        requested_stages = int(normalized.get("num_stages", 3))
        stages = _legalize_triton_num_stages(requested_stages)
        normalized["num_stages"] = stages

        if stages != requested_stages:
            normalized["id"] = f"{normalized.get('id', 'candidate')}_stages-{stages}"

        configuration = (
            int(normalized.get("num_warps", 4)),
            stages,
            tuple(sorted(dict(normalized.get("meta_parameters", {})).items())),
            tuple(
                sorted(dict(normalized.get("private_meta_parameters", {})).items())
            ),
        )

        if configuration in configurations:
            continue

        configurations.add(configuration)
        legalized.append(normalized)

    return tuple(legalized)


class TritonMaterializer(Materializer):
    target = Target.TRITON

    def jit_materialize(self, compilation, *, output_dir=None):
        del output_dir

        return _materialize(compilation)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _aot_materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("Triton built artifact does not contain an AOT binary.")

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
        )

        _, function, enter, leave = _load_aot_exports(
            built.binary_path,
            built.source.kernel_name,
        )
        specs = _runtime_specs(built.source)

        abi = _launch_abi_from_dict(built.abi)

        return _aot_wrapper(
            function,
            enter,
            leave,
            abi,
            specs,
            validate_bindings=build_layout_transfer_validator(
                built.source.metadata,
                abi,
            ),
        )


def _ensure_c_compiler() -> None:
    if os.environ.get("CC"):
        return

    candidates = (
        shutil.which("cc"),
        shutil.which("gcc"),
        "/usr/bin/cc",
        "/usr/bin/gcc",
    )

    for candidate in candidates:
        if candidate is not None and Path(candidate).is_file():
            os.environ["CC"] = candidate

            return


def _materialize(compilation):
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        _verified_runtime_launch,
        import_python_module,
    )

    _ensure_c_compiler()
    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "triton.py",
        cache_key=cache_key,
    )
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))
    module = import_python_module(source)
    launch = getattr(module, artifact.entrypoint)
    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    candidates = _legalized_tuning_candidates(compilation)
    validate_bindings = _runtime_binding_validator(compilation)
    candidate_launches = tuple(
        _candidate_launch(compilation, launch, kernel, candidate, validate_bindings)
        for candidate in candidates
    )
    tuner = None

    if len(candidate_launches) > 1:
        from ninetoothed.auto_tuner import AutoTuner

        class TritonAutoTuner(AutoTuner):
            @staticmethod
            def _make_arg_key(args, kwargs):
                return _triton_specialization_key(compilation, args, kwargs)

        tuner = TritonAutoTuner(
            candidate_launches,
            tuple((cache_key, candidate["id"]) for candidate in candidates),
            cache_namespace=f"jit_{cache_key}",
            validator=_runtime_validator(compilation, validate_bindings),
        )
        wrapped = tuner
    elif candidate_launches:
        wrapped = _verified_runtime_launch(candidate_launches[0])
    else:
        _, num_stages = _compile_schedule(compilation)
        launch = functools.partial(
            launch,
            _ninetoothed_num_stages=num_stages,
        )
        wrapped = _verified_runtime_launch(
            _runtime_wrapper(
                launch,
                compilation.launch_abi,
                specs=compilation.kernel.tensors,
                prepare_invocation=_triton_prepare_invocation(launch, kernel),
                validate_bindings=validate_bindings,
            )
        )

    handle = Handle(compilation, kernel, wrapped, source)
    handle._tuner = tuner
    handle._selected_tuning_candidate = candidates[0] if len(candidates) == 1 else None
    by_launch = dict(zip(candidate_launches, candidates))

    if tuner is not None:
        handle._launch = _tuned_runtime_launch(
            tuner,
            by_launch,
            handle,
            compilation,
            validate_bindings,
        )

    by_launch = dict(zip(candidate_launches, candidates))
    handle._launch_prevalidated_noalias = _prevalidated_noalias_runtime_launch(
        handle,
        tuner,
        compilation,
        candidates_by_launch=by_launch,
    )

    return handle


def _prevalidated_noalias_runtime_launch(
    handle, tuner, compilation, *, candidates_by_launch=None
):
    """Build a low-overhead launcher for operator wrappers with static contracts.

    The ordinary public handle remains fully checked.  This private launcher is
    for wrappers which already validate tensor shape, stride, dtype, device and
    output non-aliasing before every call.  It retains the normal first launch
    (including auto-tuning), then reuses the resulting direct invocation plan
    for later tensors with the same runtime scalar values.
    """
    from ninetoothed.compiler.runtime import (
        _first_output_from_call,
        _is_cacheable_runtime_literal,
    )

    candidates_by_launch = dict(candidates_by_launch or {})
    scalar_sources = tuple(
        dict.fromkeys(
            binding.source
            for binding in compilation.launch_abi.kernel_args
            if binding.kind in {"scalar", "constexpr", "meta"}
            and binding.source in compilation.launch_abi.public_args
        )
    )
    plans = {}

    def selected_candidate(args, kwargs):
        """Return the launch selected by the verified public path.

        Alias-safe calls intentionally bypass ``AutoTuner._best_func``: the
        public path records the safe candidate on the handle instead.  The
        private wrapper must honor that decision when arming its direct plan,
        otherwise every in-place call pays the full runtime preparation cost.
        """
        selected = None if tuner is None else tuner._best_func.get(
            tuner._make_arg_key(args, kwargs)
        )
        if selected is not None:
            return selected

        selected_id = getattr(handle, "_selected_tuning_candidate", None)
        if isinstance(selected_id, dict):
            selected_id = selected_id.get("id")

        if selected_id is None:
            return None

        return next(
            (
                function
                for function, candidate in candidates_by_launch.items()
                if candidate.get("id") == selected_id
            ),
            None,
        )

    def plan_key(args, kwargs):
        public = dict(zip(compilation.launch_abi.public_args, args))
        public.update(kwargs)
        values = []

        for source in scalar_sources:
            value = public.get(source)

            # Scalar tensors may change their device value without changing
            # object metadata; keep them on the fully verified path.
            if getattr(value, "shape", None) is not None:
                return None

            if not _is_cacheable_runtime_literal(value):
                return None

            values.append((source, type(value), value))

        key = tuple(values)

        try:
            hash(key)
        except TypeError:
            return None
        return key

    def launch(*args, **kwargs):
        key = plan_key(args, kwargs)
        invocation = plans.get(key) if key is not None else None

        if invocation is not None:
            invocation((), args, kwargs)
            return _first_output_from_call(compilation.launch_abi, args, kwargs)

        result = handle._launch(*args, **kwargs)

        if key is None:
            return result

        selected = selected_candidate(args, kwargs)

        if selected is None:
            return result

        try:
            prepared = selected._ninetoothed_prepare(args, kwargs)
        except Exception:  # noqa: BLE001
            return result

        invocation = prepared.invocation_plan

        if invocation is not None and not getattr(
            invocation, "requires_values", True
        ):
            plans[key] = invocation

        return result

    def bind(*args, **kwargs):
        """Bind an already-armed direct plan to one validated argument set.

        Operator wrappers may use the returned zero-argument callable only
        while keeping these exact tensor objects alive and revalidating the
        public contract before rebinding.  Returning the raw direct call keeps
        repeated in-place operator launches off the Python ABI preparation
        path without weakening the checked public handle.
        """
        key = plan_key(args, kwargs)
        invocation = plans.get(key) if key is not None else None

        if invocation is None or getattr(invocation, "requires_values", True):
            selected = selected_candidate(args, kwargs)

            if selected is None:
                return None

            try:
                prepared = selected._ninetoothed_prepare(args, kwargs)
                invocation = prepared.invocation_plan
                resolved = prepared.binding_plan.resolve(flatten_tensors=True)
            except (AttributeError, RuntimeError, TypeError):
                return None

            if invocation is None or resolved is None:
                return None

            values, keepalive = resolved
            bind_invocation = getattr(invocation, "bind", None)
            direct = (
                bind_invocation(values, args, kwargs)
                if bind_invocation is not None
                else None
            )

            if direct is None:
                return None

            def bound():
                result = direct()
                _ = keepalive, prepared
                return result

            bound._ninetoothed_direct_compiled_launch = True
            return bound

        direct_call = getattr(invocation, "call", None)

        if direct_call is not None:
            return functools.partial(direct_call, args, kwargs)

        return functools.partial(invocation, (), args, kwargs)

    launch._ninetoothed_bind_prevalidated_noalias = bind

    return launch


def _candidate_launch(compilation, launch, kernel, candidate, validate_bindings):
    from ninetoothed.compiler.runtime import _runtime_wrapper

    bindings = {binding.name: binding for binding in compilation.launch_abi.kernel_args}
    meta_values = {
        str(bindings[name].source): value
        for name, value in dict(candidate.get("meta_parameters", {})).items()
    }
    private_meta = {
        f"_ninetoothed_{name.lower()}": value
        for name, value in dict(candidate.get("private_meta_parameters", {})).items()
    }
    function = functools.partial(
        launch,
        **private_meta,
        _ninetoothed_num_warps=int(candidate["num_warps"]),
        _ninetoothed_num_stages=int(candidate["num_stages"]),
    )

    return _runtime_wrapper(
        function,
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
        binding_overrides=meta_values,
        prepare_invocation=_triton_prepare_invocation(function, kernel),
        validate_bindings=validate_bindings,
    )


def _bind_compiled_triton_kernel(kernel, grid, args, kwargs):
    """Bind one compiled Triton specialization to stable tensor objects.

    ``JITFunction.run`` repeats argument binding, specialization-key creation,
    backend option parsing, and cache lookup on every invocation.  Operator
    wrappers which have already validated and retained the exact tensor
    objects can safely perform those steps once.  The returned callable still
    resolves the active stream for every launch, so normal stream semantics
    are preserved.

    Vendor Triton forks do not all expose the same low-level launcher.  This
    helper therefore uses feature detection and returns ``None`` whenever the
    installed runtime cannot provide the required stable ABI.
    """
    try:
        from triton.runtime import driver

        call_kwargs = dict(kwargs)
        compiled = kernel.run(
            *args,
            grid=grid,
            warmup=True,
            **call_kwargs,
        )
        device = driver.active.get_current_device()

        if hasattr(kernel, "binder"):
            if kernel.binder is None:
                kernel.create_binder()

            binder = kernel.binder
        else:
            # Triton 3.1 vendor forks keep the specialization cache and binder
            # together in a per-device tuple.
            _cache, _target, _backend, binder = kernel.device_caches[device]

        bound_args, _signature, _constexpr, runtime_args, _excess = binder(
            *args,
            **call_kwargs,
        )
        launch_grid = grid(bound_args) if callable(grid) else grid
        launch_grid = tuple(launch_grid)

        if not launch_grid or len(launch_grid) > 3:
            return None

        grid_x, grid_y, grid_z = (*launch_grid, 1, 1)[:3]
        compiled_kernel_type = getattr(kernel, "CompiledKernel", type(compiled))

        required = (
            "function",
            "launch_metadata",
            "packed_metadata",
            "run",
        )

        if compiled is None or any(
            not hasattr(compiled, attribute) for attribute in required
        ):
            return None

        def invoke():
            stream = driver.active.get_current_stream(device)
            launch_metadata = compiled.launch_metadata(
                launch_grid,
                stream,
                *runtime_args,
            )

            return compiled.run(
                grid_x,
                grid_y,
                grid_z,
                stream,
                compiled.function,
                compiled.packed_metadata,
                launch_metadata,
                compiled_kernel_type.launch_enter_hook,
                compiled_kernel_type.launch_exit_hook,
                *runtime_args,
            )

        return invoke
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError):
        return None


def _triton_prepare_invocation(function, kernel):
    from ninetoothed.compiler.runtime import _is_cacheable_runtime_literal

    if kernel is None:
        return None

    target = getattr(function, "func", function)

    if not isinstance(target, types.FunctionType):
        return None

    kernel_names = tuple(
        name
        for name in target.__code__.co_names
        if target.__globals__.get(name) is kernel
    )

    if len(kernel_names) != 1:
        return None

    kernel_name = kernel_names[0]
    partial_args = getattr(function, "args", ())
    partial_keywords = getattr(function, "keywords", None) or {}

    @dataclass(frozen=True)
    class ValueRef:
        index: int

        def resolve(self, values, _args, _kwargs):
            return values[self.index]

    @dataclass(frozen=True)
    class CallRef:
        kind: str
        key: object

        def resolve(self, _values, args, kwargs):
            return args[self.key] if self.kind == "positional" else kwargs[self.key]

    @dataclass(frozen=True)
    class Literal:
        value: object

        def resolve(self, _values, _args, _kwargs):
            return self.value

    @dataclass(frozen=True)
    class BoundCall:
        function: object
        args: tuple
        kwargs: tuple
        grid: object

        def invoke(self, values, args, kwargs):
            self.function(
                *(value.resolve(values, args, kwargs) for value in self.args),
                **{
                    name: value.resolve(values, args, kwargs)
                    for name, value in self.kwargs
                },
            )

    @dataclass(frozen=True)
    class InvocationPlan:
        call: object
        requires_values: bool
        bound_call: object

        def __call__(self, values, args, kwargs):
            if self.requires_values:
                self.call.invoke(values, args, kwargs)
            else:
                self.call(args, kwargs)

        def bind(self, values, args, kwargs):
            call = self.bound_call
            resolved_args = tuple(
                value.resolve(values, args, kwargs) for value in call.args
            )
            resolved_kwargs = {
                name: value.resolve(values, args, kwargs) for name, value in call.kwargs
            }

            return _bind_compiled_triton_kernel(
                kernel,
                call.grid,
                resolved_args,
                resolved_kwargs,
            )

    def plan_value(value, values, static_values, call_sources):
        for index, candidate in enumerate(values):
            if value is candidate:
                if call_sources is not None and call_sources[index] is not None:
                    return CallRef(*call_sources[index])

                return (
                    Literal(static_values[index])
                    if static_values is not None
                    else ValueRef(index)
                )

        if hasattr(value, "data_ptr") or (
            hasattr(value, "shape") and hasattr(value, "dtype")
        ):
            return None

        if _is_cacheable_runtime_literal(value):
            return Literal(value)

        return None

    def build_direct_invocation(function, args, kwargs):
        namespace = {"_function": function}
        expressions = []

        def expression(value):
            if isinstance(value, CallRef):
                container = "_args" if value.kind == "positional" else "_kwargs"

                return f"{container}[{value.key!r}]"

            name = f"_value_{len(namespace)}"
            namespace[name] = value.value

            return name

        expressions.extend(expression(value) for value in args)
        expressions.extend(f"{name}={expression(value)}" for name, value in kwargs)
        source = (
            "def invoke(_args, _kwargs):\n    _function(" + ", ".join(expressions) + ")"
        )
        exec(compile(source, "<ninetoothed-triton-invocation>", "exec"), namespace)

        return namespace["invoke"]

    def prepare(values, static_values=None, call_sources=None):
        calls = []

        class KernelProxy:
            def __getitem__(self, grid):
                bound_kernel = kernel[grid]

                def capture(*args, **kwargs):
                    calls.append((bound_kernel, grid, args, kwargs))

                return capture

        globals_copy = dict(target.__globals__)
        globals_copy[kernel_name] = KernelProxy()
        dry_launch = types.FunctionType(
            target.__code__,
            globals_copy,
            target.__name__,
            target.__defaults__,
            target.__closure__,
        )
        dry_launch.__kwdefaults__ = target.__kwdefaults__
        functools.partial(
            dry_launch,
            *partial_args,
            **partial_keywords,
        )(*values)

        if len(calls) != 1:
            return None

        bound_kernel, grid, args, kwargs = calls[0]
        planned_args = tuple(
            plan_value(value, values, static_values, call_sources) for value in args
        )
        planned_kwargs = tuple(
            (
                name,
                plan_value(value, values, static_values, call_sources),
            )
            for name, value in kwargs.items()
        )

        if any(value is None for value in planned_args) or any(
            value is None for _name, value in planned_kwargs
        ):
            return None

        planned_call = BoundCall(
            bound_kernel,
            planned_args,
            planned_kwargs,
            grid,
        )

        requires_values = any(
            isinstance(value, ValueRef) for value in planned_args
        ) or any(isinstance(value, ValueRef) for _name, value in planned_kwargs)

        if requires_values:
            return InvocationPlan(planned_call, True, planned_call)

        direct_call = build_direct_invocation(
            planned_call.function,
            planned_call.args,
            planned_call.kwargs,
        )

        return InvocationPlan(direct_call, False, planned_call)

    return prepare


def _runtime_alias_signature(abi, public):
    from ninetoothed.compiler.runtime import _binding_value

    output_names = set(abi.outputs)
    physical_tensors = {}
    aliases = []

    for binding in abi.kernel_args:
        if binding.source not in public or binding.kind not in {
            "tensor",
            "scalar",
            "constexpr",
            "jagged_values",
            "jagged_offsets",
        }:
            continue

        value = _binding_value(binding, public)

        if binding.kind in {"scalar", "constexpr"} and not all(
            hasattr(value, attribute) for attribute in ("data_ptr", "shape", "stride")
        ):
            continue

        identity = (binding.source, binding.kind)
        access = binding.access or (
            "read"
            if binding.kind in {"scalar", "constexpr", "jagged_offsets"}
            else "read_write"
            if binding.source in output_names
            else "read"
        )
        previous = physical_tensors.get(identity)

        if previous is not None and previous[2] != access:
            access = "read_write"

        physical_tensors[identity] = (
            *identity,
            access,
            value,
            tensor_memory_span(value),
        )

    writers = tuple(
        tensor
        for tensor in physical_tensors.values()
        if tensor[2] in {"write", "read_write"}
    )
    readers = tuple(
        tensor
        for tensor in physical_tensors.values()
        if tensor[2] in {"read", "read_write"}
    )

    for writer_name, writer_kind, _access, writer, writer_span in writers:
        for reader_name, reader_kind, _access, reader, reader_span in readers:
            overlaps = writer is reader

            if not overlaps:
                overlaps = memory_spans_overlap(writer_span, reader_span)

            if overlaps:
                aliases.append((writer_name, writer_kind, reader_name, reader_kind))

    return tuple(aliases)


def _runtime_rebind_signature(args, kwargs):
    def is_literal(value):
        return (
            value is None
            or type(value) in (bool, int, float, complex, str, bytes)
            or (isinstance(value, tuple) and all(is_literal(item) for item in value))
        )

    def value_signature(value):
        shape = getattr(value, "shape", None)

        if shape is not None and hasattr(value, "dtype"):
            stride = getattr(value, "stride", None)
            storage_offset = getattr(value, "storage_offset", None)

            try:
                return (
                    "tensor",
                    type(value),
                    tuple(shape),
                    tuple(stride()) if callable(stride) else None,
                    str(value.dtype),
                    str(getattr(value, "device", None)),
                    storage_offset() if callable(storage_offset) else None,
                )
            except (AttributeError, RuntimeError, TypeError):
                return None

        if not is_literal(value):
            return None

        return ("literal", type(value), value)

    positional = tuple(value_signature(value) for value in args)
    keywords = tuple((name, value_signature(value)) for name, value in kwargs.items())

    if any(value is None for value in positional) or any(
        value is None for _name, value in keywords
    ):
        return None

    signature = positional, keywords

    try:
        hash(signature)
    except TypeError:
        return None

    return signature


def _tuned_runtime_launch(
    tuner,
    candidates_by_launch,
    handle,
    compilation,
    validate_bindings=None,
):
    from ninetoothed.compiler.runtime import (
        _arm_prepared_runtime_launch,
        _empty_launch,
        _first_output,
        _public_values,
        _remember_verified_runtime_call,
        _runtime_call_identity,
    )

    active_identity = None
    active = None
    prepared_calls = {}
    rebindable_calls = {}

    def evict(identity, token):
        nonlocal active, active_identity
        cached = prepared_calls.get(identity)

        if cached is not None and cached[2].cache_token is token:
            prepared_calls.pop(identity, None)

        if active is not None and active[2].cache_token is token:
            active = None
            active_identity = None

    def activate(identity, entry):
        nonlocal active, active_identity
        active_identity = identity
        active = entry
        record(entry[1])

    def record(selected):
        handle._selected_tuning_candidate = candidates_by_launch[selected]

    def remember(identity, selection_key, selected, prepared):
        token = object()

        def collected(_reference):
            evict(identity, token)

        prepared = _arm_prepared_runtime_launch(prepared, collected, token)

        if prepared is None:
            return None

        entry = (selection_key, selected, prepared)
        _remember_verified_runtime_call(prepared_calls, identity, entry)
        activate(identity, entry)

        return prepared

    def remember_best_effort(identity, selection_key, selected, prepared):
        try:
            return remember(identity, selection_key, selected, prepared)
        except TypeError:
            return None

    def remember_rebindable(rebind_key, selected, prepared):
        invocation = prepared.invocation_plan

        if (
            rebind_key is None
            or prepared.empty
            or invocation is None
            or getattr(invocation, "requires_values", True)
        ):
            return

        template = replace(
            prepared,
            binding_plan=None,
            owner_refs=(),
            cache_token=None,
        )
        _remember_verified_runtime_call(
            rebindable_calls,
            rebind_key,
            (selected, template),
        )

    def launch(*args, **kwargs):
        active_snapshot = active
        active_identity_snapshot = active_identity
        identity = _runtime_call_identity(args, kwargs)

        if (
            active_snapshot is not None
            and identity == active_identity_snapshot
            and active_snapshot[2].matches(
                args,
                kwargs,
                identity_verified=True,
            )
        ):
            return active_snapshot[1]._ninetoothed_invoke_prepared(
                active_snapshot[2],
                args,
                kwargs,
            )

        cached = prepared_calls.pop(identity, None)

        if cached is not None and cached[2].matches(
            args,
            kwargs,
            identity_verified=True,
        ):
            prepared_calls[identity] = cached
            activate(identity, cached)

            return cached[1]._ninetoothed_invoke_prepared(cached[2], args, kwargs)

        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
        )

        if _empty_launch(compilation.launch_abi, public):
            if validate_bindings is not None:
                validate_bindings(public)

            return _first_output(compilation.launch_abi, public)

        key = tuner._make_arg_key(args, kwargs)
        alias_signature = _runtime_alias_signature(compilation.launch_abi, public)
        selection_key = (key, alias_signature)
        rebind_signature = _runtime_rebind_signature(args, kwargs)
        rebind_key = (
            (selection_key, rebind_signature)
            if rebind_signature is not None
            else None
        )
        rebindable = (
            rebindable_calls.pop(rebind_key, None)
            if rebind_key is not None
            else None
        )

        if rebindable is not None:
            if validate_bindings is not None:
                validate_bindings(public)

            rebindable_calls[rebind_key] = rebindable
            selected, prepared = rebindable
            result = selected._ninetoothed_invoke_prepared(prepared, args, kwargs)
            record(selected)

            return result

        selected = next(
            (
                entry[1]
                for entry in reversed(tuple(prepared_calls.values()))
                if entry[0] == selection_key
            ),
            None,
        )

        if selected is None and cached is not None and cached[0] == selection_key:
            selected = cached[1]

        if selected is None and alias_signature:
            selected = tuner._funcs[0]

            try:
                prepared = selected._ninetoothed_prepare(
                    args,
                    kwargs,
                    public=public,
                )
            except Exception:  # noqa: BLE001
                result = selected(*args, **kwargs)
            else:
                result = selected._ninetoothed_invoke_prepared(
                    prepared,
                    args,
                    kwargs,
                )
                remember_rebindable(rebind_key, selected, prepared)
                remember_best_effort(
                    identity,
                    selection_key,
                    selected,
                    prepared,
                )

            record(selected)

            return result

        if selected is None:
            result = tuner(*args, **kwargs)
            selected = tuner._best_func[key]
            record(selected)

            try:
                prepared = selected._ninetoothed_prepare(args, kwargs, public=public)
            except Exception:  # noqa: BLE001
                return result

            remember_rebindable(rebind_key, selected, prepared)
            remember_best_effort(identity, selection_key, selected, prepared)

            return result

        try:
            prepared = selected._ninetoothed_prepare(args, kwargs, public=public)
        except Exception:  # noqa: BLE001
            result = selected(*args, **kwargs)
            record(selected)

            return result

        result = selected._ninetoothed_invoke_prepared(
            prepared,
            args,
            kwargs,
        )
        record(selected)
        remember_rebindable(rebind_key, selected, prepared)
        remember_best_effort(identity, selection_key, selected, prepared)

        return result

    return launch


def _triton_specialization_key(compilation, args, kwargs):
    from ninetoothed.auto_tuner import AutoTuner

    public = dict(zip(compilation.launch_abi.public_args, args))
    public.update(kwargs)
    scalar_values = []

    for binding in compilation.launch_abi.kernel_args:
        if (
            binding.kind not in {"scalar", "constexpr", "meta"}
            or binding.source not in public
        ):
            continue

        value = public[binding.source]
        shape = getattr(value, "shape", None)

        if shape is not None and not tuple(shape) and hasattr(value, "item"):
            value = value.item()

        scalar_values.append((binding.source, type(value).__name__, repr(value)))

    base = AutoTuner._make_arg_key(args, kwargs)

    return f"{base}, specialization={tuple(scalar_values)!r}"


def _runtime_validator(compilation, validate_bindings):
    from ninetoothed.compiler.runtime import _public_values

    def validate(args, kwargs):
        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
        )

        if validate_bindings is not None:
            validate_bindings(public)

    return validate


def _runtime_binding_validator(compilation):
    validators = tuple(
        validator
        for validator in (
            _runtime_vector_validator(compilation),
            build_layout_transfer_validator(
                getattr(compilation.artifact, "metadata", {}),
                compilation.launch_abi,
            ),
        )
        if validator is not None
    )

    if not validators:
        return None

    def validate(public):
        for validator in validators:
            validator(public)

    return validate


def _runtime_vector_validator(compilation):
    metadata = getattr(compilation.artifact, "metadata", {})
    limit = metadata.get("vector_numel_limit")
    schedule = metadata.get("ssa_schedule", {})
    reduction = schedule.get("reduction", {})

    if reduction.get("mode") != "row-vector":
        return None

    import sympy

    extent_expression = sympy.sympify(str(reduction["extent"]))
    program_constraints = tuple(
        (
            tuple(sympy.sympify(str(value)) for value in actual),
            tuple(sympy.sympify(str(value)) for value in expected),
        )
        for actual, expected in reduction.get("program_constraints", ())
    )
    expressions = (
        extent_expression,
        *(
            expression
            for constraint in program_constraints
            for shape in constraint
            for expression in shape
        ),
    )
    bindings = {binding.name: binding for binding in compilation.launch_abi.kernel_args}

    try:
        symbols = tuple(
            (symbol, bindings[str(symbol)])
            for symbol in sorted(
                set().union(*(expression.free_symbols for expression in expressions)),
                key=str,
            )
        )
    except KeyError as exc:
        raise ValueError(
            f"Triton reduction extent references unknown symbol `{exc.args[0]}`."
        ) from exc

    from ninetoothed.compiler.runtime import _binding_value

    last_values = None

    def validate(public):
        nonlocal last_values
        values = tuple(_binding_value(binding, public) for _symbol, binding in symbols)

        if values == last_values:
            return

        substitutions = {
            symbol: value for (symbol, _binding), value in zip(symbols, values)
        }

        def resolve(expression):
            resolved = expression.subs(substitutions)

            if resolved.free_symbols:
                names = ", ".join(sorted(map(str, resolved.free_symbols)))
                raise ValueError(f"Unresolved Triton row-vector symbols: {names}.")
            return int(resolved)

        extent = resolve(extent_expression)

        if limit is not None and extent > limit:
            block = 1 << (extent - 1).bit_length()
            raise ValueError(
                f"Triton row-vector reduction extent {extent} requires "
                f"BLOCK={block}, exceeding the backend tensor numel limit "
                f"{limit}; hierarchical reduction is not implemented."
            )

        for actual, expected in program_constraints:
            actual_total = 1
            expected_total = 1

            for expression in actual:
                actual_total *= resolve(expression)

            for expression in expected:
                expected_total *= resolve(expression)

            if actual_total != expected_total:
                raise ValueError(
                    "Triton row-vector reduction operand and output program "
                    "domains do not match; separate kernels are required."
                )

        last_values = values

    return validate


def _aot_materialize(compilation, *, output_dir):
    from ninetoothed.compiler.runtime import Handle, _publish_library

    compilation = _select_aot_candidate(compilation)
    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "triton.py",
        cache_key=cache_key,
    )
    cache_library = artifact_directory(cache_key) / f"{artifact.kernel_name}.triton.so"

    with cache_lock(cache_library):
        _ensure_aot_library(compilation, source, cache_library)

        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _triton_aot_manifest(compilation, cache_key, source, cache_library),
        )

    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.triton.so",
    )
    _, function, enter, leave = _load_aot_exports(
        library_path,
        artifact.kernel_name,
    )
    wrapped = _aot_wrapper(
        function,
        enter,
        leave,
        compilation.launch_abi,
        compilation.kernel.tensors,
        validate_bindings=build_layout_transfer_validator(
            getattr(artifact, "metadata", {}),
            compilation.launch_abi,
        ),
    )
    handle = Handle(compilation, function, wrapped, source, library_path)
    write_manifest(
        handle._built_artifact.manifest_path,
        _triton_aot_manifest(compilation, cache_key, source, library_path),
    )

    return handle


def _select_aot_candidate(compilation):
    launch_plan = getattr(compilation, "launch_plan", None)

    if launch_plan is None:
        return compilation

    candidates = _legalized_tuning_candidates(compilation)

    if candidates != tuple(launch_plan.tuning_candidates):
        compilation = replace(
            compilation,
            launch_plan=replace(
                launch_plan,
                tuning_candidates=candidates,
            ),
        )
        launch_plan = compilation.launch_plan

    if len(candidates) <= 1:
        return compilation

    metadata = getattr(compilation.artifact, "metadata", {})

    if not metadata.get("layout_transfer"):
        raise ValueError(
            "Triton AOT accepts one launch configuration; use build() to "
            "benchmark and package multiple explicit configurations."
        )

    return replace(
        compilation,
        launch_plan=replace(
            compilation.launch_plan,
            tuning_candidates=(candidates[0],),
        ),
    )


def _ensure_aot_library(compilation, source: Path, library: Path) -> None:
    manifest = read_manifest(library.with_suffix(".manifest.json"))
    schema = None if manifest is None else manifest.get("triton_aot_launcher_schema")

    if not library.is_file() or schema != _TRITON_AOT_LAUNCHER_SCHEMA:
        _compile_aot_library(compilation, source, library)


def _triton_aot_manifest(compilation, cache_key, source, library):
    from ninetoothed.compiler.runtime import _built_manifest

    return dict(_built_manifest(compilation, cache_key, source, library)) | {
        "triton_aot_launcher_schema": _TRITON_AOT_LAUNCHER_SCHEMA,
    }


def _load_aot_exports(library_path, kernel_name):
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, f"{kernel_name}_kernel_default")

    try:
        enter = library.ninetoothed_triton_enter
        leave = library.ninetoothed_triton_leave
    except AttributeError as exc:
        raise RuntimeError(
            "Triton AOT artifact is missing the CUDA context guard exports; "
            "rebuild the artifact with the current NineToothed version."
        ) from exc

    function.restype = ctypes.c_int
    enter.argtypes = []
    enter.restype = ctypes.c_int
    leave.argtypes = []
    leave.restype = None

    return library, function, enter, leave


def _compile_aot_library(compilation, source: Path, library: Path) -> None:
    _ensure_c_compiler()
    artifact = compilation.artifact
    kernel_name = f"{artifact.kernel_name}_kernel"
    signature = _compile_signature(compilation)
    grid = _compile_grid(compilation)
    num_warps, num_stages = _compile_schedule(compilation)
    library.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    environment["TRITON_CACHE_DIR"] = str(TRITON_CACHE_DIR)

    with tempfile.TemporaryDirectory(dir=library.parent) as temporary_dir:
        temporary = Path(temporary_dir)
        compiled = temporary / "compiled"
        linked = temporary / "linked"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "triton.tools.compile",
                str(source),
                "--kernel-name",
                kernel_name,
                "--signature",
                signature,
                "--grid",
                grid,
                "--num-warps",
                str(num_warps),
                "--num-stages",
                str(num_stages),
                "--out-name",
                kernel_name,
                "--out-path",
                str(compiled),
            ],
            check=True,
            env=environment,
        )
        headers = tuple(temporary.glob("compiled.*.h"))
        sources = tuple(temporary.glob("compiled.*.c"))

        if not headers or not sources:
            raise RuntimeError("Triton AOT compiler did not produce C artifacts.")

        kernel_names = _triton_aot_kernel_names(sources)
        context_guard = temporary / "ninetoothed_triton_context_guard.cu"
        context_guard.write_text(
            _triton_context_guard_source(kernel_names),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "triton.tools.link",
                *(str(path) for path in headers),
                "--out",
                str(linked),
            ],
            check=True,
            env=environment,
        )
        output = temporary / library.name
        subprocess.run(
            [
                find_nvcc(),
                "-shared",
                "-std=c++17",
                "-Xcompiler",
                "-fPIC",
                "-Xcompiler",
                "-pthread",
                "-O3",
                *(str(path) for path in sources),
                str(linked.with_suffix(".c")),
                str(context_guard),
                "-lcuda",
                "-Xlinker",
                "-z",
                "-Xlinker",
                "defs",
                "-o",
                str(output),
            ],
            check=True,
        )
        os.replace(output, library)


def _triton_aot_kernel_names(sources: tuple[Path, ...]) -> tuple[str, ...]:
    """Return low-level kernels with matching module, function, and loader symbols."""
    if not sources:
        raise ValueError("Unsupported Triton AOT source format: no C sources found.")

    kernel_names = []

    for source in sources:
        text = source.read_text(encoding="utf-8")
        modules = _TRITON_MODULE_PATTERN.findall(text)
        functions = _TRITON_FUNCTION_PATTERN.findall(text)
        loaders = _TRITON_LOADER_PATTERN.findall(text)
        module_names = set(modules)

        if (
            not modules
            or len(modules) != len(module_names)
            or len(functions) != len(set(functions))
            or len(loaders) != len(set(loaders))
            or module_names != set(functions)
            or module_names != set(loaders)
        ):
            raise ValueError(
                f"Unsupported Triton AOT source format in `{source}`: expected "
                "one matching `CUmodule`, `CUfunction`, and `load_*` symbol per "
                "low-level kernel."
            )

        kernel_names.extend(modules)

    if len(kernel_names) != len(set(kernel_names)):
        raise ValueError(
            "Unsupported Triton AOT source format: duplicate low-level kernel symbols."
        )
    return tuple(kernel_names)


def _triton_context_guard_source(kernel_names: tuple[str, ...]) -> str:
    """Generate the CUDA-context guard linked beside Triton's AOT launchers."""
    declarations = "\n".join(
        (
            f'extern "C" CUmodule {name}_mod;\n'
            f'extern "C" CUfunction {name}_func;\n'
            f'extern "C" void load_{name}(void);'
        )
        for name in kernel_names
    )
    resets_and_loads = "\n".join(
        (
            f"        {name}_mod = nullptr;\n"
            f"        {name}_func = nullptr;\n"
            f"        load_{name}();\n"
            f"        if ({name}_mod == nullptr || {name}_func == nullptr) {{\n"
            "            launch_mutex.unlock();\n"
            "            return CUDA_ERROR_INVALID_HANDLE;\n"
            "        }"
        )
        for name in kernel_names
    )
    stores = "\n".join(
        (
            f"        state.modules[{index}] = {name}_mod;\n"
            f"        state.functions[{index}] = {name}_func;"
        )
        for index, name in enumerate(kernel_names)
    )
    restores = "\n".join(
        (
            f"        {name}_mod = found->second.modules[{index}];\n"
            f"        {name}_func = found->second.functions[{index}];"
        )
        for index, name in enumerate(kernel_names)
    )

    return f"""#include <array>
#include <cuda.h>
#include <mutex>
#include <unordered_map>

{declarations}

namespace {{
struct State {{
    std::array<CUmodule, {len(kernel_names)}> modules{{}};
    std::array<CUfunction, {len(kernel_names)}> functions{{}};
}};

std::mutex launch_mutex;
std::unordered_map<CUcontext, State> context_states;
}}

extern "C" CUresult ninetoothed_triton_enter(void) {{
    launch_mutex.lock();

    CUcontext context = nullptr;
    CUresult result = cuCtxGetCurrent(&context);

    if (result != CUDA_SUCCESS) {{
        launch_mutex.unlock();
        return result;
    }}

    if (context == nullptr) {{
        launch_mutex.unlock();
        return CUDA_ERROR_INVALID_CONTEXT;
    }}

    try {{
        auto found = context_states.find(context);

        if (found == context_states.end()) {{
{resets_and_loads}

            State state{{}};
{stores}
            context_states.emplace(context, state);
        }} else {{
{restores}
        }}
    }} catch (...) {{
        launch_mutex.unlock();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }}

    return CUDA_SUCCESS;
}}

extern "C" void ninetoothed_triton_leave(void) {{
    launch_mutex.unlock();
}}
"""


def _compile_signature(compilation) -> str:
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    values = []

    for binding in compilation.launch_abi.kernel_args:
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
            values.append(f"*{_triton_dtype(specs[binding.source].dtype)}")
        elif binding.kind == "scalar":
            values.append(_triton_scalar_dtype(specs[binding.source].dtype))
        elif binding.kind in {"constexpr", "meta"}:
            if binding.value is None:
                raise ValueError(f"Triton AOT requires a value for `{binding.name}`.")

            values.append(str(binding.value))
        else:
            values.append("i64")

    private_meta = _private_meta_values(compilation)
    values.extend(
        str(private_meta[name])
        for name in dict(compilation.artifact.metadata.get("layout_transfer", {})).get(
            "private_meta_parameters", ()
        )
    )
    values.append(str(_compile_block(compilation)))

    return ",".join(values)


def _triton_dtype(dtype) -> str:
    name = str(dtype).split(".")[-1]
    aliases = {
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "fp64": "fp64",
        "float64": "fp64",
        "bool": "i1",
    }

    if name in aliases:
        return aliases[name]

    if name.startswith(("int", "uint")):
        prefix = "i" if name.startswith("int") else "u"

        return prefix + "".join(character for character in name if character.isdigit())

    raise TypeError(f"Unsupported Triton AOT dtype: {dtype!r}.")


def _triton_scalar_dtype(dtype) -> str:
    name = str(dtype).split(".")[-1]

    if name in {
        "fp16",
        "float16",
        "bf16",
        "bfloat16",
        "fp32",
        "float32",
        "fp64",
        "float64",
    }:
        return "fp64"
    return _triton_dtype(dtype)


def _compile_block(compilation) -> int:
    mode = dict(compilation.artifact.metadata.get("program_mode", {}))
    reduction = dict(
        compilation.artifact.metadata.get("ssa_schedule", {}).get("reduction", {})
    )

    if mode.get("block") or mode.get("scalar"):
        return 1

    if mode.get("vector"):
        total = (
            _constant_reduction_extent(compilation, reduction)
            if reduction.get("mode") == "row-vector"
            else _constant_grid_total(compilation)
        )
        block = 1 << max(0, (total - 1).bit_length())
        limit = compilation.artifact.metadata.get("vector_numel_limit")

        if limit is not None and block > limit:
            raise ValueError(
                f"Triton row-vector reduction extent {total} requires "
                f"BLOCK={block}, exceeding the backend tensor numel limit "
                f"{limit}; hierarchical reduction is not implemented."
            )
        return block
    return 256


def _constant_reduction_extent(compilation, reduction) -> int:
    expression = replace_symbols(
        str(reduction["extent"]),
        {
            binding.name: str(binding.value)
            for binding in compilation.launch_abi.kernel_args
            if binding.kind in {"meta", "constexpr"} and binding.value is not None
        },
    )

    try:
        import sympy

        value = sympy.sympify(expression)

        if value.free_symbols:
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Triton AOT row-vector reductions require a statically "
            "specialized reduction extent."
        ) from exc


def _constant_grid_total(compilation) -> int:
    expression = _specialized_grid_total(compilation)

    try:
        import sympy

        value = sympy.sympify(expression)

        if value.free_symbols:
            raise ValueError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Triton AOT vector programs require a statically specialized domain."
        ) from exc


def _compile_grid(compilation) -> str:
    total = _specialized_grid_total(compilation).replace("//", "/")
    mode = dict(compilation.artifact.metadata.get("program_mode", {}))

    if not any(mode.get(name) for name in ("block", "scalar", "vector")):
        block = _compile_block(compilation)
        total = f"((({total}) + {block - 1}) / {block})"
    return f"{total},1,1"


def _specialized_grid_total(compilation) -> str:
    expression = compilation.launch_plan.grid[0].render()
    replacements = {
        binding.name: str(binding.value)
        for binding in compilation.launch_abi.kernel_args
        if binding.kind in {"meta", "constexpr"} and binding.value is not None
    }
    replacements.update(
        {
            f"_ninetoothed_{name.lower()}": str(value)
            for name, value in _private_meta_values(compilation).items()
        }
    )

    return replace_symbols(expression, replacements)


def _private_meta_values(compilation) -> dict[str, int]:
    candidates = tuple(compilation.launch_plan.tuning_candidates)

    if not candidates:
        return {}

    return {
        str(name): int(value)
        for name, value in dict(
            candidates[0].get("private_meta_parameters", {})
        ).items()
    }


def _compile_schedule(compilation) -> tuple[int, int]:
    metadata = getattr(compilation.artifact, "metadata", {}) or {}
    schedule = dict(metadata.get("ssa_schedule", {}))
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    selected = dict(candidates[0]) if candidates else {}
    request = getattr(compilation, "request", None)
    warps = (
        getattr(request, "num_warps", None)
        or selected.get("num_warps")
        or schedule.get("num_warps")
        or 4
    )
    stages = (
        getattr(request, "num_stages", None)
        or selected.get("num_stages")
        or schedule.get("num_stages")
        or 3
    )

    if isinstance(warps, tuple):
        warps = warps[0]

    if isinstance(stages, tuple):
        stages = stages[0]
    return int(warps), _legalize_triton_num_stages(stages)


def _aot_wrapper(
    function,
    enter,
    leave,
    abi,
    tensor_specs,
    *,
    validate_bindings=None,
):
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _bound_values,
        _empty_launch,
        _first_output,
        _public_values,
    )

    specs = {spec.name: spec for spec in tensor_specs}
    runtime_bindings = tuple(
        binding
        for binding in abi.kernel_args
        if binding.kind not in {"constexpr", "meta"}
    )
    function.argtypes = [
        ctypes.c_void_p,
        *(_triton_aot_ctype(binding, specs) for binding in runtime_bindings),
    ]

    def launch(*args, **kwargs):
        import torch

        public = _public_values(abi, args, kwargs, specs=tensor_specs)
        _validate_aot_constants(abi, public)

        if validate_bindings is not None:
            validate_bindings(public)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        runtime_abi = replace(
            abi,
            kernel_args=runtime_bindings,
        )
        values, keepalive = _bound_values(
            runtime_abi,
            public,
            scalar_mode="cuda",
            specs=specs,
            cuda_scalar=_triton_aot_scalar,
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        enter_result = enter()

        if enter_result != 0:
            raise KernelLaunchError(enter_result)

        try:
            result = function(stream, *values)
        finally:
            leave()

        del keepalive

        if result != 0:
            raise KernelLaunchError(result)
        return _first_output(abi, public)

    return launch


def _triton_aot_scalar(value, dtype):
    name = str(dtype).split(".")[-1]
    name = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(name, name)

    if hasattr(value, "item"):
        value = value.item()

    if name in {"float16", "bfloat16", "float32", "float64"}:
        return ctypes.c_double(float(value))

    ctype = {
        "bool": ctypes.c_int8,
        "int8": ctypes.c_int8,
        "uint8": ctypes.c_uint8,
        "int16": ctypes.c_int16,
        "uint16": ctypes.c_uint16,
        "int32": ctypes.c_int32,
        "uint32": ctypes.c_uint32,
        "int64": ctypes.c_int64,
        "uint64": ctypes.c_uint64,
    }.get(name)

    if ctype is None:
        raise TypeError(f"Unsupported Triton AOT scalar dtype: {dtype!r}.")
    return ctype(value)


def _triton_aot_ctype(binding, specs):
    if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
        return ctypes.c_void_p

    if binding.kind == "scalar":
        name = str(specs[binding.source].dtype).split(".")[-1]
        name = {
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
            "bf16": "bfloat16",
        }.get(name, name)

        if name in {"float16", "bfloat16", "float32", "float64"}:
            return ctypes.c_double

        ctype = {
            "bool": ctypes.c_int8,
            "int8": ctypes.c_int8,
            "uint8": ctypes.c_uint8,
            "int16": ctypes.c_int16,
            "uint16": ctypes.c_uint16,
            "int32": ctypes.c_int32,
            "uint32": ctypes.c_uint32,
            "int64": ctypes.c_int64,
            "uint64": ctypes.c_uint64,
        }.get(name)

        if ctype is None:
            raise TypeError(f"Unsupported Triton AOT scalar dtype: {name!r}.")
        return ctype
    return ctypes.c_int64


def _validate_aot_constants(abi, public) -> None:
    for binding in abi.kernel_args:
        if binding.kind not in {"constexpr", "meta"} or binding.value is None:
            continue

        if binding.source not in public:
            continue

        actual = public[binding.source]

        if hasattr(actual, "item"):
            actual = actual.item()

        if actual != binding.value:
            raise ValueError(
                f"Kernel argument `{binding.source}` is specialized to "
                f"{binding.value!r}, but received {actual!r}."
            )


__all__ = ["TritonMaterializer"]
