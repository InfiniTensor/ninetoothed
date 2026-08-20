"""Multi-configuration builds backed by unified SSA compilations."""

import inspect
import threading
from dataclasses import dataclass

from ninetoothed.auto_tuner import AutoTuner
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


@dataclass(frozen=True, kw_only=True)
class _Variant:
    key: tuple
    handle: object


class _CandidateGroup:
    def __init__(self, handles, keys, *, cache_namespace):
        self._handles = tuple(handles)
        self._tuner = (
            AutoTuner(
                self._handles,
                tuple(keys),
                cache_namespace=cache_namespace,
                validator=_handle_validator(self._handles[0]),
            )
            if len(self._handles) > 1
            else None
        )
        self._selected = self._handles[0]
        self._sync(self._selected)

    def __call__(self, *args, **kwargs):
        if self._tuner is None:
            result = self._selected(*args, **kwargs)
        else:
            result = self._tuner(*args, **kwargs)
            arg_key = self._tuner._make_arg_key(args, kwargs)
            self._selected = self._tuner._best_func[arg_key]
            self._sync(self._selected)
        return result

    def _sync(self, handle):
        for name in (
            "_source",
            "_artifact",
            "_backend",
            "_platform",
            "_kernel",
            "_library",
            "_ssa",
            "_pass_trace",
            "_launch_plan",
            "_built_artifact",
        ):
            setattr(self, name, getattr(handle, name))


class _BuildHandle:
    def __init__(self, variants, num_key_args):
        self._variants = tuple(variants)
        self._num_key_args = num_key_args
        first = self._variants[0].handle
        self._sync(first)
        self._launch = self.__call__

    def __call__(self, *args, **kwargs):
        key = args[-self._num_key_args :] if self._num_key_args else ()
        tensor_args = args[: -self._num_key_args] if self._num_key_args else args
        normalized = tuple(_arg_key(value) for value in key)

        for variant in self._variants:
            if variant.key == normalized:
                result = variant.handle(*tensor_args, **kwargs)
                self._sync(variant.handle)

                return result

        raise ValueError(f"No compiled kernel configuration matches {normalized}.")

    def _sync(self, handle):
        for name in (
            "_source",
            "_artifact",
            "_backend",
            "_platform",
            "_kernel",
            "_library",
            "_ssa",
            "_pass_trace",
            "_launch_plan",
            "_built_artifact",
        ):
            setattr(self, name, getattr(handle, name))


def _handle_validator(handle):
    from ninetoothed.compiler.runtime import _public_values
    from ninetoothed.targets import runtime_device_types

    def validate(args, kwargs):
        _public_values(
            handle._compilation.launch_abi,
            args,
            kwargs,
            specs=handle._compilation.kernel.tensors,
            device_types=runtime_device_types(handle._compilation),
        )

    return validate


class _LazyKernel:
    def __init__(self, factory):
        self._factory = factory
        self._kernel = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        if self._kernel is None:
            with self._lock:
                if self._kernel is None:
                    self._kernel = self._factory()
        return self._kernel(*args, **kwargs)


def build(
    premake,
    configs,
    *,
    backend=None,
    platform=None,
    compute_arch=None,
    caller="cuda",
    output_dir,
    kernel_name=None,
    meta_parameters=None,
    lazy=False,
    pipeline=None,
    pass_options=None,
    **backend_options,
):
    """Compile configured variants and return a common callable dispatcher."""
    configs = tuple(configs)
    meta_parameters = tuple(meta_parameters or ())

    if lazy:
        return _LazyKernel(
            lambda: build(
                premake,
                configs,
                meta_parameters=meta_parameters,
                caller=caller,
                backend=backend,
                platform=platform,
                compute_arch=compute_arch,
                kernel_name=kernel_name,
                output_dir=output_dir,
                pipeline=pipeline,
                pass_options=pass_options,
                **backend_options,
            )
        )

    signature = inspect.signature(premake)
    runtime_names = tuple(
        name for name in signature.parameters if name not in meta_parameters
    )
    base_name = kernel_name or _callable_name(premake)
    grouped = {}

    for index, (args, kwargs, compiler_options) in enumerate(configs):
        arrangement, application, tensors = premake(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(_arg_key(bound.arguments[name]) for name in runtime_names)
        variant_name = f"{base_name}_{index}" if len(configs) > 1 else base_name
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=arrangement,
                application=application,
                tensors=tuple(tensors),
                backend=backend,
                platform=platform,
                compute_arch=compute_arch,
                caller=caller,
                kernel_name=variant_name,
                num_warps=compiler_options.get("num_warps"),
                num_stages=compiler_options.get("num_stages"),
                max_num_configs=1,
                pipeline=compiler_options.get("pipeline", pipeline),
                pass_options=compiler_options.get("pass_options", pass_options),
                backend_options=backend_options
                | dict(compiler_options.get("backend_options", {})),
            )
        )
        handle = DEFAULT_COMPILER.materialize(
            compilation,
            output_dir=output_dir,
            mode="aot",
        )
        group = grouped.setdefault(key, {"handles": [], "keys": []})
        group["handles"].append(handle)
        group["keys"].append(
            (
                handle._built_artifact.cache_key,
                tuple(
                    sorted((str(name), repr(value)) for name, value in kwargs.items())
                ),
                tuple(
                    sorted(
                        (str(name), repr(value))
                        for name, value in compiler_options.items()
                    )
                ),
            )
        )

    if not grouped:
        raise ValueError("At least one build configuration is required.")

    variants = tuple(
        _Variant(
            key=key,
            handle=_CandidateGroup(
                group["handles"],
                group["keys"],
                cache_namespace=(
                    f"build_{base_name}_{backend or 'triton'}_"
                    f"{platform or 'generic'}_{compute_arch or 'default'}"
                ),
            ),
        )
        for key, group in grouped.items()
    )

    return _BuildHandle(variants, len(runtime_names))


def _arg_key(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _callable_name(function):
    while hasattr(function, "func"):
        function = function.func
    return getattr(function, "__name__", type(function).__name__.lower())


__all__ = ["build"]
