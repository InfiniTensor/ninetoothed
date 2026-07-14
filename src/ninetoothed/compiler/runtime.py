"""Materialize backend artifacts into callable runtime handles."""

import copy
import ctypes
import importlib.util
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.compiler.cache import (
    atomic_write_bytes,
    compilation_cache_key,
    write_manifest,
    write_source,
)
from ninetoothed.ir import LaunchABI, LaunchBinding, ir_to_dict


class KernelLaunchError(RuntimeError):
    """Raised when a materialized backend launcher reports an error."""

    def __init__(self, error_code):
        super().__init__(f"Kernel launch failed with error code: {error_code}.")


def overflow_terms(argument_names, tensor_ndims):
    """Return C launcher guards for shape and stride values outside int32."""
    int32_min = -(2**31)
    int32_max = 2**31 - 1

    return tuple(
        term
        for name, ndim in zip(argument_names, tensor_ndims)
        for dim in range(ndim)
        for term in (
            f"{name}.shape[{dim}] > {int32_max}ULL",
            f"{name}.strides[{dim}] > {int32_max}LL",
            f"{name}.strides[{dim}] < {int32_min}LL",
        )
    )


class Handle:
    def __init__(self, compilation, kernel, launch, source, library=None):
        self._compilation = compilation
        self._artifact = compilation.artifact
        self._backend = compilation.artifact.backend.value
        self._kernel = kernel
        self._launch = launch
        self._source = str(source)
        self._library = None if library is None else str(library)
        self._ssa = compilation.kernel.ssa
        self._pass_trace = compilation.pass_trace
        self._launch_plan = compilation.launch_plan
        cache_key = compilation_cache_key(compilation)
        manifest = (
            Path(self._library).with_suffix(".manifest.json")
            if self._library is not None
            else Path(self._source).with_suffix(".manifest.json")
        )
        write_manifest(
            manifest,
            _built_manifest(
                compilation,
                cache_key,
                Path(self._source),
                None if self._library is None else Path(self._library),
            ),
        )
        self._built_artifact = BuiltArtifact(
            source=compilation.artifact,
            cache_key=cache_key,
            source_path=self._source,
            binary_path=self._library,
            manifest_path=str(manifest),
            abi=compilation.artifact.metadata.get("launch_abi", {}),
        )

    def __call__(self, *args, **kwargs):
        return self._launch(*args, **kwargs)


def materialize(
    compilation,
    *,
    output_dir: str | Path | None = None,
    mode: str = "jit",
) -> Handle:
    target = compilation.artifact.backend

    if target != Target.TRITON and _requires_runtime_specialization(compilation):
        return _materialize_lazy(compilation, output_dir=output_dir, mode=mode)

    from ninetoothed.backends.materializers import materializer_for

    materializer = materializer_for(target)

    if mode == "jit":
        return materializer.jit_materialize(compilation, output_dir=output_dir)

    if mode == "aot":
        if output_dir is None:
            raise ValueError("AOT materialization requires an output directory.")
        return materializer.aot_build(compilation, output_dir=output_dir)

    raise ValueError(f"Unknown materialization mode `{mode}`.")


def load_built_artifact(built: BuiltArtifact):
    """Load a materialized binary and restore its public Launch ABI callable."""
    from ninetoothed.backends.materializers import materializer_for

    return materializer_for(built.source.backend).load_built_artifact(built)


def _launch_abi_from_dict(value) -> LaunchABI:
    return LaunchABI(
        public_args=tuple(value.get("public_args", ())),
        kernel_args=tuple(
            LaunchBinding(**dict(binding)) for binding in value.get("kernel_args", ())
        ),
        outputs=tuple(value.get("outputs", ())),
        shape_params=tuple(value.get("shape_params", ())),
    )


def _runtime_specs(artifact):
    return tuple(
        SimpleNamespace(
            name=str(value["name"]),
            ndim=int(value.get("ndim", 0)),
            dtype=value.get("dtype"),
            attrs=dict(value.get("attrs", {})),
        )
        for value in artifact.metadata.get("tensors", ())
    )


def _requires_runtime_specialization(compilation) -> bool:
    if any(tensor.dtype is None for tensor in compilation.kernel.tensors):
        return True

    if compilation.request.specialization_values:
        return False
    return compilation.artifact.backend == Target.TILELANG and any(
        binding.kind
        in {
            "shape",
            "stride",
            "meta",
            "jagged_values_numel",
            "jagged_offsets_numel",
        }
        for binding in compilation.launch_abi.kernel_args
    )


def _materialize_lazy(compilation, *, output_dir=None, mode="jit") -> Handle:
    artifact = compilation.artifact
    suffix = {
        Target.CUDA: "cu",
        Target.TILELANG: "tilelang.py",
    }[artifact.backend]
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        suffix,
        cache_key=cache_key,
    )
    specializations: dict[tuple[tuple[str, str], ...], Handle] = {}
    handle = None

    def launch(*args, **kwargs):
        nonlocal handle
        public = _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
        )
        dtypes = _runtime_dtypes(compilation, public)
        specialization_values = _runtime_specialization_values(compilation, public)
        key = (
            *(f"dtype:{name}={value}" for name, value in sorted(dtypes.items())),
            *(
                f"symbol:{name}={value!r}"
                for name, value in sorted(specialization_values.items())
            ),
        )

        if key not in specializations:
            from ninetoothed.compiler import compile_kernel

            request = replace(
                compilation.request,
                tensors=copy.deepcopy(compilation.request.tensors),
                tensor_dtypes=dtypes,
                specialization_values=specialization_values,
            )
            specializations[key] = materialize(
                compile_kernel(request), output_dir=output_dir, mode=mode
            )

        specialized = specializations[key]

        if handle is not None:
            handle._artifact = specialized._artifact
            handle._kernel = specialized._kernel
            handle._source = specialized._source
            handle._library = specialized._library
            handle._ssa = specialized._ssa
            handle._pass_trace = specialized._pass_trace
            handle._launch_plan = specialized._launch_plan
            handle._built_artifact = specialized._built_artifact
        return specialized(
            *args,
            **_filter_runtime_kwargs(specialized._compilation.launch_abi, kwargs),
        )

    handle = Handle(compilation, None, launch, source)

    return handle


def _runtime_specialization_values(compilation, public) -> dict[str, Any]:
    values = {}
    dynamic_parameters = set(compilation.launch_plan.dynamic_parameters)

    for binding in compilation.launch_abi.kernel_args:
        if binding.name not in dynamic_parameters:
            continue

        value = _binding_value(binding, public)

        if hasattr(value, "item"):
            value = value.item()

        values[binding.name] = value
    return values


def _runtime_dtypes(compilation, public) -> dict[str, str]:
    result = {}

    for tensor in compilation.kernel.tensors:
        if tensor.dtype is not None or tensor.name not in public:
            continue

        value = public[tensor.name]
        dtype = getattr(value, "dtype", None)

        if dtype is not None:
            result[tensor.name] = str(dtype).split(".")[-1]
        elif isinstance(value, bool):
            result[tensor.name] = "bool"
        elif isinstance(value, int):
            result[tensor.name] = "int64"
        elif isinstance(value, float):
            result[tensor.name] = "float32"
    return result


def _runtime_wrapper(
    function,
    abi: LaunchABI,
    *,
    low_level: bool = True,
    specs=(),
):
    def launch(*args, **kwargs):
        public = _public_values(abi, args, kwargs, specs=specs)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        values, keepalive = _bound_values(abi, public, scalar_mode="value")
        bindings = abi.kernel_args

        if low_level:
            values, flattened = _flatten_ffi_tensor_args(bindings, values)
            keepalive.extend(flattened)

        result = function(*values) if low_level else function(*args)
        del keepalive

        if result is not None:
            return result
        return _first_output(abi, public)

    return launch


def _public_values(abi: LaunchABI, args, kwargs, *, specs=()) -> dict[str, Any]:
    if len(args) > len(abi.public_args):
        raise TypeError(f"Expected at most {len(abi.public_args)} arguments.")

    positional_names = abi.public_args[: len(args)]
    duplicates = tuple(name for name in positional_names if name in kwargs)

    if duplicates:
        raise TypeError(
            f"Kernel arguments passed twice: {', '.join(sorted(duplicates))}."
        )

    accepted = set(abi.public_args) | {
        binding.source
        for binding in abi.kernel_args
        if binding.kind == "meta" and binding.source is not None
    }
    unknown = tuple(name for name in kwargs if name not in accepted)

    if unknown:
        raise TypeError(f"Unknown kernel arguments: {', '.join(sorted(unknown))}.")

    values = dict(zip(abi.public_args, args))
    values.update(kwargs)
    missing = tuple(name for name in abi.public_args if name not in values)

    if missing:
        raise TypeError(f"Missing kernel arguments: {', '.join(missing)}.")

    _validate_runtime_values(values, specs)

    return values


def _filter_runtime_kwargs(abi: LaunchABI, kwargs) -> dict[str, Any]:
    accepted = set(abi.public_args) | {
        binding.source
        for binding in abi.kernel_args
        if binding.kind == "meta" and binding.source is not None
    }

    return {name: value for name, value in kwargs.items() if name in accepted}


def _validate_runtime_values(values, specs) -> None:
    expected_device = None

    for spec in specs:
        if spec.name not in values:
            continue

        value = values[spec.name]
        expected_device = _validate_tensor_contract(spec, value, expected_device)
        _validate_dtype_contract(spec, value)


def _validate_tensor_contract(spec, value, expected_device):
    source_ndim = int(spec.attrs.get("source_ndim", spec.ndim))

    if source_ndim == 0:
        return expected_device

    shape = getattr(value, "shape", None)

    if shape is None:
        raise TypeError(f"Kernel argument `{spec.name}` must be a tensor.")

    if len(shape) != source_ndim:
        raise TypeError(
            f"Kernel argument `{spec.name}` has rank {len(shape)}; "
            f"expected {source_ndim}."
        )

    device = getattr(value, "device", None)

    if device is None:
        return expected_device

    device_type = getattr(device, "type", str(device).split(":")[0])

    if device_type != "cuda":
        raise TypeError(f"Kernel argument `{spec.name}` must be on a CUDA device.")

    if expected_device is not None and device != expected_device:
        raise TypeError("All tensor arguments must use the same CUDA device.")
    return device if expected_device is None else expected_device


def _validate_dtype_contract(spec, value) -> None:
    source_dtype = spec.attrs.get("source_dtype", spec.dtype)

    if source_dtype is None or not hasattr(value, "dtype"):
        return

    actual = str(value.dtype).split(".")[-1]
    expected = _canonical_dtype(source_dtype)

    if actual != expected:
        raise TypeError(
            f"Kernel argument `{spec.name}` has dtype {actual}; expected {expected}."
        )


def _canonical_dtype(dtype) -> str:
    name = str(dtype).split(".")[-1]

    return {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(name, name)


def _bound_values(abi, public, *, scalar_mode, specs=None, cuda_scalar=None):
    values = []
    keepalive = []

    for binding in abi.kernel_args:
        value = _binding_value(binding, public)

        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
            if scalar_mode == "cuda":
                value = ctypes.c_void_p(value.data_ptr())
        elif binding.kind in {"scalar", "constexpr"}:
            if hasattr(value, "item"):
                value = value.item()

            if scalar_mode == "cuda":
                spec = specs.get(binding.source) if specs is not None else None

                if spec is not None and cuda_scalar is None:
                    raise RuntimeError(
                        "CUDA scalar binding requires a dtype converter."
                    )

                value = (
                    cuda_scalar(value, spec.dtype)
                    if spec is not None
                    else ctypes.c_int64(int(value))
                )
        elif scalar_mode == "cuda":
            value = ctypes.c_int64(int(value))

        values.append(value)
    return values, keepalive


def _flatten_ffi_tensor_args(bindings, values):
    flattened = []
    result = []

    for binding, value in zip(bindings, values):
        if binding.kind in {
            "tensor",
            "jagged_values",
            "jagged_offsets",
        } and hasattr(value, "as_strided"):
            storage_numel = value.untyped_storage().nbytes() // value.element_size()
            storage_offset = value.storage_offset()
            value = value.as_strided(
                (storage_numel - storage_offset,),
                (1,),
                storage_offset=storage_offset,
            )
            flattened.append(value)

        result.append(value)
    return result, flattened


def _binding_value(binding: LaunchBinding, public):
    if binding.kind in {"tensor", "scalar", "constexpr"}:
        return public[binding.source]

    if binding.kind.startswith("jagged_"):
        return _jagged_binding_value(binding, public[binding.source])

    if binding.kind == "shape":
        return public[binding.source].shape[binding.dim]

    if binding.kind == "stride":
        return public[binding.source].stride(binding.dim)

    if binding.kind == "meta":
        if binding.source in public:
            return public[binding.source]

        if binding.value is not None:
            return binding.value

        raise TypeError(f"Missing launch meta-parameter `{binding.name}`.")
    return binding.value


def _jagged_binding_value(binding: LaunchBinding, value):
    accessors = {
        "jagged_values": lambda: value.values(),
        "jagged_offsets": lambda: value.offsets(),
        "jagged_values_numel": lambda: value.values().numel(),
        "jagged_offsets_numel": lambda: value.offsets().numel(),
        "jagged_max_seq_len": lambda: value.offsets().diff().max().item(),
    }

    try:
        return accessors[binding.kind]()
    except KeyError:
        return binding.value


def _first_output(abi, public):
    return public[abi.outputs[0]] if abi.outputs else None


def _empty_launch(abi, public) -> bool:
    names = abi.outputs or tuple(
        binding.source
        for binding in abi.kernel_args
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}
        and binding.source is not None
    )

    return any(
        name in public and hasattr(public[name], "numel") and public[name].numel() == 0
        for name in names
    )


def import_python_module(path: str | Path, module_name: str | None = None):
    """Import a generated Python module from an artifact path."""
    path = Path(path)
    module_name = module_name or f"_ninetoothed_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generated artifact `{path}`.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _export_library_atomic(module, library: Path) -> None:
    temporary = _temporary_output(library)

    try:
        module.export_library(str(temporary))
        os.replace(temporary, library)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()

    return temporary


def _replace_file(source: Path, destination: Path) -> None:
    temporary = _temporary_output(destination)

    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_library(cache_library: Path, output_dir, filename: str) -> Path:
    if output_dir is None:
        return cache_library

    destination = Path(output_dir) / filename
    atomic_write_bytes(destination, cache_library.read_bytes())

    return destination


def _built_manifest(compilation, cache_key, source, library):
    return {
        "schema": 2,
        "cache_key": cache_key,
        "backend": compilation.artifact.backend.value,
        "kernel_name": compilation.artifact.kernel_name,
        "entrypoint": compilation.artifact.entrypoint,
        "source": str(source),
        "library": None if library is None else str(library),
        "launch_abi": compilation.artifact.metadata.get("launch_abi", {}),
        "launch_plan": ir_to_dict(compilation.launch_plan),
        "pass_trace": compilation.pass_trace,
    }


__all__ = [
    "Handle",
    "KernelLaunchError",
    "import_python_module",
    "load_built_artifact",
    "materialize",
    "overflow_terms",
]
