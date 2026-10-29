"""Share native pointer ABIs and verified launch preparation across backends."""

import ctypes
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ninetoothed.dtype import normalize_dtype


def materialize_native_library(
    compilation,
    *,
    source_suffix,
    compile_library,
    wrapper,
    output_dir=None,
    wrapper_options=None,
):
    """Build and bind a native library using one resolved SDK identity."""
    from ninetoothed.compiler.cache import (
        artifact_directory,
        cache_lock,
        compilation_cache_key,
        compilation_toolchain_identity,
        write_manifest,
        write_source,
    )
    from ninetoothed.compiler.runtime import Handle, _built_manifest, _publish_library

    artifact = compilation.artifact

    toolchain = compilation_toolchain_identity(compilation)
    compiler = toolchain[artifact.backend.value]

    if not compiler.get("available"):
        raise RuntimeError(
            f"The {artifact.backend.value} backend requires its local SDK compiler."
        )

    cache_key = compilation_cache_key(compilation, toolchain=toolchain)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        source_suffix,
        cache_key=cache_key,
    )
    library_name = f"{artifact.kernel_name}.{artifact.backend.value}.so"
    cache_library = artifact_directory(cache_key) / library_name

    with cache_lock(cache_library):
        if not cache_library.is_file():
            compile_library(
                source,
                cache_library,
                arch=compiler.get(
                    "arch",
                    dict(compilation.request.backend_options or {}).get(
                        "arch", "native"
                    ),
                ),
                compiler=compiler["path"],
            )

        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _built_manifest(
                compilation, cache_key, source, cache_library, toolchain=toolchain
            ),
        )

    library_path = _publish_library(cache_library, output_dir, library_name)
    function = _bind_native_function(cache_library, artifact.entrypoint)
    wrapped = wrapper(
        function,
        compilation.launch_abi,
        compilation.kernel.tensors,
        **(wrapper_options or {}),
    )

    return Handle(
        compilation,
        function,
        wrapped,
        source,
        library_path,
        cache_key=cache_key,
        toolchain=toolchain,
    )


def _bind_native_function(path, entrypoint, *, snapshot=False):
    if snapshot:
        from ninetoothed.compiler.cache import CACHE_DIR, atomic_write_bytes

        data = Path(path).read_bytes()
        path = (
            CACHE_DIR
            / "loaded-libraries"
            / hashlib.sha256(data).hexdigest()
            / "kernel.so"
        )
        atomic_write_bytes(path, data)

    library = ctypes.CDLL(str(path))
    function = getattr(library, entrypoint)
    function.restype = ctypes.c_int

    return function


def load_native_artifact(built, wrapper, *, validate=None, wrapper_options=None):
    """Rebind the saved ABI after target validation, without analyzing SSA."""
    if built.binary_path is None:
        raise ValueError(
            f"Built artifact for {built.source.backend.value} does not contain a binary path."
        )

    from ninetoothed.compiler.runtime import _launch_abi_from_dict, _runtime_specs

    abi, specs = _launch_abi_from_dict(built.abi), _runtime_specs(built.source)

    if validate is not None:
        validate(built, abi, specs)

    function = _bind_native_function(
        built.binary_path, built.source.entrypoint, snapshot=True
    )

    return wrapper(function, abi, specs, **(wrapper_options or {}))


@contextmanager
def native_library_output(library):
    """Publish a complete library atomically and clean up failed builds."""
    library.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=library.parent,
        prefix=f".{library.stem}.",
        suffix=library.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()

    try:
        yield temporary
        os.replace(temporary, library)
    finally:
        temporary.unlink(missing_ok=True)


def _native_argument(binding, value, spec_by_name):
    if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
        return ctypes.c_void_p(value.data_ptr())

    if binding.kind in {"scalar", "constexpr"}:
        spec = spec_by_name.get(binding.source)

        return native_scalar(value, spec.dtype if spec is not None else "int64")
    return ctypes.c_int64(int(value))


def native_scalar(value, dtype):
    """Convert a scalar to the C representation declared by the backend ABI."""
    dtype = normalize_dtype(str(dtype))

    if dtype in {"float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        import torch

        storage_dtype = (
            torch.uint16 if dtype in {"float16", "bfloat16"} else torch.uint8
        )
        scalar = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value)
        )
        bits = scalar.to(dtype=getattr(torch, dtype)).view(storage_dtype).item()
        ctype = ctypes.c_uint16 if storage_dtype == torch.uint16 else ctypes.c_uint8

        return ctype(bits)

    ctype = {
        "float32": ctypes.c_float,
        "float64": ctypes.c_double,
        "int8": ctypes.c_int8,
        "uint8": ctypes.c_uint8,
        "int16": ctypes.c_int16,
        "uint16": ctypes.c_uint16,
        "int32": ctypes.c_int32,
        "uint32": ctypes.c_uint32,
        "int64": ctypes.c_int64,
        "uint64": ctypes.c_uint64,
        "bool": ctypes.c_bool,
    }.get(dtype)

    if ctype is None:
        raise TypeError(f"Unsupported native scalar dtype: {dtype!r}.")

    return ctype(value.item() if hasattr(value, "item") else value)


def native_wrapper(function, abi, tensor_specs, *, device_types, module, stream_attr):
    """Cache numeric FFI arguments behind weak ownership and full value guards."""
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _runtime_wrapper,
        _verified_runtime_launch,
    )

    spec_by_name = {spec.name: spec for spec in tensor_specs}

    def prepare_invocation(values, _static_values, _call_sources):
        bound = []
        device = None

        for binding, value in zip(abi.kernel_args, values):
            if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
                device = (
                    device if device is not None else getattr(value, "device", None)
                )

            bound.append(_native_argument(binding, value, spec_by_name))

        import torch

        runtime = getattr(torch, module, None)

        if runtime is None:
            raise RuntimeError(f"The native backend requires torch.{module}.")

        def launch():
            stream = getattr(runtime.current_stream(device), stream_attr)
            result = function(*bound, ctypes.c_void_p(stream))

            if result != 0:
                raise KernelLaunchError(result)

            return None

        def invoke(_values, _args, _kwargs):
            current_device = getattr(runtime, "current_device", None)
            index = getattr(device, "index", None)

            if (
                current_device is not None
                and index is not None
                and current_device() != index
            ):
                with runtime.device(device):
                    return launch()

            return launch()

        invoke.requires_values = False

        return invoke

    def fallback(*values):
        return prepare_invocation(values, None, None)((), (), {})

    launch = _runtime_wrapper(
        fallback,
        abi,
        specs=tensor_specs,
        device_types=device_types,
        prepare_invocation=prepare_invocation,
    )

    return _verified_runtime_launch(launch)
