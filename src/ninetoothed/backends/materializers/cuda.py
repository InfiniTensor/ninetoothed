"""CUDA artifact compilation, launch binding, and reload."""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import cuda_compile_command
from ninetoothed.compiler.cache import (
    TOOLCHAIN_LOCK_DIR,
    artifact_directory,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)


class CudaMaterializer(Materializer):
    target = Target.CUDA

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("CUDA built artifact does not contain a binary path.")

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
        )

        library = ctypes.CDLL(built.binary_path)
        function = getattr(library, built.source.entrypoint)
        function.restype = ctypes.c_int
        specs = _runtime_specs(built.source)

        return _cuda_wrapper(function, _launch_abi_from_dict(built.abi), specs)


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.compiler.runtime import (
        Handle,
        _built_manifest,
        _publish_library,
    )

    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "cu",
        cache_key=cache_key,
    )
    cache_library = artifact_directory(cache_key) / f"{artifact.kernel_name}.cuda.so"

    with cache_lock(cache_library):
        if not cache_library.is_file():
            _compile_library(
                source,
                cache_library,
                arch=dict(compilation.request.backend_options or {}).get(
                    "arch", "native"
                ),
            )

        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _built_manifest(compilation, cache_key, source, cache_library),
        )

    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.cuda.so",
    )
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, artifact.entrypoint)
    function.restype = ctypes.c_int
    wrapped = _cuda_wrapper(
        function, compilation.launch_abi, compilation.kernel.tensors
    )

    return Handle(compilation, function, wrapped, source, library_path)


def _cuda_wrapper(function, abi, tensor_specs):
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _bound_values,
        _empty_launch,
        _first_output,
        _public_values,
    )

    spec_by_name = {spec.name: spec for spec in tensor_specs}

    def launch(*args, **kwargs):
        public = _public_values(abi, args, kwargs, specs=tensor_specs)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        import torch

        values, keepalive = _bound_values(
            abi,
            public,
            scalar_mode="cuda",
            specs=spec_by_name,
            cuda_scalar=_cuda_scalar,
        )
        stream = torch.cuda.current_stream().cuda_stream
        result = function(*values, ctypes.c_void_p(stream))
        del keepalive

        if result != 0:
            raise KernelLaunchError(result)
        return _first_output(abi, public)

    return launch


def _cuda_scalar(value, dtype):
    dtype = str(dtype).split(".")[-1]
    dtype = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(dtype, dtype)

    if dtype in {"float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        import torch

        torch_dtype = getattr(torch, dtype)
        storage_dtype = (
            torch.uint16 if dtype in {"float16", "bfloat16"} else torch.uint8
        )
        scalar = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value)
        )
        bits = scalar.to(dtype=torch_dtype).view(storage_dtype).item()

        return (ctypes.c_uint16 if storage_dtype == torch.uint16 else ctypes.c_uint8)(
            bits
        )

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
        raise TypeError(f"Unsupported CUDA scalar dtype: {dtype!r}.")
    return ctype(value.item() if hasattr(value, "item") else value)


def _compile_library(source: Path, library: Path, *, arch: str) -> None:
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
        with cache_lock(TOOLCHAIN_LOCK_DIR / "nvcc"):
            subprocess.run(
                cuda_compile_command(source, temporary, arch=arch),
                check=True,
            )

        os.replace(temporary, library)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["CudaMaterializer"]
