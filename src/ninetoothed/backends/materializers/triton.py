"""Triton artifact materialization."""

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.emitters.expressions import replace_symbols
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    artifact_directory,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)


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

        library = ctypes.CDLL(built.binary_path)
        function = getattr(library, f"{built.source.kernel_name}_kernel_default")
        function.restype = ctypes.c_int
        specs = _runtime_specs(built.source)
        return _aot_wrapper(function, _launch_abi_from_dict(built.abi), specs)


def _materialize(compilation):
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        import_python_module,
    )

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
    wrapped = _runtime_wrapper(
        launch,
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
    )
    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    return Handle(compilation, kernel, wrapped, source)


def _aot_materialize(compilation, *, output_dir):
    from ninetoothed.compiler.runtime import Handle, _built_manifest, _publish_library

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
        if not cache_library.is_file():
            _compile_aot_library(compilation, source, cache_library)
        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _built_manifest(compilation, cache_key, source, cache_library),
        )
    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.triton.so",
    )
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, f"{artifact.kernel_name}_kernel_default")
    function.restype = ctypes.c_int
    wrapped = _aot_wrapper(function, compilation.launch_abi, compilation.kernel.tensors)
    return Handle(compilation, function, wrapped, source, library_path)


def _compile_aot_library(compilation, source: Path, library: Path) -> None:
    from ninetoothed.backends.materializers.cuda import _nvcc

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
                _nvcc(),
                "-shared",
                "-Xcompiler",
                "-fPIC",
                "-O3",
                "-lcuda",
                *(str(path) for path in sources),
                str(linked.with_suffix(".c")),
                "-o",
                str(output),
            ],
            check=True,
        )
        os.replace(output, library)


def _compile_signature(compilation) -> str:
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    values = []
    for binding in compilation.launch_abi.kernel_args:
        if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}:
            values.append(f"*{_triton_dtype(specs[binding.source].dtype)}")
        elif binding.kind == "scalar":
            values.append(_triton_dtype(specs[binding.source].dtype))
        elif binding.kind in {"constexpr", "meta"}:
            if binding.value is None:
                raise ValueError(f"Triton AOT requires a value for `{binding.name}`.")
            values.append(str(binding.value))
        else:
            values.append("i64")
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


def _compile_block(compilation) -> int:
    mode = dict(compilation.artifact.metadata.get("program_mode", {}))
    if mode.get("block") or mode.get("scalar"):
        return 1
    if mode.get("vector"):
        total = _constant_grid_total(compilation)
        return 1 << max(0, (total - 1).bit_length())
    return 256


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
    expression = str(compilation.artifact.metadata.get("launch_grid", ("1",))[0])
    replacements = {
        binding.name: str(binding.value)
        for binding in compilation.launch_abi.kernel_args
        if binding.kind in {"meta", "constexpr"} and binding.value is not None
    }
    return replace_symbols(expression, replacements)


def _compile_schedule(compilation) -> tuple[int, int]:
    schedule = dict(compilation.artifact.metadata.get("ssa_schedule", {}))
    warps = compilation.request.num_warps or schedule.get("num_warps") or 4
    stages = compilation.request.num_stages or schedule.get("num_stages") or 3
    if isinstance(warps, tuple):
        warps = warps[0]
    if isinstance(stages, tuple):
        stages = stages[0]
    return int(warps), int(stages)


def _aot_wrapper(function, abi, tensor_specs):
    from ninetoothed.backends.materializers.cuda import _cuda_scalar
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _bound_values,
        _empty_launch,
        _first_output,
        _public_values,
    )

    specs = {spec.name: spec for spec in tensor_specs}

    def launch(*args, **kwargs):
        import torch

        public = _public_values(abi, args, kwargs, specs=tensor_specs)
        if _empty_launch(abi, public):
            return _first_output(abi, public)
        values, keepalive = _bound_values(
            abi,
            public,
            scalar_mode="cuda",
            specs=specs,
            cuda_scalar=_cuda_scalar,
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        result = function(stream, *values)
        del keepalive
        if result != 0:
            raise KernelLaunchError(result)
        return _first_output(abi, public)

    return launch


__all__ = ["TritonMaterializer"]
