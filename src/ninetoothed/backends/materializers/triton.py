"""Triton artifact materialization."""

import ctypes
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
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


class TritonMaterializer(Materializer):
    target = Target.TRITON

    def jit_materialize(self, compilation, *, output_dir=None):
        del output_dir

        return _materialize(compilation)

    def aot_build(self, compilation, *, output_dir: str | Path):
        if len(compilation.launch_plan.tuning_candidates) > 1:
            raise ValueError(
                "Triton AOT accepts one launch configuration; use build() to "
                "benchmark and package multiple explicit configurations."
            )
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

        return _aot_wrapper(
            function,
            enter,
            leave,
            _launch_abi_from_dict(built.abi),
            specs,
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
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    candidate_launches = tuple(
        _candidate_launch(compilation, launch, candidate) for candidate in candidates
    )
    tuner = None

    if len(candidate_launches) > 1:
        from ninetoothed.auto_tuner import AutoTuner

        tuner = AutoTuner(
            candidate_launches,
            tuple((cache_key, candidate["id"]) for candidate in candidates),
            cache_namespace=f"jit_{cache_key}",
            validator=_runtime_validator(compilation),
        )
        wrapped = tuner
    elif candidate_launches:
        wrapped = candidate_launches[0]
    else:
        wrapped = _runtime_wrapper(
            launch,
            compilation.launch_abi,
            specs=compilation.kernel.tensors,
        )

    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    handle = Handle(compilation, kernel, wrapped, source)
    handle._tuner = tuner
    handle._selected_tuning_candidate = candidates[0] if len(candidates) == 1 else None

    if tuner is not None:
        by_launch = dict(zip(candidate_launches, candidates))

        def tuned_launch(*args, **kwargs):
            result = tuner(*args, **kwargs)
            arg_key = tuner._make_arg_key(args, kwargs)
            handle._selected_tuning_candidate = by_launch[tuner._best_func[arg_key]]

            return result

        handle._launch = tuned_launch

    return handle


def _candidate_launch(compilation, launch, candidate):
    from ninetoothed.compiler.runtime import _runtime_wrapper

    wrapped = _runtime_wrapper(
        functools.partial(
            launch,
            _ninetoothed_num_warps=int(candidate["num_warps"]),
            _ninetoothed_num_stages=int(candidate["num_stages"]),
        ),
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
    )
    bindings = {binding.name: binding for binding in compilation.launch_abi.kernel_args}
    meta_kwargs = {
        str(bindings[name].source): value
        for name, value in dict(candidate.get("meta_parameters", {})).items()
    }

    if not meta_kwargs:
        return wrapped

    def launch_candidate(*args, **kwargs):
        return wrapped(*args, **(dict(kwargs) | meta_kwargs))

    return launch_candidate


def _runtime_validator(compilation):
    from ninetoothed.compiler.runtime import _public_values

    def validate(args, kwargs):
        _public_values(
            compilation.launch_abi,
            args,
            kwargs,
            specs=compilation.kernel.tensors,
        )

    return validate


def _aot_materialize(compilation, *, output_dir):
    from ninetoothed.compiler.runtime import Handle, _publish_library

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
    )
    handle = Handle(compilation, function, wrapped, source, library_path)
    write_manifest(
        handle._built_artifact.manifest_path,
        _triton_aot_manifest(compilation, cache_key, source, library_path),
    )

    return handle


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
    expression = compilation.launch_plan.grid[0].render()
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


def _aot_wrapper(function, enter, leave, abi, tensor_specs):
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
