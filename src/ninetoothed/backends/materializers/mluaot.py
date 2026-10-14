"""MLU ahead-of-time compilation for Triton kernels.

The Cambricon Triton distribution cannot run ``triton.tools.compile``:
it imports ``triton.backends.nvidia``, which the distribution does not
ship, and its launcher templates are absent. MLU AOT builds therefore
compile the kernel in-process through ``triton.compile`` and render a
self-contained CNDRV launcher that loads the resulting ``cnbin`` and
launches it via ``cnInvokeKernel``. The rendered source is linked
against the Neuware runtime libraries.
"""

import binascii
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_LAUNCHER_TEMPLATE = """#include "cn_api.h"
#include "cnrt.h"
#include <cstdio>
#include <mutex>
#include <cstdint>

static const unsigned char kernel_bin[] = {{{bin_data}}};

static CNmodule kernel_mod = nullptr;
static CNkernel kernel_fun = nullptr;
static std::once_flag load_once;

static void load_kernel() {{
  CNcontext ctx = nullptr;
  int device = 0;

  if (cnrtGetDevice(&device) != cnrtSuccess) {{
    return;
  }}

  if (cnSharedContextAcquire(&ctx, device) != CN_SUCCESS) {{
    return;
  }}

  cnCtxSetCurrent(ctx);

  if (cnModuleLoadFatBinary((const void *)kernel_bin, &kernel_mod) !=
      CN_SUCCESS) {{
    return;
  }}

  cnModuleGetKernel(kernel_mod, "{kernel_function_name}", &kernel_fun);
}}

extern "C" int ninetoothed_triton_enter() {{
  std::call_once(load_once, load_kernel);
  return kernel_fun == nullptr ? 1 : 0;
}}

extern "C" void ninetoothed_triton_leave() {{}}

extern "C" int ninetoothed_triton_is_mlu() {{
  return 1;
}}

extern "C" int {entry_name}(CNqueue stream{param_decls}) {{
  ninetoothed_triton_enter();

  if (kernel_fun == nullptr) {{
    return 1;
  }}

  CNaddr global_scratch = 0;
  void *params[] = {{ {params_list} }};

  unsigned int grid_x = {grid_x};
  unsigned int grid_y = {grid_y};
  unsigned int grid_z = {grid_z};

  // Cambricon's JIT launcher scales gridX by num_warps before launching.
  grid_x *= {num_warps};

  if (grid_x * grid_y * grid_z <= 0) {{
    return 0;
  }}

  CNresult status = cnInvokeKernel(
      kernel_fun, grid_x, grid_y, grid_z, (KernelClass){func_type}, 0,
      stream, params, NULL);

  return status == CN_SUCCESS ? 0 : (int)status;
}}
"""


def mlu_available() -> bool:
    """Return whether the MLU runtime is importable in this environment."""
    import importlib.util

    return importlib.util.find_spec("torch_mlu") is not None


def mlu_requested(compilation) -> bool:
    """Return whether this compilation should build the MLU AOT library.

    Device types are configuration-driven and may still hold the generic
    CUDA default when NineToothed runs on an MLU-only host, so an explicit
    MLU listing wins; otherwise the absence of a usable CUDA runtime
    decides.
    """
    if not mlu_available():
        return False

    if "mlu" in _compilation_device_types(compilation):
        return True

    try:
        import torch
    except ImportError:
        return False

    return not torch.cuda.is_available()


def _compilation_device_types(compilation) -> tuple[str, ...]:
    """Read the resolved device types from the compilation target."""
    try:
        from ninetoothed.targets import target_device_types

        return tuple(
            str(device_type).lower()
            for device_type in target_device_types(compilation)
        )
    except Exception:
        return ()


def compile_mlu_aot(compilation, source: Path, library: Path) -> None:
    """Compile `source` for MLU and link it into the `library` shared object."""
    import triton
    from triton.backends.mlu.driver import ty_to_cpp
    from triton.compiler import ASTSource

    from .triton import (
        _compile_grid,
        _compile_schedule,
        _compile_signature,
        _triton_dtype,
    )

    artifact = compilation.artifact
    kernel_name = f"{artifact.kernel_name}_kernel"
    signature_text = _compile_signature(compilation)
    grid_x, grid_y, grid_z = _compile_grid(compilation).split(",")
    num_warps, num_stages = _compile_schedule(compilation)

    # The rendered grid is the uncapped logical program count; clamp it to
    # the platform grid limit before the launcher's num_warps scaling, the
    # same way the JIT emitter clamps its launch grid. The kernel body is a
    # grid-stride loop, so the capped launch still covers every program.
    platform_metadata = getattr(
        getattr(compilation.target, "platform", None), "metadata", {}
    )
    grid_limit = int(platform_metadata.get("triton_grid_limit", 65535))
    launch_cap = f"{grid_limit // num_warps}u"

    sys_path = list(sys.path)
    source_parent = str(Path(source).parent)
    sys.path.insert(0, source_parent)
    try:
        spec = importlib.util.spec_from_file_location(
            Path(source).stem, Path(source)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(source_parent)

    kernel = getattr(module, kernel_name)
    arg_names = list(kernel.arg_names)

    parts = [part.strip(" ") for part in signature_text.split(",")]

    if len(parts) != len(arg_names):
        raise ValueError(
            f"Triton AOT signature has {len(parts)} entries for "
            f"{len(arg_names)} kernel arguments."
        )

    # The NV AOT signature text widens every float scalar to fp64, but MLU
    # slots follow the declaring dtype, with fp64 narrowed to fp32 (the MLU
    # backend rejects fp64 in binary operations).
    scalar_overrides = {}
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    for binding in compilation.launch_abi.kernel_args:
        if binding.kind != "scalar":
            continue

        dtype_name = _triton_dtype(specs[binding.source].dtype)
        scalar_overrides[binding.name] = (
            "fp32" if dtype_name == "fp64" else dtype_name
        )

    signature = {}
    constants = {}
    for name, part in zip(arg_names, parts):
        base = part.split(":")[0]

        try:
            value = int(base)
        except ValueError:
            value = None

        if value is None:
            if name in scalar_overrides:
                base = scalar_overrides[name]

            signature[name] = base
        else:
            constants[name] = value
            signature[name] = "constexpr"

    srcobj = ASTSource(
        fn=kernel, signature=signature, constexprs=constants
    )
    ccinfo = triton.compile(
        srcobj, options={"num_warps": num_warps, "num_stages": num_stages}
    )

    binary = ccinfo.asm["cnbin"]
    hex_text = str(binascii.hexlify(binary))[2:-1]
    bin_data = ", ".join(
        f"0x{high}{low}" for high, low in zip(hex_text[::2], hex_text[1::2])
    )

    metadata = ccinfo.metadata
    kernel_function_name = metadata.name

    try:
        func_type = int(getattr(metadata, "func_type", 0))
    except (TypeError, ValueError):
        func_type = 0

    runtime_args = [
        (name, signature[name])
        for name in arg_names
        if signature[name] != "constexpr"
    ]

    # Float scalars cross the MLU device ABI as raw IEEE-754 bit patterns in
    # unsigned slots (matching the MLU driver's own launcher tables);
    # pointers and integers keep their ty_to_cpp declarations.
    bit_slot_types = {
        "fp16": "uint16_t",
        "bf16": "uint16_t",
        "fp32": "uint32_t",
        "f32": "uint32_t",
        "fp64": "uint64_t",
    }

    param_decls = "".join(
        f", {bit_slot_types.get(ty, ty_to_cpp(ty))} {name}"
        for name, ty in runtime_args
    )
    params_list = ", ".join(
        [f"&{name}" for name, _ in runtime_args] + ["&global_scratch"]
    )
    entry_name = f"{artifact.kernel_name}_kernel_default"

    rendered = _LAUNCHER_TEMPLATE.format(
        bin_data=bin_data,
        kernel_function_name=kernel_function_name,
        entry_name=entry_name,
        param_decls=param_decls,
        params_list=params_list,
        grid_x=f"((({grid_x}) < {launch_cap}) ? ({grid_x}) : {launch_cap})",
        grid_y=f"((({grid_y}) < {launch_cap}) ? ({grid_y}) : {launch_cap})",
        grid_z=f"((({grid_z}) < {launch_cap}) ? ({grid_z}) : {launch_cap})",
        num_warps=num_warps,
        func_type=func_type,
    )

    library.parent.mkdir(parents=True, exist_ok=True)
    c_path = library.with_suffix(".mlu_aot.c")
    c_path.write_text(rendered, encoding="utf-8")

    command = [
        _select_host_compiler(),
        "-shared",
        "-fPIC",
        "-O3",
        f"-I{_neuware_home()}/include",
        str(c_path),
        f"-L{_neuware_home()}/lib64",
        "-lcndrv",
        "-lcnrt",
        "-lcnrtc",
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)


def _neuware_home() -> str:
    return os.environ.get("NEUWARE_HOME") or "/usr/local/neuware"


def _select_host_compiler() -> str:
    explicit = os.environ.get("NINETOOTHED_CUDA_COMPILER")

    if explicit:
        return explicit

    try:
        from ninetoothed.backends.toolchain import find_nvcc

        return find_nvcc()
    except RuntimeError:
        pass

    for candidate in ("/usr/bin/g++", "/usr/bin/gcc"):
        if Path(candidate).is_file():
            return candidate

    return "g++"
