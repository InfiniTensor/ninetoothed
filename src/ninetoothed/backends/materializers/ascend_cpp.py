"""Generate standalone C++ launchers for Ascend Triton AOT artifacts."""

from pathlib import Path

from ninetoothed.backends.emitters.expressions import replace_symbols
from ninetoothed.backends.materializers.cpp import (
    _configuration_value,
    _identifier,
    _literal,
    _scalar_type,
    _signature,
    write_header,
)
from ninetoothed.compiler.cache import atomic_write_text

_TYPE_NAMES = {
    "i1": ("int32_t", 4),
    "i8": ("int8_t", 1),
    "i16": ("int16_t", 2),
    "i32": ("int32_t", 4),
    "i64": ("int64_t", 8),
    "u1": ("uint8_t", 1),
    "u8": ("uint8_t", 1),
    "u16": ("uint16_t", 2),
    "u32": ("uint32_t", 4),
    "u64": ("uint64_t", 8),
    "fp16": ("float", 4),
    "bf16": ("float", 4),
    "fp32": ("float", 4),
    "fp64": ("double", 8),
}


def _cpp_type(dtype):
    if dtype.startswith("*"):
        return "void *", 8

    try:
        return _TYPE_NAMES[dtype]
    except KeyError as exc:
        raise TypeError(f"Ascend C++ export does not support dtype {dtype!r}.") from exc


def _public_refs(compilation):
    return {
        name: f"tensor_{index}"
        for index, name in enumerate(compilation.launch_abi.public_args)
    }


def _binding_values(compilation, signature):
    refs = _public_refs(compilation)
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    values = {}
    replacements = {}

    for binding in compilation.launch_abi.kernel_args:
        public = refs.get(binding.source)

        if binding.kind in {"constexpr", "meta"}:
            continue

        if public is None:
            raise ValueError(
                f"Ascend C++ export cannot bind {binding.name!r} to a public tensor."
            )

        if binding.kind == "tensor":
            value = f"{public}.data"
        elif binding.kind == "scalar":
            scalar = _scalar_type(specs[binding.source].dtype)
            target, _ = _cpp_type(signature[binding.name])
            value = (
                f"static_cast<{target}>(*static_cast<const {scalar} *>({public}.data))"
            )
        elif binding.kind in {"shape", "stride"}:
            member = "shape" if binding.kind == "shape" else "strides"
            value = f"static_cast<int64_t>({public}.{member}[{binding.dim}])"
        else:
            raise ValueError(
                f"Ascend C++ export does not support binding kind {binding.kind!r}."
            )

        values[binding.name] = value
        replacements[binding.name] = value

    return values, replacements


def _grid_expression(compilation, replacements):
    from ninetoothed.backends.materializers.triton import _compile_grid

    return replace_symbols(_compile_grid(compilation).split(",", 1)[0], replacements)


def _launcher_source(compilation, compiled, signature, binary):
    name = _identifier(compilation.artifact.kernel_name)
    metadata = compiled.metadata

    if int(getattr(metadata, "workspace_size", 0) or 0) != 0:
        raise ValueError("Ascend C++ export does not support workspace kernels yet.")

    if int(getattr(metadata, "lock_num", 0) or 0) != 0:
        raise ValueError("Ascend C++ export does not support lock kernels yet.")

    if getattr(metadata, "force_simt_only", False):
        raise ValueError("Ascend C++ export does not support SIMT-only kernels yet.")

    values, replacements = _binding_values(compilation, signature)
    grid = _grid_expression(compilation, replacements)
    fields = [
        "    void *ffts_addr __attribute__((aligned(8)));",
        "    void *sync_block_lock __attribute__((aligned(8)));",
        "    void *workspace __attribute__((aligned(8)));",
    ]
    initializers = [
        "        reinterpret_cast<void *>(ffts_address),",
        "        nullptr,",
        "        nullptr,",
    ]

    for binding_name, dtype in signature.items():
        target, alignment = _cpp_type(dtype)
        fields.append(
            f"    {target} {binding_name} __attribute__((aligned({alignment})));"
        )
        initializers.append(f"        static_cast<{target}>({values[binding_name]}),")

    fields.extend(
        (
            "    int32_t grid_x __attribute__((aligned(4)));",
            "    int32_t grid_y __attribute__((aligned(4)));",
            "    int32_t grid_z __attribute__((aligned(4)));",
        )
    )
    initializers.extend(
        (f"        static_cast<int32_t>({grid}),", "        1,", "        1,")
    )
    data = "".join(f"\\x{byte:02x}" for byte in binary)
    kernel_name, mix_mode = str(compiled.name).split()
    symbol = _identifier(kernel_name)
    magic = (
        "RT_DEV_BINARY_MAGIC_ELF_AIVEC"
        if mix_mode == "aiv"
        else "RT_DEV_BINARY_MAGIC_ELF"
    )
    public_args = compilation.launch_abi.public_args
    public_signature = _signature(f"{name}_variant", public_args)

    return f"""#include "ninetoothed.h"

#include "runtime/runtime/rt.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {{
const unsigned char k_{name}_binary[] = "{data}";

struct {name}_State {{
  void *function = nullptr;
  std::unique_ptr<size_t> stub;
}};

std::mutex {name}_mutex;
std::unordered_map<int32_t, {name}_State> {name}_states;

void *{name}_function() {{
  int32_t device = 0;

  if (rtGetDevice(&device) != RT_ERROR_NONE) {{
    return nullptr;
  }}

  std::lock_guard<std::mutex> lock({name}_mutex);
  auto found = {name}_states.find(device);

  if (found != {name}_states.end()) {{
    return found->second.function;
  }}

  if (rtSetDevice(device) != RT_ERROR_NONE) {{
    return nullptr;
  }}

  rtDevBinary_t binary = {{}};
  binary.data = k_{name}_binary;
  binary.length = sizeof(k_{name}_binary);
  binary.magic = {magic};
  binary.version = 0;
  void *module = nullptr;

  if (rtDevBinaryRegister(&binary, &module) != RT_ERROR_NONE) {{
    return nullptr;
  }}

  auto state = {name}_State{{}};
  state.stub = std::make_unique<size_t>(0);
  std::string stub_name = "{symbol}_" + std::to_string(device);

  if (rtFunctionRegister(
          module,
          state.stub.get(),
          stub_name.c_str(),
          const_cast<void *>(reinterpret_cast<const void *>("{symbol}")),
          0
      ) != RT_ERROR_NONE) {{
    return nullptr;
  }}

  state.function = state.stub.get();
  auto inserted = {name}_states.emplace(device, std::move(state));
  return inserted.first->second.function;
}}
}}  // namespace

extern "C" {public_signature} {{
  uint64_t ffts_address = 0;
  uint32_t ffts_length = 0;

  if (rtGetC2cCtrlAddr(&ffts_address, &ffts_length) != RT_ERROR_NONE) {{
    return 1;
  }}

  void *function = {name}_function();

  if (function == nullptr) {{
    return 1;
  }}

  struct __attribute__((packed)) {{
{chr(10).join(fields)}
  }} args = {{
{chr(10).join(initializers)}
  }};
  uint32_t blocks = static_cast<uint32_t>(args.grid_x) * args.grid_y * args.grid_z;

  return rtKernelLaunch(
      function,
      blocks,
      &args,
      sizeof(args),
      nullptr,
      static_cast<rtStream_t>(stream)
  );
}}
"""


def write_launcher(compilation, compiled, signature, binary_path, output_dir):
    """Write one embedded-binary launcher for a concrete Ascend build."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_header(output)
    source = _launcher_source(
        compilation,
        compiled,
        signature,
        Path(binary_path).read_bytes(),
    )
    path = output / f"{compilation.artifact.kernel_name}.ascend.launch.cpp"
    atomic_write_text(path, source)

    return path


def supports_cpp_export(variants):
    """Return whether all Ascend variants share the historical public ABI."""
    variants = tuple(variants)

    if not variants:
        return False

    public_args = variants[0][1].launch_abi.public_args

    return all(
        compilation.launch_abi.public_args == public_args
        for _, compilation, _ in variants
    )


def write_dispatcher(kernel_name, runtime_names, variants, output_dir):
    """Write the historical configuration dispatcher for Ascend variants."""
    variants = tuple(variants)
    public_args = variants[0][1].launch_abi.public_args
    configurations = tuple(
        tuple(_configuration_value(value) for value in key) for key, _, _ in variants
    )
    types = tuple(
        "double"
        if any(isinstance(key[index], float) for key in configurations)
        else "int"
        for index, _name in enumerate(runtime_names)
    )
    signature = _signature(kernel_name, public_args, types)
    declarations = []
    branches = []
    arguments = ", ".join(
        ("stream", *(f"tensor_{index}" for index in range(len(public_args))))
    )

    for key, (_, compilation, wrapper_name) in zip(configurations, variants):
        del compilation
        declarations.append(
            f'extern "C" {_signature(wrapper_name, public_args, prefix="")};'
        )
        condition = " && ".join(
            f"config_{index} == {_literal(value)}" for index, value in enumerate(key)
        )
        branches.append(f"    if ({condition}) return {wrapper_name}({arguments});")

    header = (
        f"#ifndef NINETOOTHED_{kernel_name.upper()}_H\n"
        f"#define NINETOOTHED_{kernel_name.upper()}_H\n\n"
        '#include "ninetoothed.h"\n\n#ifdef __cplusplus\nextern "C"\n#endif\n'
        f"{signature};\n\n#endif\n"
    )
    source = (
        f'#include "{kernel_name}.h"\n\n'
        + "\n".join(declarations)
        + f'\n\nextern "C" {signature} {{\n'
        + "\n".join(branches)
        + "\n    return 1;\n}\n"
    )
    output = Path(output_dir)
    write_header(output)
    atomic_write_text(output / f"{kernel_name}.h", header)
    atomic_write_text(output / f"{kernel_name}.cpp", source)

    return output / f"{kernel_name}.cpp"


__all__ = ["supports_cpp_export", "write_dispatcher", "write_launcher"]
