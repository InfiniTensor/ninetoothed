"""Generate the public C++ ABI for standalone native kernel exports."""

import math
import re
from pathlib import Path

from ninetoothed.backends.emitters.expressions import normalize_dtype
from ninetoothed.compiler.cache import atomic_write_text

_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)


def write_header(output_dir):
    """Write the shared historical tensor, stream, and configuration ABI."""
    types = ",\n    ".join(f"NINETOOTHED_{name.upper()}" for name in _DTYPES)
    path = Path(output_dir) / "ninetoothed.h"
    atomic_write_text(
        path,
        f"""#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

#define NINETOOTHED_TRUE 1
#define NINETOOTHED_FALSE 0
#define NINETOOTHED_NONE 0

enum NineToothedDataType {{
    {types}
}};

typedef struct {{
    void *data;
    uint64_t *shape;
    int64_t *strides;
}} NineToothedTensor;

typedef void *NineToothedStream;
typedef int NineToothedResult;

#endif
""",
    )

    return path


def _identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"C++ export requires a valid kernel identifier: {name!r}.")

    return name


def _signature(name, public_args, config_types=(), *, prefix="launch_"):
    parameters = ["NineToothedStream stream"]
    parameters.extend(
        f"NineToothedTensor tensor_{index}" for index, _name in enumerate(public_args)
    )
    parameters.extend(
        f"{type_} config_{index}" for index, type_ in enumerate(config_types)
    )

    return f"NineToothedResult {prefix}{_identifier(name)}({', '.join(parameters)})"


def _scalar_type(dtype):
    if dtype is None:
        raise TypeError("C++ export requires statically known scalar dtypes.")

    dtype = normalize_dtype(dtype)

    if dtype in {"float16", "bfloat16", "float32", "float64"}:
        return "double"

    if dtype == "bool":
        return "int8_t"

    if dtype in _DTYPES[:8]:
        return f"{dtype}_t"

    raise TypeError(f"C++ export does not support scalar dtype {dtype!r}.")


def wrapper_source(compilation, enter_name, leave_name):
    """Adapt the public tensor ABI to one guarded Triton AOT launcher."""
    abi = compilation.launch_abi
    names = {name: f"tensor_{index}" for index, name in enumerate(abi.public_args)}
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    kernel_name = _identifier(compilation.artifact.kernel_name)
    enter_name = _identifier(enter_name)
    leave_name = _identifier(leave_name)
    checks = []
    empty = []
    pointer_checks = []

    for name, public in names.items():
        spec = specs[name]
        shape = spec.attrs.get("source_shape", spec.shape)
        rank = int(spec.attrs.get("source_ndim", len(shape)))

        if getattr(spec, "jagged_dim", None) is not None:
            raise ValueError("C++ export does not support jagged tensor bindings.")

        if rank:
            checks.append(f"{public}.shape == nullptr || {public}.strides == nullptr")

        for dim in range(rank):
            checks.append(f"{public}.shape[{dim}] > INT64_MAX")

            if dim < len(shape) and str(shape[dim]).isdigit():
                checks.append(f"{public}.shape[{dim}] != {shape[dim]}ULL")

            if name in abi.outputs or not abi.outputs:
                empty.append(f"{public}.shape[{dim}] == 0")

        pointer_checks.append(f"{public}.data == nullptr")

    types = ["CUstream"]
    values = ["static_cast<CUstream>(stream)"]

    for binding in abi.kernel_args:
        public = names.get(binding.source)

        if binding.kind in {"constexpr", "meta"}:
            if binding.value is None:
                raise ValueError(f"C++ export requires a value for `{binding.name}`.")

            if public is not None:
                type_ = _scalar_type(specs[binding.source].dtype)
                checks.append(
                    f"{public}.data == nullptr || "
                    f"*static_cast<const {type_} *>({public}.data) != "
                    f"{_literal(binding.value)}"
                )

            continue

        if public is None:
            raise ValueError(
                f"C++ export cannot bind `{binding.name}` to a public tensor."
            )

        if binding.kind == "tensor":
            type_, value = (
                "CUdeviceptr",
                f"reinterpret_cast<CUdeviceptr>({public}.data)",
            )
        elif binding.kind == "scalar":
            type_ = _scalar_type(specs[binding.source].dtype)
            value = f"*static_cast<const {type_} *>({public}.data)"
        elif binding.kind in {"shape", "stride"}:
            member = "shape" if binding.kind == "shape" else "strides"
            type_, value = (
                "int64_t",
                f"static_cast<int64_t>({public}.{member}[{binding.dim}])",
            )
        else:
            raise ValueError(
                f"C++ export does not support binding kind `{binding.kind}`."
            )

        types.append(type_)
        values.append(value)

    body = [f"    if ({check}) return CUDA_ERROR_INVALID_VALUE;" for check in checks]

    if empty:
        body.append(f"    if ({' || '.join(empty)}) return CUDA_SUCCESS;")

    if pointer_checks:
        body.append(
            f"    if ({' || '.join(pointer_checks)}) return CUDA_ERROR_INVALID_VALUE;"
        )

    body.extend(
        (
            f"    CUresult result = {enter_name}();",
            "    if (result != CUDA_SUCCESS) return result;",
            f"    result = {kernel_name}_kernel_default({', '.join(values)});",
            f"    {leave_name}();",
            "    return result;",
        )
    )
    signature = _signature(f"{kernel_name}_variant", abi.public_args)

    return (
        '#include "ninetoothed.h"\n#include <cuda.h>\n#include <stdint.h>\n\n'
        f'extern "C" CUresult {kernel_name}_kernel_default({", ".join(types)});\n'
        f'extern "C" CUresult {enter_name}(void);\n'
        f'extern "C" void {leave_name}(void);\n\n'
        f'extern "C" {signature} {{\n' + "\n".join(body) + "\n}\n"
    )


def _configuration_value(value):
    if isinstance(value, str):
        dtype = normalize_dtype(value)
        dtype = re.sub(
            r"^([iu])(8|16|32|64)$",
            lambda match: ("int" if match[1] == "i" else "uint") + match[2],
            dtype,
        )

        if dtype in _DTYPES:
            return _DTYPES.index(dtype)

    if value is None:
        return 0

    if isinstance(value, int) and not -(1 << 31) <= value < (1 << 31):
        raise ValueError("C++ configuration integers must fit the historical int ABI.")

    if isinstance(value, (int, float)):
        return value

    raise TypeError(f"C++ export does not support configuration value {value!r}.")


def _literal(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("C++ export requires finite configuration constants.")

        return repr(value)

    return str(int(value))


def supports_cpp_export(variants):
    """Return whether all variants fit the standalone C++ ABI without writing files."""
    variants = tuple(variants)

    if not variants:
        return False

    public_args = variants[0][1].launch_abi.public_args

    try:
        for key, compilation, wrapper_name in variants:
            if compilation.launch_abi.public_args != public_args:
                return False

            _identifier(wrapper_name)

            for value in key:
                _literal(_configuration_value(value))

            name = compilation.artifact.kernel_name
            wrapper_source(compilation, f"{name}_triton_enter", f"{name}_triton_leave")
    except (TypeError, ValueError):
        return False

    return True


def write_dispatcher(kernel_name, runtime_names, variants, output_dir):
    """Export a dispatcher using the first candidate for each configuration key.

    Python retains runtime auto-tuning; the native dispatcher selects candidates
    deterministically in their build configuration order.
    """
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
    seen = set()
    arguments = ", ".join(
        ("stream", *(f"tensor_{i}" for i, _ in enumerate(public_args)))
    )

    for key, (_, compilation, wrapper_name) in zip(configurations, variants):
        if compilation.launch_abi.public_args != public_args:
            raise ValueError("C++ build variants must share the same public arguments.")

        if key in seen:
            continue

        seen.add(key)
        declarations.append(
            f'extern "C" {_signature(wrapper_name, public_args, prefix="")};'
        )
        condition = (
            " && ".join(
                f"config_{index} == {_literal(value)}"
                for index, value in enumerate(key)
            )
            or "true"
        )
        branches.append(f"    if ({condition}) return {wrapper_name}({arguments});")

    header = (
        f"#ifndef NINETOOTHED_{kernel_name.upper()}_H\n"
        f"#define NINETOOTHED_{kernel_name.upper()}_H\n\n"
        '#include "ninetoothed.h"\n\n#ifdef __cplusplus\nextern "C"\n#endif\n'
        f"{signature};\n\n#endif\n"
    )
    source = (
        f'#include "{kernel_name}.h"\n#include <cuda.h>\n\n'
        + "\n".join(declarations)
        + f'\n\nextern "C" {signature} {{\n'
        + "\n".join(branches)
        + "\n    return CUDA_ERROR_INVALID_VALUE;\n}\n"
    )
    directory = Path(output_dir)
    write_header(directory)
    atomic_write_text(directory / f"{kernel_name}.h", header)
    atomic_write_text(directory / f"{kernel_name}.cpp", source)

    return directory / f"{kernel_name}.cpp"
