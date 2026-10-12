"""Generate RVNE operator translation units and link a native model executable."""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from examples.rvne.operators import (
    PreparedWeights,
    RvneOperators,
    _conv2d_application,
    _identity,
    _lif_application,
    _max_pool2d_application,
    _pair,
)
from ninetoothed import Tensor
from ninetoothed.backends.rvne_toolchain import (
    find_rvne_toolchain,
    rvne_compile_command,
    rvne_compiler_identity,
)
from ninetoothed.compiler import CompileRequest, compile_kernel
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ir_to_dict

_C_TYPES = {"int32": "int32_t", "uint32": "uint32_t", "uint64": "uint64_t"}


def generate_native_bundle(config, output_dir):
    """Lower operators to source without compiling or launching individual kernels.

    ``convs`` entries provide ``input_shape``, integer OIHW ``weights`` (or a
    PreparedWeights instance), and optional ``stride``/``padding``. ``pools``
    entries provide NCHW ``input_shape``; ``lifs`` entries are element counts.
    Packed weights are embedded by an assembler translation unit at link time.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kernel_dir = output / "kernels"
    kernel_dir.mkdir(exist_ok=True)
    preparer = RvneOperators(output)
    kernels = []
    wrappers = []
    conv_rows = []
    pool_rows = []
    lif_rows = []
    conv_records = []
    pool_records = []
    lif_records = []
    assembly = [".section .rodata", ".balign 8"]
    weight_declarations = []
    weight_path = output / "weights.bin"

    with weight_path.open("wb") as weight_file:
        for index, spec in enumerate(config.get("convs", ())):
            shape = _shape(spec["input_shape"], rank=4)
            raw_weights = spec["weights"]
            weights = (
                raw_weights
                if isinstance(raw_weights, PreparedWeights)
                else preparer.prepare_weights(raw_weights)
            )
            batch, channels, height, width = shape
            output_channels, weight_channels, kernel_height, kernel_width = (
                weights.shape
            )
            stride = _pair(spec.get("stride", 1), "stride", minimum=1)
            padding = _pair(spec.get("padding", 0), "padding", minimum=0)

            if channels != weight_channels:
                raise ValueError("Native RVNE convolution channels must match.")

            output_height = (height + 2 * padding[0] - kernel_height) // stride[0] + 1
            output_width = (width + 2 * padding[1] - kernel_width) // stride[1] + 1

            if output_height <= 0 or output_width <= 0:
                raise ValueError("Native RVNE convolution requires nonempty output.")

            blocks = (channels * kernel_height * kernel_width + 31) // 32
            positions = batch * output_height * output_width
            shapes = (
                (blocks, positions),
                (blocks, output_channels),
                (blocks, output_channels),
                (output_channels, positions),
            )
            record, wrapper = _generate_kernel(
                _conv2d_application,
                "conv",
                index,
                shapes,
                ("uint32", "uint64", "uint64", "int32"),
                kernel_dir,
            )
            kernels.append(record)
            wrappers.append(wrapper)

            for suffix, values in (("low", weights.low), ("high", weights.high)):
                symbol = f"nt_conv_{index}_weight_{suffix}"
                offset = weight_file.tell()
                data = values.astype("<u8", copy=False).tobytes(order="C")
                weight_file.write(data)
                weight_declarations.append(f'extern "C" const uint64_t {symbol}[];')
                assembly.extend(
                    (
                        ".balign 8",
                        f".globl {symbol}",
                        f".type {symbol}, @object",
                        f"{symbol}:",
                        f'.incbin "weights.bin", {offset}, {len(data)}',
                        f".size {symbol}, . - {symbol}",
                    )
                )

            dimensions = (
                *shape,
                output_channels,
                kernel_height,
                kernel_width,
                *stride,
                *padding,
            )
            conv_rows.append(
                "{"
                + ", ".join(str(value) for value in dimensions)
                + f", nt_conv_wrapper_{index}, nt_conv_{index}_weight_low, "
                + f"nt_conv_{index}_weight_high"
                + "}"
            )
            conv_records.append(
                {
                    "input_shape": list(shape),
                    "weight_shape": list(weights.shape),
                    "output_shape": [
                        batch,
                        output_channels,
                        output_height,
                        output_width,
                    ],
                    "stride": list(stride),
                    "padding": list(padding),
                    "packed_weight_bytes": weights.packed_nbytes,
                }
            )

    for index, spec in enumerate(config.get("pools", ())):
        shape = _shape(spec["input_shape"], rank=4)
        batch, channels, height, width = shape
        output_height, output_width = height // 2, width // 2

        if output_height == 0 or output_width == 0:
            raise ValueError("Native RVNE max pool requires spatial dimensions >= 2.")

        count = batch * channels * output_height * output_width
        record, wrapper = _generate_kernel(
            _max_pool2d_application,
            "pool",
            index,
            ((4, count), (count,)),
            ("int32", "int32"),
            kernel_dir,
        )
        kernels.append(record)
        wrappers.append(wrapper)
        pool_rows.append(
            "{"
            + ", ".join(str(value) for value in shape)
            + f", nt_pool_wrapper_{index}"
            + "}"
        )
        pool_records.append(
            {
                "input_shape": list(shape),
                "output_shape": [batch, channels, output_height, output_width],
            }
        )

    for index, value in enumerate(config.get("lifs", ())):
        count = _shape((value,), rank=1)[0]
        record, wrapper = _generate_kernel(
            _lif_application,
            "lif",
            index,
            ((count,),) * 6,
            ("int32",) * 6,
            kernel_dir,
        )
        kernels.append(record)
        wrappers.append(wrapper)
        lif_rows.append(f"{{{count}, nt_lif_wrapper_{index}}}")
        lif_records.append({"count": count})

    registry = output / "registry.cpp"
    registry.write_text(
        '#include "native_runtime.h"\n\n'
        + "\n".join(weight_declarations)
        + "\n\n"
        + "\n\n".join(wrappers)
        + "\n\n"
        + _registry_array("Conv", "conv", conv_rows)
        + _registry_array("Pool", "pool", pool_rows)
        + _registry_array("Lif", "lif", lif_rows),
        encoding="utf-8",
    )
    assembly.append('.section .note.GNU-stack,"",@progbits')
    assembly_path = output / "weights.S"
    assembly_path.write_text("\n".join(assembly) + "\n", encoding="utf-8")
    manifest = {
        "schema": 1,
        "status": "generated",
        "output_dir": str(output),
        "registry_path": str(registry),
        "assembly_path": str(assembly_path),
        "weights_path": str(weight_path),
        "weights_sha256": _sha256(weight_path),
        "sources": [record["source_path"] for record in kernels] + [str(registry)],
        "kernels": kernels,
        "operator_counts": {
            "conv": len(conv_rows),
            "pool": len(pool_rows),
            "lif": len(lif_rows),
        },
        "convs": conv_records,
        "pools": pool_records,
        "lifs": lif_records,
        "manifest_path": str(output / "native_manifest.json"),
    }
    _write_manifest(manifest)

    return manifest


def compile_native_bundle(
    config,
    output_dir,
    extra_sources=(),
    *,
    toolchain_root=None,
    binary_name="rvne_model.elf",
    extra_flags=(),
):
    """Link generated kernels, embedded weights, native runtime, and caller code."""
    if not binary_name or Path(binary_name).name != binary_name:
        raise ValueError("Native RVNE binary_name must be a filename.")

    manifest = generate_native_bundle(config, output_dir)
    output = Path(manifest["output_dir"])
    runtime_dir = Path(__file__).resolve().parent
    header = runtime_dir / "native_runtime.h"
    runtime = runtime_dir / "native_runtime.cpp"
    extras = tuple(Path(value).resolve() for value in extra_sources)

    for source in (header, runtime, *extras):
        if not source.is_file():
            raise FileNotFoundError(f"Native RVNE source is missing: `{source}`.")

    published_header = output / "native_runtime.h"
    published_runtime = output / "native_runtime.cpp"

    if header != published_header:
        shutil.copy2(header, published_header)

    if runtime != published_runtime:
        shutil.copy2(runtime, published_runtime)

    sources = [
        *manifest["sources"],
        str(published_runtime),
        *(str(path) for path in extras),
    ]
    toolchain = find_rvne_toolchain(toolchain_root)
    binary = output / binary_name
    c_sources = tuple(Path(source) for source in sources if Path(source).suffix == ".c")
    cpp_sources = [source for source in sources if Path(source).suffix != ".c"]
    c_dependencies = set()

    with tempfile.TemporaryDirectory(prefix="native-link-", dir=output) as temporary:
        temporary_binary = Path(temporary) / binary_name
        objects = []

        for index, source in enumerate(c_sources):
            object_path = Path(temporary) / f"source_{index}.o"
            dependencies = _compile_c_source(
                toolchain, source, object_path, output, extra_flags
            )
            c_dependencies.update(dependencies)
            objects.append(str(object_path))

        base = list(rvne_compile_command(toolchain, cpp_sources[0], temporary_binary))
        source_index = base.index(cpp_sources[0])
        command = (
            base[:source_index]
            + list(extra_flags)
            + ["-I", str(output), *cpp_sources]
            + ["-x", "none", *objects]
            + ["-x", "assembler-with-cpp", manifest["assembly_path"], "-x", "none"]
            + base[source_index + 1 :]
        )

        try:
            subprocess.run(
                command,
                cwd=output,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostic = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"Native RVNE link failed: {diagnostic}.") from exc

        with temporary_binary.open("rb") as executable:
            elf_header = executable.read(20)

        if (
            elf_header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(elf_header[18:20], "little") != 243
        ):
            raise RuntimeError("Native RVNE link did not produce a RISC-V ELF64.")

        os.replace(temporary_binary, binary)

    manifest.update(
        {
            "status": "compiled",
            "binary_path": str(binary),
            "binary_sha256": _sha256(binary),
            "elf_machine": 243,
            "elf_class": 64,
            "header_path": str(published_header),
            "runtime_path": str(published_runtime),
            "sources": sources,
            "c_sources": [str(source) for source in c_sources],
            "c_dependencies": sorted(str(path) for path in c_dependencies),
            "toolchain": rvne_compiler_identity(toolchain_root, required=True),
            "extra_flags": list(extra_flags),
            "source_hashes": {
                str(path): _sha256(Path(path))
                for path in (
                    *sources,
                    str(published_header),
                    manifest["assembly_path"],
                    *sorted(str(path) for path in c_dependencies),
                )
            },
        }
    )
    _write_manifest(manifest)

    return manifest


def _compile_c_source(toolchain, source, object_path, output_dir, extra_flags):
    base = list(rvne_compile_command(toolchain, source, object_path))
    source_index = base.index(str(source))
    flags = [
        flag for flag in base[:source_index] if flag not in {"-static", "-fuse-ld=lld"}
    ]
    flags[flags.index("-x") + 1] = "c"
    flags[flags.index("-std=c++17")] = "-std=gnu11"
    dependency_path = object_path.with_suffix(".d")
    command = (
        flags
        + list(extra_flags)
        + [
            "-I",
            str(output_dir),
            "-I",
            str(source.parent),
            "-MMD",
            "-MF",
            str(dependency_path),
            "-MT",
            "native_c_object",
            "-c",
            str(source),
            "-o",
            str(object_path),
        ]
    )

    try:
        subprocess.run(
            command,
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        diagnostic = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Native RVNE C compilation failed: {diagnostic}.") from exc

    dependencies = dependency_path.read_text(encoding="utf-8").replace("\\\n", "")
    paths = shlex.split(dependencies.split(":", 1)[1])

    return tuple((output_dir / path).resolve() for path in paths)


def _generate_kernel(application, operator, index, shapes, dtypes, output_dir):
    kernel_name = f"nt_{operator}_{index}"
    compilation = compile_kernel(
        CompileRequest(
            arrangement=_identity,
            application=application,
            tensors=tuple(
                Tensor(shape=shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)
            ),
            backend="rvne",
            caller="native",
            kernel_name=kernel_name,
        )
    )
    artifact = compilation.artifact
    source_path = output_dir / f"{kernel_name}.cpp"
    source_path.write_text(artifact.primary_source, encoding="utf-8")
    metadata_path = output_dir / f"{kernel_name}.json"
    metadata_path.write_text(
        json.dumps(ir_to_dict(artifact.metadata), indent=2) + "\n", encoding="utf-8"
    )
    abi = compilation.launch_abi
    specs = {spec.name: spec for spec in compilation.kernel.tensors}
    source_shapes = dict(zip(abi.public_args, shapes))
    parameters = []

    for name, dtype in zip(abi.public_args, dtypes):
        const = "" if name in abi.outputs else "const "
        parameters.append(f"{const}{_C_TYPES[dtype]} *{name}")

    prototype = []
    arguments = []

    for binding in abi.kernel_args:
        if binding.kind == "tensor":
            dtype = normalize_dtype(specs[binding.source].dtype)
            const = "" if binding.source in abi.outputs else "const "
            prototype.append(f"{const}{_C_TYPES[dtype]} *{binding.name}")
            arguments.append(binding.source)
        elif binding.kind == "shape":
            prototype.append(f"int64_t {binding.name}")
            arguments.append(str(source_shapes[binding.source][binding.dim]))
        elif binding.kind == "stride":
            prototype.append(f"int64_t {binding.name}")
            arguments.append(
                str(int(np.prod(source_shapes[binding.source][binding.dim + 1 :])))
            )
        else:
            raise ValueError(
                f"Unsupported native RVNE launch binding `{binding.kind}`."
            )

    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", artifact.entrypoint):
        raise ValueError("Native RVNE kernel has an invalid entrypoint.")

    wrapper = (
        f'extern "C" int {artifact.entrypoint}({", ".join(prototype)});\n'
        f"static int nt_{operator}_wrapper_{index}({', '.join(parameters)}) {{\n"
        f"    return {artifact.entrypoint}({', '.join(arguments)});\n"
        "}"
    )

    return {
        "operator": operator,
        "index": index,
        "kernel_name": kernel_name,
        "entrypoint": artifact.entrypoint,
        "source_path": str(source_path),
        "metadata_path": str(metadata_path),
        "source_sha256": _sha256(source_path),
        "source_route": artifact.metadata.get("source_route"),
        "launch_abi": ir_to_dict(abi),
        "pass_trace": list(compilation.pass_trace),
        "uses_spike_accumulator": "__builtin_riscv_calc_acc_32"
        in artifact.primary_source,
    }, wrapper


def _registry_array(type_name, name, rows):
    values = ",\n    ".join(rows) if rows else "{}"

    return (
        f'extern "C" const Nt{type_name}Spec nt_{name}_specs[] = {{\n    {values}\n}};\n'
        f'extern "C" const size_t nt_{name}_count = {len(rows)};\n\n'
    )


def _shape(values, *, rank):
    values = tuple(values)

    if len(values) != rank or any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
        for value in values
    ):
        raise ValueError(
            f"Native RVNE shape must have {rank} positive integer dimensions."
        )

    return tuple(int(value) for value in values)


def _sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _write_manifest(manifest):
    Path(manifest["manifest_path"]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


__all__ = ["generate_native_bundle", "compile_native_bundle"]
