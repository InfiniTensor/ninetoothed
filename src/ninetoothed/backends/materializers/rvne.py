"""Build RVNE AOT executables and run NumPy arguments through QEMU."""

import math
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from ninetoothed.backends.core import Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.rvne_toolchain import (
    find_rvne_toolchain,
    rvne_compile_command,
)
from ninetoothed.compiler.cache import (
    artifact_directory,
    atomic_write_bytes,
    atomic_write_text,
    cache_lock,
    compilation_cache_key,
    stable_digest,
    write_manifest,
)
from ninetoothed.dtype import normalize_dtype

_C_TYPES = {
    "bool": "bool",
    "float32": "float",
    "float64": "double",
    **{
        f"{prefix}{bits}": f"{prefix}{bits}_t"
        for prefix in ("int", "uint")
        for bits in (8, 16, 32, 64)
    },
}


class RvneMaterializer(Materializer):
    target = Target.RVNE

    def jit_materialize(self, compilation, *, output_dir=None):
        raise ValueError("RVNE supports AOT execution through QEMU only.")

    def aot_build(self, compilation, *, output_dir):
        from ninetoothed.compiler.runtime import Handle, _built_manifest

        root = dict(compilation.request.backend_options or {}).get("toolchain_root")
        toolchain = find_rvne_toolchain(root)
        artifact = compilation.artifact
        specs = {value["name"]: value for value in artifact.metadata["tensors"]}
        harness = _harness(artifact.entrypoint, compilation.launch_abi, specs)
        cache_key = stable_digest(
            {"compilation": compilation_cache_key(compilation), "harness": harness}
        )
        directory = artifact_directory(cache_key)
        source = directory / f"{artifact.kernel_name}.rvne.cpp"
        binary = directory / f"{artifact.kernel_name}.rvne.elf"
        atomic_write_text(source, artifact.primary_source + "\n" + harness)

        with cache_lock(binary):
            if not binary.is_file():
                _compile_executable(toolchain, source, binary)

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        stem = f"{artifact.kernel_name}.{cache_key[:16]}.rvne"
        published_source = output / f"{stem}.cpp"
        published_binary = output / f"{stem}.elf"
        atomic_write_text(published_source, source.read_text(encoding="utf-8"))
        atomic_write_bytes(published_binary, binary.read_bytes())
        published_binary.chmod(0o755)
        runtime_artifact = replace(
            artifact,
            metadata=dict(artifact.metadata)
            | {"rvne_runtime": {"toolchain_root": str(toolchain.root), "protocol": 1}},
        )
        launch = _wrapper(published_binary, compilation.launch_abi, specs, toolchain)
        handle = Handle(compilation, None, launch, published_source, published_binary)
        handle._built_artifact = replace(
            handle._built_artifact, source=runtime_artifact, cache_key=cache_key
        )
        manifest = _built_manifest(
            compilation, cache_key, published_source, published_binary
        )
        manifest["rvne_runtime"] = runtime_artifact.metadata["rvne_runtime"]
        write_manifest(handle._built_artifact.manifest_path, manifest)

        return handle

    def load_built_artifact(self, built):
        from ninetoothed.compiler.runtime import _launch_abi_from_dict

        if built.binary_path is None or not Path(built.binary_path).is_file():
            raise ValueError("RVNE built artifact does not contain an executable file.")

        runtime = dict(built.source.metadata.get("rvne_runtime", {}))

        if runtime.get("protocol") != 1:
            raise ValueError("Unsupported RVNE subprocess ABI; rebuild the artifact.")

        toolchain = find_rvne_toolchain(runtime.get("toolchain_root"))
        specs = {value["name"]: value for value in built.source.metadata["tensors"]}

        return _wrapper(
            Path(built.binary_path), _launch_abi_from_dict(built.abi), specs, toolchain
        )


def _compile_executable(toolchain, source, binary):
    binary.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=binary.parent) as temporary:
        output = Path(temporary) / binary.name

        try:
            result = subprocess.run(
                rvne_compile_command(toolchain, source, output),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostic = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"RVNE compilation failed: {diagnostic}.") from exc

        if not output.is_file():
            raise RuntimeError(
                f"RVNE compiler did not produce an executable: {result.stderr}."
            )

        os.replace(output, binary)


def _dtype(binding, specs):
    if binding.kind in {"tensor", "scalar", "constexpr"}:
        name = normalize_dtype(str(specs[binding.source]["dtype"]))
    else:
        name = "int64"

    if name not in _C_TYPES:
        raise TypeError(f"Unsupported RVNE ABI dtype `{name}`.")
    return name


def _harness(entrypoint, abi, specs):
    if not entrypoint or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", entrypoint) is None:
        raise ValueError("RVNE artifact has an invalid entrypoint.")

    statements = []
    writes = []
    arguments = []
    cleanup = []

    for index, binding in enumerate(abi.kernel_args):
        c_type = _C_TYPES[_dtype(binding, specs)]
        name = f"nt_arg_{index}"
        arguments.append(name)

        if binding.kind == "tensor":
            statements.append(f"uint64_t nt_count_{index}; {c_type} *{name} = nullptr;")
            statements.append(
                f"if (!nt_read_array(argv[{index + 1}], (void **)&{name}, sizeof({c_type}), &nt_count_{index})) return 71;"
            )
            cleanup.append(f"free({name});")

            if (
                binding.access in {"write", "read_write"}
                or binding.source in abi.outputs
            ):
                writes.append(
                    f"if (!nt_write_array(argv[{index + 1}], {name}, sizeof({c_type}), nt_count_{index})) return 72;"
                )
        elif binding.kind in {"scalar", "constexpr", "shape", "stride", "meta"}:
            statements.append(f"{c_type} {name};")
            statements.append(
                f"if (!nt_read_scalar(argv[{index + 1}], &{name}, sizeof({name}))) return 71;"
            )
        else:
            raise ValueError(f"Unsupported RVNE launch binding `{binding.kind}`.")

    return (
        _HARNESS_IO
        + "\nint main(int argc, char **argv) {\n"
        + "\n".join(
            f"  {line}"
            for line in (
                f"if (argc != {len(abi.kernel_args) + 1}) return 70;",
                *statements,
                f"int result = {entrypoint}({', '.join(arguments)});",
                "if (result != 0) return result;",
                *writes,
                *cleanup,
                "return 0;",
            )
        )
        + "\n}\n"
    )


def _binding_value(binding, public):
    if binding.kind == "stride":
        value = public[binding.source]

        return value.strides[binding.dim] // value.itemsize

    from ninetoothed.compiler.runtime import _binding_value as bind

    return bind(binding, public)


def _validate(public, abi, specs):
    import numpy as np

    arrays = {}
    symbols = {}

    for binding in abi.kernel_args:
        if binding.kind != "tensor":
            value = _binding_value(binding, public)
            _scalar_value(value, np.dtype(_dtype(binding, specs)), binding.name)

            if (
                binding.kind == "constexpr"
                and binding.value is not None
                and value != binding.value
            ):
                raise ValueError(
                    f"RVNE argument `{binding.source}` does not match its specialized value."
                )

            continue

        name = binding.source
        value = public[name]
        spec = specs[name]

        if not isinstance(value, np.ndarray):
            raise TypeError(f"RVNE tensor `{name}` must be a NumPy array.")

        if value.dtype != np.dtype(_dtype(binding, specs)) or not value.dtype.isnative:
            raise TypeError(
                f"RVNE tensor `{name}` requires dtype {_dtype(binding, specs)}."
            )

        if not value.flags.c_contiguous:
            raise ValueError(f"RVNE tensor `{name}` must be C-contiguous.")

        if (
            binding.access in {"write", "read_write"} or name in abi.outputs
        ) and not value.flags.writeable:
            raise ValueError(f"RVNE output `{name}` must be writable.")

        attrs = spec.get("attrs", {})
        shape = attrs.get("source_shape", spec.get("shape", ()))
        ndim = attrs.get("source_ndim", len(shape))

        if value.ndim != ndim:
            raise ValueError(f"RVNE tensor `{name}` requires rank {ndim}.")

        for expected, actual in zip(shape, value.shape):
            text = str(expected)

            if text.isdigit():
                if int(text) != actual:
                    raise ValueError(
                        f"RVNE tensor `{name}` has incompatible shape {value.shape}."
                    )
            elif text.isidentifier():
                if text in symbols and symbols[text] != actual:
                    raise ValueError(
                        f"RVNE shape symbol `{text}` has inconsistent values."
                    )

                symbols[text] = actual
            else:
                raise ValueError(f"Unsupported RVNE source shape expression `{text}`.")

        arrays[name] = value

    items = list(arrays.items())

    for index, (name, value) in enumerate(items):
        for other_name, other in items[index + 1 :]:
            if np.shares_memory(value, other):
                raise ValueError(
                    f"RVNE arguments `{name}` and `{other_name}` must not alias."
                )


def _scalar_value(value, dtype, name):
    """Validate scalar types and bounds before converting untyped Python values."""
    import numpy as np

    if isinstance(value, (np.generic, np.ndarray)):
        scalar = np.asarray(value)

        if scalar.ndim != 0:
            raise TypeError(f"RVNE scalar `{name}` must be a scalar value.")

        if scalar.dtype != dtype or not scalar.dtype.isnative:
            raise TypeError(f"RVNE scalar `{name}` requires dtype {dtype}.")
        return scalar

    if type(value) is bool:
        if dtype.kind != "b":
            raise TypeError(f"RVNE scalar `{name}` requires dtype {dtype}, not bool.")
    elif type(value) is int:
        if dtype.kind in {"i", "u"}:
            limits = np.iinfo(dtype)

            if value < limits.min or value > limits.max:
                raise OverflowError(
                    f"RVNE scalar `{name}` is outside the {dtype} range."
                )
        elif dtype.kind == "f":
            if abs(value) > float(np.finfo(dtype).max):
                raise OverflowError(
                    f"RVNE scalar `{name}` is outside the {dtype} range."
                )
        else:
            raise TypeError(f"RVNE scalar `{name}` requires dtype {dtype}.")
    elif type(value) is float:
        if dtype.kind != "f":
            raise TypeError(f"RVNE scalar `{name}` requires dtype {dtype}, not float.")

        if math.isfinite(value) and abs(value) > float(np.finfo(dtype).max):
            raise OverflowError(f"RVNE scalar `{name}` is outside the {dtype} range.")
    else:
        raise TypeError(f"RVNE scalar `{name}` must be a Python or NumPy scalar.")

    return np.asarray(value, dtype=dtype)


def _wrapper(binary, abi, specs, toolchain):
    def launch(*args, **kwargs):
        import numpy as np

        from ninetoothed.compiler.runtime import _first_output, _public_values

        public = _public_values(abi, args, kwargs)
        _validate(public, abi, specs)

        with tempfile.TemporaryDirectory(prefix="ninetoothed-rvne-") as temporary:
            paths = []
            outputs = []

            for index, binding in enumerate(abi.kernel_args):
                path = Path(temporary) / f"argument-{index}.bin"
                paths.append(str(path))
                value = _binding_value(binding, public)
                dtype = np.dtype(_dtype(binding, specs))

                if binding.kind == "tensor":
                    path.write_bytes(
                        struct.pack("<Q", value.size) + value.tobytes(order="C")
                    )

                    if (
                        binding.access in {"write", "read_write"}
                        or binding.source in abi.outputs
                    ):
                        outputs.append((path, value))
                else:
                    scalar = _scalar_value(value, dtype, binding.name)
                    path.write_bytes(scalar.tobytes())

            try:
                subprocess.run(
                    [
                        str(toolchain.emulator),
                        "-L",
                        str(toolchain.sysroot),
                        str(binary),
                        *paths,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                diagnostic = getattr(exc, "stderr", "") or str(exc)
                raise RuntimeError(
                    f"RVNE QEMU execution failed: {diagnostic}."
                ) from exc

            updates = []

            for path, value in outputs:
                data = path.read_bytes()

                if (
                    len(data) != 8 + value.nbytes
                    or struct.unpack("<Q", data[:8])[0] != value.size
                ):
                    raise RuntimeError(
                        "RVNE executable returned an invalid output buffer."
                    )

                updates.append(
                    (
                        value,
                        np.frombuffer(data, dtype=value.dtype, offset=8).reshape(
                            value.shape
                        ),
                    )
                )

            for value, updated in updates:
                np.copyto(value, updated)

        return _first_output(abi, public)

    return launch


_HARNESS_IO = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static bool nt_read_scalar(const char *path, void *value, size_t size) {
  FILE *file = fopen(path, "rb");
  if (!file) return false;
  bool ok = fread(value, 1, size, file) == size && fgetc(file) == EOF;
  fclose(file);
  return ok;
}
static bool nt_read_array(const char *path, void **value, size_t item, uint64_t *count) {
  FILE *file = fopen(path, "rb");
  if (!file) return false;
  if (fread(count, sizeof(*count), 1, file) != 1 || *count > SIZE_MAX / item) {
    fclose(file);
    return false;
  }
  size_t bytes = *count * item;
  *value = malloc(bytes ? bytes : 1);
  bool ok = *value && fread(*value, 1, bytes, file) == bytes && fgetc(file) == EOF;
  fclose(file);
  return ok;
}
static bool nt_write_array(const char *path, const void *value, size_t item, uint64_t count) {
  FILE *file = fopen(path, "wb");
  if (!file) return false;
  size_t bytes = count * item;
  bool ok = fwrite(&count, sizeof(count), 1, file) == 1 && fwrite(value, 1, bytes, file) == bytes;
  return fclose(file) == 0 && ok;
}
"""
