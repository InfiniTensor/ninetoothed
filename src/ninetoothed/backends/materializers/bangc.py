"""BangC artifact compilation, launch binding, and reload.

Binaries are built with the Cambricon ``cncc`` toolchain.  When cncc is not
installed locally the materializer transparently forwards compilation to a
remote host (optionally inside a docker container) described by
``NINETOOTHED_BANGC_SSH``/``NINETOOTHED_BANGC_CONTAINER``, then loads the
resulting shared library through the same C-ABI launcher contract as the CUDA
backend, bound to ``cnrtQueue_t`` streams.
"""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import (
    bangc_compile_command,
    bangc_remote_base_command,
    bangc_remote_spec,
    resolve_bangc_arch,
)
from ninetoothed.compiler.cache import (
    TOOLCHAIN_LOCK_DIR,
    artifact_directory,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)
from ninetoothed.targets import runtime_device_types


class BangCMaterializer(Materializer):
    target = Target.BANGC

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("BangC built artifact does not contain a binary path.")

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
        )

        library = ctypes.CDLL(built.binary_path)
        function = getattr(library, built.source.entrypoint)
        function.restype = ctypes.c_int
        specs = _runtime_specs(built.source)

        return _bangc_wrapper(
            function,
            _launch_abi_from_dict(built.abi),
            specs,
            device_types=runtime_device_types(built.source),
        )


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
        "mlu",
        cache_key=cache_key,
    )
    cache_library = artifact_directory(cache_key) / f"{artifact.kernel_name}.bangc.so"

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
        f"{artifact.kernel_name}.bangc.so",
    )
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, artifact.entrypoint)
    function.restype = ctypes.c_int
    wrapped = _bangc_wrapper(
        function,
        compilation.launch_abi,
        compilation.kernel.tensors,
        device_types=runtime_device_types(compilation),
    )

    return Handle(compilation, function, wrapped, source, library_path)


def _bangc_wrapper(function, abi, tensor_specs, *, device_types=("mlu",)):
    from ninetoothed.compiler.runtime import (
        KernelLaunchError,
        _bound_values,
        _empty_launch,
        _first_output,
        _public_values,
    )

    spec_by_name = {spec.name: spec for spec in tensor_specs}

    def launch(*args, **kwargs):
        public = _public_values(
            abi, args, kwargs, specs=tensor_specs, device_types=device_types
        )

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        import torch

        if not hasattr(torch, "mlu"):
            raise RuntimeError(
                "The BangC backend requires torch.mlu; import torch_mlu before "
                "launching kernels."
            )

        values, keepalive = _bound_values(
            abi,
            public,
            scalar_mode="cuda",
            specs=spec_by_name,
            cuda_scalar=_bangc_scalar,
        )
        stream = torch.mlu.current_stream().mlu_stream
        result = function(*values, ctypes.c_void_p(stream))
        del keepalive

        if result != 0:
            raise KernelLaunchError(result)
        return _first_output(abi, public)

    return launch


def _bangc_scalar(value, dtype):
    from ninetoothed.backends.materializers.cuda import _cuda_scalar

    return _cuda_scalar(value, dtype)


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
        with cache_lock(TOOLCHAIN_LOCK_DIR / "cncc"):
            arch = resolve_bangc_arch(arch)

            try:
                command = bangc_compile_command(source, temporary, arch=arch)
                subprocess.run(
                    command,
                    check=True,
                )
            except (RuntimeError, subprocess.CalledProcessError) as local_error:
                remote = bangc_remote_spec()

                if remote is None:
                    raise

                try:
                    _compile_library_remote(source, temporary, arch=arch, spec=remote)
                except (RuntimeError, subprocess.CalledProcessError) as remote_error:
                    raise RuntimeError(
                        f"BangC compilation failed locally ({local_error}) and "
                        f"remotely ({remote_error})."
                    ) from remote_error

        os.replace(temporary, library)
    finally:
        temporary.unlink(missing_ok=True)


def _compile_library_remote(
    source: Path, library: Path, *, arch: str, spec: dict
) -> None:
    """Compile on a remote cncc host and stream the binary back.

    The driver shell is ``bash -s``, so the entire exchange travels over the
    ssh stdin/stdout channel: the generated source is embedded base64 (safe
    against any quoting layer), the remote side decodes + compiles it, and
    the resulting shared library is streamed back base64 on stdout between
    sentinels.  This avoids both argv mangling on Windows OpenSSH builds and
    any need for shared filesystems between hosts and containers.
    """
    import base64

    remote_dir = str(spec["remote_dir"])
    base = bangc_remote_base_command(spec)
    source_remote = f"{remote_dir}/{source.name}"
    library_remote = f"{remote_dir}/{library.name}"
    source_b64 = base64.b64encode(source.read_bytes()).decode("ascii")
    script = (
        "set -e\n"
        f"mkdir -p {remote_dir}\n"
        f"printf '%s' '{source_b64}' | base64 -d > {source_remote}\n"
        "echo NT_COMPILE_BEGIN\n"
        "if ! cncc --shared -fPIC -O3 --bang-arch="
        f"{arch} {source_remote} -o {library_remote} 1>&2; then\n"
        "  echo NT_COMPILE_FAILED 1>&2\n"
        "  exit 1\n"
        "fi\n"
        "echo NT_COMPILE_OK\n"
        f"base64 -w0 {library_remote}\n"
        "echo\n"
        "echo NT_COMPILE_END\n"
    )

    result = subprocess.run(
        base,
        input=script.encode("utf-8"),
        capture_output=True,
        timeout=600,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")

    if result.returncode != 0 or "NT_COMPILE_OK" not in stdout:
        stderr = result.stderr.decode("utf-8", errors="replace")

        raise RuntimeError(
            f"Remote BangC compilation failed (rc={result.returncode}): "
            f"{stderr[-2000:]}."
        )

    payload = stdout.split("NT_COMPILE_OK", 1)[1]
    payload = payload.split("NT_COMPILE_BEGIN", 1)[-1]
    encoded = payload.split("NT_COMPILE_END", 1)[0].strip()
    library.write_bytes(base64.b64decode(encoded))


__all__ = ["BangCMaterializer"]
