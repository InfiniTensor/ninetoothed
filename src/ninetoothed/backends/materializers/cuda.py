"""CUDA artifact compilation, launch binding, and reload."""

import subprocess
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import cuda_compile_command
from ninetoothed.compiler.cache import (
    TOOLCHAIN_LOCK_DIR,
    cache_lock,
)


class CudaMaterializer(Materializer):
    target = Target.CUDA

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        from ninetoothed.backends.materializers.native import load_native_artifact

        return load_native_artifact(built, _cuda_wrapper)


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.backends.materializers.native import materialize_native_library

    return materialize_native_library(
        compilation,
        source_suffix="cu",
        compile_library=_compile_library,
        wrapper=_cuda_wrapper,
        output_dir=output_dir,
    )


def _cuda_wrapper(function, abi, tensor_specs):
    from ninetoothed.backends.materializers.native import native_wrapper

    return native_wrapper(
        function,
        abi,
        tensor_specs,
        device_types=("cuda",),
        module="cuda",
        stream_attr="cuda_stream",
    )


def _compile_library(source: Path, library: Path, *, arch: str, compiler=None) -> None:
    from ninetoothed.backends.materializers.native import native_library_output

    with native_library_output(library) as temporary:
        with cache_lock(TOOLCHAIN_LOCK_DIR / "nvcc"):
            subprocess.run(
                cuda_compile_command(source, temporary, arch=arch, nvcc=compiler),
                check=True,
            )


__all__ = ["CudaMaterializer"]
