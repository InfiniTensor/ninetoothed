"""Bangc local SDK compilation, launch binding, and reload."""

import subprocess
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import (
    bangc_compile_command,
    resolve_bangc_arch,
)
from ninetoothed.compiler.cache import (
    TOOLCHAIN_LOCK_DIR,
    cache_lock,
)
from ninetoothed.targets import runtime_device_types


class BangCMaterializer(Materializer):
    target = Target.BANGC

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        from ninetoothed.backends.materializers.staged import aot_monolithic_compilation

        return _materialize(
            aot_monolithic_compilation(compilation), output_dir=output_dir
        )

    def load_built_artifact(self, built: BuiltArtifact):
        from ninetoothed.backends.materializers.native import load_native_artifact

        return load_native_artifact(
            built,
            _bangc_wrapper,
            wrapper_options={"device_types": runtime_device_types(built.source)},
        )


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.backends.materializers.native import materialize_native_library

    return materialize_native_library(
        compilation,
        source_suffix="mlu",
        compile_library=_compile_library,
        wrapper=_bangc_wrapper,
        stage_materializer=_materialize,
        device_type="mlu",
        output_dir=output_dir,
        wrapper_options={"device_types": runtime_device_types(compilation)},
    )


def _bangc_wrapper(function, abi, tensor_specs, *, device_types=("mlu",)):
    from ninetoothed.backends.materializers.native import native_wrapper

    return native_wrapper(
        function,
        abi,
        tensor_specs,
        device_types=device_types,
        module="mlu",
        stream_attr="mlu_stream",
    )


def _compile_library(source: Path, library: Path, *, arch: str, compiler=None) -> None:
    from ninetoothed.backends.materializers.native import native_library_output

    with native_library_output(library) as temporary:
        with cache_lock(TOOLCHAIN_LOCK_DIR / "cncc"):
            arch = resolve_bangc_arch(arch)

            command = bangc_compile_command(source, temporary, arch=arch, cncc=compiler)
            subprocess.run(command, check=True)


__all__ = ["BangCMaterializer"]
