"""Local CANN compilation, NPU stream binding and artifact reload."""

import subprocess
from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.backends.toolchain import ascendc_compile_command
from ninetoothed.compiler.cache import TOOLCHAIN_LOCK_DIR, cache_lock
from ninetoothed.dtype import normalize_dtype


class AscendCMaterializer(Materializer):
    target = Target.ASCENDC

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        from ninetoothed.backends.materializers.native import load_native_artifact

        def wrapper(function, abi, specs):
            _validate_half_host_abi(built, abi, specs)

            return _ascendc_wrapper(function, abi, specs)

        return load_native_artifact(built, wrapper)


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.backends.materializers.native import materialize_native_library

    return materialize_native_library(
        compilation,
        source_suffix="ascendc",
        compile_library=_compile_library,
        wrapper=_ascendc_wrapper,
        output_dir=output_dir,
    )


def _ascendc_wrapper(function, abi, tensor_specs):
    from ninetoothed.backends.materializers.native import native_wrapper

    return native_wrapper(
        function,
        abi,
        tensor_specs,
        device_types=("npu",),
        module="npu",
        stream_attr="npu_stream",
    )


def _compile_library(source: Path, library: Path, *, arch: str, compiler=None) -> None:
    from ninetoothed.backends.materializers.native import native_library_output

    with native_library_output(library) as temporary:
        with cache_lock(TOOLCHAIN_LOCK_DIR / "ccec"):
            from ninetoothed.backends.toolchain import resolve_ascendc_arch

            arch = resolve_ascendc_arch(arch)

            command = ascendc_compile_command(
                source, temporary, arch=arch, ccec=compiler
            )
            subprocess.run(command, check=True)


def _validate_half_host_abi(built, abi, specs):
    spec_by_name = {spec.name: spec for spec in specs}
    half_bindings = tuple(
        binding.name
        for binding in abi.kernel_args
        if binding.kind in {"scalar", "constexpr"}
        and (spec := spec_by_name.get(binding.source or binding.name)) is not None
        and normalize_dtype(str(spec.dtype)) == "float16"
    )

    if half_bindings:
        host_abi = built.source.metadata.get("ascendc_host_scalar_abi", {})

        if host_abi.get("version") != "uint16-half-v1" or not set(
            half_bindings
        ).issubset(host_abi.get("half_storage_bindings", ())):
            raise ValueError(
                "AscendC artifacts with float16 scalar arguments require "
                "the uint16-half-v1 host ABI; rebuild this legacy artifact."
            )
