"""TileLang artifact compilation, launch binding, and reload."""

from pathlib import Path

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.cache import (
    artifact_directory,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)


class TileLangMaterializer(Materializer):
    target = Target.TILELANG

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("TileLang built artifact does not contain a binary path.")

        import tilelang  # noqa: F401 -- exposes its bundled TVM runtime
        import tvm

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
            _runtime_wrapper,
            _verified_runtime_launch,
        )

        module = tvm.runtime.load_module(built.binary_path)
        function = module[built.source.kernel_name]

        return _verified_runtime_launch(
            _runtime_wrapper(
                _tilelang_streamed_call(function),
                _launch_abi_from_dict(built.abi),
                specs=_runtime_specs(built.source),
            )
        )


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.compiler.runtime import (
        Handle,
        _built_manifest,
        _export_library_atomic,
        _publish_library,
        _replace_file,
        import_python_module,
    )

    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "tilelang.py",
        cache_key=cache_key,
    )
    cache_library = (
        artifact_directory(cache_key) / f"{artifact.kernel_name}.tilelang.so"
    )

    with cache_lock(cache_library):
        if not cache_library.is_file():
            source_module = import_python_module(source)
            prim_func = getattr(source_module, artifact.entrypoint)()
            import tilelang

            kernel = tilelang.compile(
                prim_func,
                execution_backend="tvm_ffi",
                target="cuda",
            )
            bundled_library = getattr(kernel.adapter, "libpath", None)

            if bundled_library and Path(bundled_library).is_file():
                _replace_file(Path(bundled_library), cache_library)
            else:
                _export_library_atomic(kernel, cache_library)

        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _built_manifest(compilation, cache_key, source, cache_library),
        )

    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.tilelang.so",
    )
    import tilelang  # noqa: F401 -- exposes its bundled TVM runtime
    import tvm

    runtime_module = tvm.runtime.load_module(str(cache_library))
    function = runtime_module[artifact.kernel_name]
    from ninetoothed.compiler.runtime import (
        _runtime_wrapper,
        _verified_runtime_launch,
        runtime_device_types,
    )

    wrapped = _verified_runtime_launch(
        _runtime_wrapper(
            _tilelang_streamed_call(function),
            compilation.launch_abi,
            specs=compilation.kernel.tensors,
            device_types=runtime_device_types(compilation),
        )
    )

    return Handle(
        compilation,
        (runtime_module, function),
        wrapped,
        source,
        library_path,
    )


def _tilelang_streamed_call(function):
    """Bind the TileLang kernel to the caller's current CUDA stream per call."""
    import torch
    import tvm

    # Use ``torch._C._cuda_getCurrentRawStream`` to avoid building a ``Stream``
    # object per call, and refresh the tvm binding only when the (device,
    # stream) actually changes (the common case is a repeated launch).
    get_raw_stream = torch._C._cuda_getCurrentRawStream
    state = {}

    def call(*values):
        device = torch.cuda.current_device()
        stream = get_raw_stream(device)

        if state.get("key") != (device, stream):
            tvm.cuda(device).set_raw_stream(stream)
            state["key"] = (device, stream)

        function(*values)

    return call


__all__ = ["TileLangMaterializer"]
