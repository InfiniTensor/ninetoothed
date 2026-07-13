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
        )

        module = tvm.runtime.load_module(built.binary_path)
        function = module[built.source.kernel_name]

        return _host_wrapper(
            function,
            _launch_abi_from_dict(built.abi),
            specs=_runtime_specs(built.source),
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
    wrapped = _host_wrapper(
        function,
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
    )

    return Handle(
        compilation,
        (runtime_module, function),
        wrapped,
        source,
        library_path,
    )


def _host_wrapper(function, abi, *, specs=()):
    from ninetoothed.compiler.runtime import (
        _bound_values,
        _empty_launch,
        _first_output,
        _flatten_ffi_tensor_args,
        _public_values,
    )

    def launch(*args, **kwargs):
        import torch
        import tvm

        public = _public_values(abi, args, kwargs, specs=specs)

        if _empty_launch(abi, public):
            return _first_output(abi, public)

        values, keepalive = _bound_values(abi, public, scalar_mode="value")
        values, flattened = _flatten_ffi_tensor_args(abi.kernel_args, values)
        keepalive.extend(flattened)
        converted = [
            tvm.runtime.from_dlpack(value)
            if isinstance(value, torch.Tensor) and value.ndim > 0
            else value.item()
            if hasattr(value, "item")
            else value
            for value in values
        ]
        device = torch.cuda.current_device()
        tvm.cuda(device).set_raw_stream(torch.cuda.current_stream(device).cuda_stream)
        function(*converted)
        del keepalive

        return _first_output(abi, public)

    return launch


__all__ = ["TileLangMaterializer"]
