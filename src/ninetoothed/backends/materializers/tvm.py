"""TVM artifact compilation, host/device launch binding, and reload."""

import os
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


class TvmMaterializer(Materializer):
    target = Target.TVM

    def jit_materialize(self, compilation, *, output_dir=None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.binary_path is None:
            raise ValueError("TVM built artifact does not contain a binary path.")
        import tilelang  # noqa: F401 -- exposes the bundled TVM runtime
        import tvm

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _runtime_specs,
            import_python_module,
        )

        runtime_module = tvm.runtime.load_module(built.binary_path)
        abi = _launch_abi_from_dict(built.abi)
        source_module = import_python_module(built.source_path)
        runtime_config = getattr(source_module, "NINETOOTHED_TVM_RUNTIME", None)
        if runtime_config is not None:
            ir_module = getattr(source_module, built.source.entrypoint)()
            metadata = _device_launch_metadata(ir_module, tvm)
            compute = _device_function(
                runtime_module, metadata, str(runtime_config["compute"])
            )
            cast = _device_function(
                runtime_module, metadata, str(runtime_config["cast"])
            )
            return _device_pipeline_wrapper(compute, cast, runtime_config, abi)
        function = runtime_module[built.source.kernel_name]
        return _host_wrapper(
            function,
            abi,
            specs=_runtime_specs(built.source),
        )


def _materialize(compilation, *, output_dir=None):
    from ninetoothed.compiler.runtime import (
        Handle,
        _built_manifest,
        _export_library_atomic,
        _publish_library,
        import_python_module,
    )

    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "tvm.py",
        cache_key=cache_key,
    )
    module = import_python_module(source)
    ir_module = getattr(module, artifact.entrypoint)()
    import tvm

    cache_library = artifact_directory(cache_key) / f"{artifact.kernel_name}.tvm.so"
    with cache_lock(cache_library):
        if not cache_library.is_file():
            runtime_module = _build_module(ir_module, tvm)
            _export_library_atomic(runtime_module, cache_library)
        write_manifest(
            cache_library.with_suffix(".manifest.json"),
            _built_manifest(compilation, cache_key, source, cache_library),
        )
    runtime_module = tvm.runtime.load_module(str(cache_library))
    library_path = _publish_library(
        cache_library,
        output_dir,
        f"{artifact.kernel_name}.tvm.so",
    )
    runtime_config = getattr(module, "NINETOOTHED_TVM_RUNTIME", None)
    if runtime_config is not None:
        device_metadata = _device_launch_metadata(ir_module, tvm)
        compute = _device_function(
            runtime_module,
            device_metadata,
            str(runtime_config["compute"]),
        )
        cast = _device_function(
            runtime_module,
            device_metadata,
            str(runtime_config["cast"]),
        )
        wrapped = _device_pipeline_wrapper(
            compute,
            cast,
            runtime_config,
            compilation.launch_abi,
        )
        return Handle(
            compilation,
            (runtime_module, compute[0], cast[0]),
            wrapped,
            source,
            library_path,
        )
    kernel = runtime_module[artifact.kernel_name]
    wrapped = _host_wrapper(
        kernel,
        compilation.launch_abi,
        specs=compilation.kernel.tensors,
    )
    return Handle(compilation, (runtime_module, kernel), wrapped, source, library_path)


def _device_launch_metadata(ir_module, tvm):
    target = tvm.target.Target("cuda")
    bound_target = target.with_host(tvm.target.Target("llvm"))
    bound = tvm.tirx.transform.BindTarget(bound_target)(ir_module)
    lowered = tvm.tirx.get_default_tir_pipeline(target)(bound)
    metadata = {}
    analyzer = tvm.arith.Analyzer()

    for global_var, function in lowered.functions_items():
        attrs = function.attrs
        if int(attrs.get("calling_conv", 0)) != 2:
            continue
        symbol = str(attrs.get("global_symbol", global_var.name_hint))
        thread_extent = {}
        for name, value in dict(attrs.get("thread_extent", {})).items():
            simplified = analyzer.simplify(value)
            try:
                thread_extent[str(name)] = int(simplified)
            except TypeError as exc:
                raise RuntimeError(
                    f"TVM launch extent `{name}` is not static after specialization: "
                    f"{simplified!r}."
                ) from exc
        launch_values = []
        for tag_value in attrs.get("tirx.kernel_launch_params", ()):
            tag = str(tag_value)
            if tag == "tirx.use_dyn_shared_memory":
                launch_values.append(int(attrs.get("dyn_shared_memory_buf", 0)))
            else:
                launch_values.append(thread_extent[tag])
        metadata[symbol] = {
            "function": symbol,
            "parameters": tuple(str(param) for param in function.params),
            "launch_values": tuple(launch_values),
        }
    return metadata


def _device_function(runtime_module, metadata, source_name):
    candidates = (f"{source_name}_kernel", source_name)
    symbol = next((name for name in candidates if name in metadata), None)
    if symbol is None:
        raise RuntimeError(f"TVM did not lower device function `{source_name}`.")
    for imported in runtime_module.imports_:
        try:
            return imported[symbol], metadata[symbol]
        except AttributeError:
            continue
    raise RuntimeError(f"TVM runtime module does not export `{symbol}`.")


def _device_pipeline_wrapper(compute, cast, config, abi):
    from ninetoothed.compiler.runtime import _public_values

    compute_function, compute_metadata = compute
    cast_function, cast_metadata = cast

    def launch(*args, **kwargs):
        import torch
        import tvm

        public = _public_values(abi, args, kwargs)
        output = public[str(config["output"])]
        workspace = torch.empty(
            int(config["workspace_numel"]),
            dtype=getattr(torch, str(config["workspace_dtype"])),
            device=output.device,
        )

        def values(argument_map, metadata):
            result = []
            for parameter in metadata["parameters"]:
                name = argument_map[str(parameter)]
                value = workspace if name == "$workspace" else public[str(name)]
                if hasattr(value, "data_ptr"):
                    value = value.data_ptr()
                elif hasattr(value, "item"):
                    value = value.item()
                result.append(value)
            return result

        device = torch.cuda.current_device()
        tvm.cuda(device).set_raw_stream(torch.cuda.current_stream(device).cuda_stream)
        compute_function(
            *values(config["compute_args"], compute_metadata),
            *compute_metadata["launch_values"],
        )
        cast_function(
            *values(config["cast_args"], cast_metadata),
            *cast_metadata["launch_values"],
        )
        return output

    return launch


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


def _build_module(ir_module, tvm):
    from ninetoothed.backends.materializers.cuda import _nvcc

    nvcc_dir = str(Path(_nvcc()).parent)
    previous_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join((nvcc_dir, previous_path))
    try:
        return tvm.tirx.build(ir_module, target="cuda")
    finally:
        os.environ["PATH"] = previous_path


__all__ = ["TvmMaterializer"]
