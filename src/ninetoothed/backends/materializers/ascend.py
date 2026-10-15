import functools
import inspect
import os
from pathlib import Path
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    compilation_cache_key,
    stable_digest,
    triton_cache_directory,
    write_source,
)


class AscendMaterializer(Materializer):
    target = Target.ASCEND

    def jit_materialize(
        self, compilation: Any, *, output_dir: Path | str | None = None
    ):
        del output_dir

        return _materialize_ascend(compilation)

    def aot_build(self, compilation: Any, *, output_dir: Path | str):
        from ninetoothed.backends.materializers.ascend_aot import build

        return build(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: Any):
        from ninetoothed.backends.materializers.ascend_aot import load

        return load(built)


def _materialize_ascend(compilation: Any):
    from ninetoothed.backends.ascend import activate_triton_backend
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        _verified_runtime_launch,
        import_python_module,
    )
    from ninetoothed.targets import runtime_device_types

    activate_triton_backend()
    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)

    # 1. 写入替换/转换后的源码（标明后缀或驱动）.
    source_path = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "ascend_triton.py",
        cache_key=cache_key,
    )

    cache_directory = triton_cache_directory(TRITON_CACHE_DIR)
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_directory))

    # 2. 动态加载 Python 模块并获取入口 launch 函数与 kernel.
    module = import_python_module(str(source_path))
    launch = getattr(module, artifact.entrypoint)
    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    candidates = tuple(compilation.launch_plan.tuning_candidates)
    candidate = candidates[0] if candidates else None
    function = launch
    meta_values = {}

    if candidate is not None:
        bindings = {
            binding.name: binding for binding in compilation.launch_abi.kernel_args
        }
        meta_values = {
            str(bindings[name].source): value
            for name, value in dict(candidate.get("meta_parameters", {})).items()
        }
        private_meta = {
            f"_ninetoothed_{name.lower()}": value
            for name, value in dict(
                candidate.get("private_meta_parameters", {})
            ).items()
        }
        function = functools.partial(
            launch,
            **private_meta,
            _ninetoothed_num_warps=int(candidate["num_warps"]),
            _ninetoothed_num_stages=int(candidate["num_stages"]),
        )

    specialized_launch = _specialize_runtime_bounds(
        function,
        artifact.kernel_name,
        artifact.primary_source,
        cache_key,
    )

    # 3. 封装 Python 运行时 Launch.
    wrapped = _verified_runtime_launch(
        _runtime_wrapper(
            specialized_launch,
            compilation.launch_abi,
            specs=compilation.kernel.tensors,
            device_types=runtime_device_types(compilation),
            binding_overrides=meta_values,
        )
    )

    # 4. 返回包含模块与句柄的 Handle 对象.
    handle = Handle(compilation, kernel, wrapped, source_path)
    handle._selected_tuning_candidate = candidate

    return handle


def _specialize_runtime_bounds(launch, kernel_name, source, cache_key):
    signature = inspect.signature(launch)
    modules = {}

    def specialized(*values):
        scalar_values = {}

        for parameter, value in zip(signature.parameters.values(), values):
            if isinstance(value, (bool, int)):
                scalar_values[parameter.name] = value

        if not scalar_values:
            return launch(*values)

        specialization_key = stable_digest(
            {"cache_key": cache_key, "scalars": scalar_values}
        )

        if specialization_key not in modules:
            from ninetoothed.ascendifier import Ascendifier

            specialized_source = Ascendifier(
                omit_static_bounds=True,
                scalar_values=scalar_values,
            ).transform(source)
            specialized_path = write_source(
                kernel_name,
                specialized_source,
                "ascend_triton.py",
                cache_key=specialization_key,
            )
            from ninetoothed.compiler.runtime import import_python_module

            specialized_module = import_python_module(str(specialized_path))
            modules[specialization_key] = getattr(
                specialized_module,
                getattr(launch, "__name__", launch.func.__name__),
            )

        return modules[specialization_key](*values)

    return specialized
