import os
from pathlib import Path
from typing import Any

from ninetoothed.backends.core import Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.cache import (
    TRITON_CACHE_DIR,
    compilation_cache_key,
    write_source,
)


class AscendMaterializer(Materializer):
    target = Target.ASCEND

    def jit_materialize(self, compilation: Any, *, output_dir: Path | str | None = None):
        del output_dir
        return _materialize_ascend(compilation)

    def aot_build(self, compilation: Any, *, output_dir: Path | str):
        raise NotImplementedError("Ascend AOT materialization is not yet implemented.")

    def load_built_artifact(self, built: Any):
        raise NotImplementedError("Ascend AOT artifact loading is not yet implemented.")


def _materialize_ascend(compilation: Any):
    from ninetoothed.compiler.runtime import (
        Handle,
        _runtime_wrapper,
        _verified_runtime_launch,
        import_python_module,
    )
    from ninetoothed.targets import runtime_device_types

    artifact = compilation.artifact
    cache_key = compilation_cache_key(compilation)

    # 1. 写入替换/转换后的源码（标明后缀或驱动）
    source_path = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "ascend_triton.py",
        cache_key=cache_key,
    )

    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))

    # 2. 动态加载 Python 模块并获取入口 launch 函数与 kernel
    module = import_python_module(str(source_path))
    launch = getattr(module, artifact.entrypoint)
    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)

    # 3. 封装 Python 运行时 Launch
    wrapped = _verified_runtime_launch(
        _runtime_wrapper(
            launch,
            compilation.launch_abi,
            specs=compilation.kernel.tensors,
            device_types=runtime_device_types(compilation),
        )
    )

    # 4. 返回包含模块与句柄的 Handle 对象
    return Handle(compilation, kernel, wrapped, source_path)