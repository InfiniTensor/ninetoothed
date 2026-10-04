from typing import Any

from ninetoothed.ascendifier import Ascendifier
from ninetoothed.backends.core import Artifact, Target
from ninetoothed.backends.emitters.base import EmitterTarget
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.ir import Kernel


class AscendEmitter(EmitterTarget):
    target = Target.ASCEND

    def __init__(self, target_context: Any = None):
        self.target_context = target_context

    def emit(self, kernel: Kernel) -> Artifact:
        # 1. 复用 TritonEmitter 生成基础 Triton 源码与元数据
        triton_emitter = TritonTarget(target_context=self.target_context)
        base_artifact = triton_emitter.emit(kernel)

        # 2. 实例化 Ascendifier 并对生成的源码进行改写
        ascendifier = Ascendifier()
        ascend_source = ascendifier.transform(base_artifact.primary_source)

        # 3. 构建并返回封装了 Ascend 源码的 Artifact 对象
        primary_filename = f"{base_artifact.kernel_name}.cpp"

        return Artifact(
            backend=Target.ASCEND,
            kernel_name=base_artifact.kernel_name,
            language="cpp",
            sources={primary_filename: ascend_source},
            entrypoint=base_artifact.entrypoint,
            metadata=dict(base_artifact.metadata) | {"target": Target.ASCEND},
        )