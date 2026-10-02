from typing import Any
from ninetoothed.backends.core import Artifact, Target
from ninetoothed.backends.emitters.base import EmitterTarget 
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.ir import Kernel
from ninetoothed.ascendifier import Ascendifier  # 导入已 checkout 出来的 Ascendifier


class AscendEmitter(EmitterTarget):
    target = Target.ASCEND

    def __init__(self, target_context: Any = None):
        self.target_context = target_context

    def emit(self, kernel: Kernel) -> Artifact:
        # 1. 复用 TritonEmitter 生成基础 Triton 源码与元数据
        triton_emitter = TritonTarget(target_context=self.target_context)
        base_artifact = triton_emitter.emit(kernel)

        # 2. 实例化 Ascendifier 并对生成的源码/AST 进行改写
        ascendifier = Ascendifier()
        
        # 假设 Ascendifier 接受源码字符串并返回修改后的源码
        # （如果你的 Ascendifier 接受的是 AST，可通过 ast.parse / ast.unparse 处理）
        ascend_source = ascendifier.transform(base_artifact.primary_source)

        # 3. 构建并返回封装了 Ascend 源码的 Artifact 对象
        return Artifact(
            kernel_name=base_artifact.kernel_name,
            entrypoint=base_artifact.entrypoint,
            primary_source=ascend_source,
            abi=base_artifact.abi,
            metadata=dict(base_artifact.metadata) | {"target": Target.ASCEND},
        )