from ninetoothed.backends.core import Backend, Target, Artifact
from ninetoothed.backends.emitters.ascend import AscendEmitter
from ninetoothed.ir import Kernel
from typing import Mapping, Any
from ninetoothed.compiler.passes import OptimizeSchedule

class AscendOptimizeSchedule(OptimizeSchedule):
    name = "ssa.ascend.optimize_schedule"
    supported_backends = (Target.ASCEND,)

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
    ):
        # 如果目前尚未针对 Ascend 编写特殊的调度分析逻辑，
        # 可以先直接返回空列表/默认逻辑（参照 tilelang 或 triton 中的基础实现）
        return []

def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.ASCEND,
        optimize_schedule=AscendOptimizeSchedule,
    )

class AscendBackend(Backend):
    target = Target.ASCEND  # 请确保 core.py 中有 ASCEND 枚举，没有的话需补充定义

    def normalize_options(self, options: dict) -> dict:
        return options

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        return kernel

    def emit(self, kernel: Kernel) -> Artifact:
        emitter = AscendEmitter()
        return emitter.emit(kernel)