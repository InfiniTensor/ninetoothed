"""Ascend source emission using the shared Triton syntax hooks."""

from dataclasses import dataclass, replace

from ninetoothed.ascendifier import Ascendifier
from ninetoothed.backends.core import Artifact, Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.ir import Kernel


@dataclass(frozen=True, kw_only=True)
class AscendEmitter(TritonTarget):
    backend: Target = Target.ASCEND
    suffix: str = "ascend_triton.py"
    source_route: str = "ssa-unified-ascend-emitter"

    def emit(self, kernel: Kernel) -> Artifact:
        artifact = common.emit(kernel, self)
        transformer = Ascendifier()

        return replace(
            artifact,
            sources={
                name: (
                    transformer.transform(source)
                    if name == artifact.primary_source_name
                    else source
                )
                for name, source in artifact.sources.items()
            },
        )
