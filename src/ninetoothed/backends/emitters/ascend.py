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
        transformer = Ascendifier(omit_static_bounds=_has_static_tensor_shapes(kernel))

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


def _has_static_tensor_shapes(kernel: Kernel) -> bool:
    saw_tensor = False

    for tensor in kernel.tensors:
        source_shape = tensor.attrs.get("source_shape", ())

        if not source_shape:
            continue

        saw_tensor = True

        if any(_static_positive_integer(value) is None for value in source_shape):
            return False

    return saw_tensor


def _static_positive_integer(value) -> int | None:
    try:
        extent = int(value)
    except (TypeError, ValueError):
        return None

    return extent if extent > 0 else None
