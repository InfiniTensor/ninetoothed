"""CUDA accumulation modes retain the same type and complete-region guards."""

import functools
import re

from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from tests.test_ascendc_dot_plan import _matmul
from tests.test_matmul import arrangement


def _compile(application=_matmul, *, backend_options=None):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
            ),
            application=application,
            tensors=(
                Tensor(shape=(64, 64), dtype="float16", other=0),
                Tensor(shape=(64, 64), dtype="float16", other=0),
                Tensor(shape=(64, 64), dtype="float32", other=0),
            ),
            backend="cuda",
            backend_options=backend_options,
        )
    )


def test_ordered_resets_each_source_dot_group_and_adds_float_carry():
    artifact = _compile(backend_options={"cuda_wmma_accumulation": "ordered"}).artifact
    source = artifact.primary_source
    assert "wmma_reduction_carry" in source
    assert "__fadd_rn" in source
    reset = re.search(r"wmma::fill_fragment\([^,]+_acc, 0\.0f\)", source)
    assert reset is not None
    outer = source.rfind("for (", 0, reset.start())
    assert outer >= 0
    assert reset.start() < source.index("wmma::mma_sync", outer)
    assert "nt_wmma_component" in source
    metadata = artifact.metadata["cuda_wmma_accumulation"]
    assert metadata["mode"] == "ordered"
    assert metadata["accumulator_dtype"] == "float32"
    assert metadata["rounding"] == "per-source-dot-partial-then-ordered-f32-add-rn"
    assert metadata["source_dot_add_groups_preserved"] is True
