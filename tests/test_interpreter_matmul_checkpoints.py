"""Check default decomposition intermediates against independent input formulas."""

from dataclasses import replace

import numpy as np
import pytest

from ninetoothed.interpreter.debugger import compare_programs
from ninetoothed.ir.provenance import ProvenancePass, record_pass, seed_origins

from .test_interpreter_matmul import _tensors, matrix_dot, matrix_tiles
from .test_interpreter_provenance import _real_pipeline_case


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize(
    "fault", ("lhs", "rhs", "product", "accumulator", "step", "none")
)
@pytest.mark.parametrize("dtype", (np.float32, np.int32), ids=("float32", "int32"))
def test_default_matmul_pass_locates_a_bad_term_or_partial_sum(
    backend, fault, dtype, monkeypatch, tmp_path
):
    a = (np.arange(21).reshape(7, 3) + 3).astype(dtype)
    b = (np.arange(18).reshape(3, 6) + 5).astype(dtype)
    inputs = {"a": a, "b": b, "out": np.full((7, 6), -731, dtype=dtype)}
    case = ("dot", matrix_tiles, matrix_dot, _tensors(dtype), inputs, a @ b)
    _, kernel, pipeline, context, inputs, _, options = _real_pipeline_case(
        backend, case
    )
    derive = ProvenancePass.derive

    def faulty(self, targets, sources, **kwargs):
        def mutate(op):
            regions = tuple(
                replace(
                    region,
                    operations=tuple(mutate(child) for child in region.operations),
                )
                for region in op.regions
            )
            op = replace(op, regions=regions)

            if op.attrs.get("decomposition") == "matmul":
                if fault == "product" and op.opcode == "arith.mul":
                    op = replace(op, opcode="arith.add")
                elif (
                    fault == "accumulator"
                    and op.opcode == "arith.add"
                    and "%acc_iter" in op.operands
                ):
                    op = replace(op, opcode="arith.sub")
                elif (
                    fault in {"lhs", "rhs"}
                    and op.opcode == "tensor.extract"
                    and op.attrs.get("operand") == fault
                ):
                    tensor, first, second = op.operands
                    op = replace(op, operands=(tensor, second, first))
                elif (
                    fault == "step"
                    and op.opcode == "arith.constant"
                    and op.attrs.get("value") == 1
                ):
                    op = replace(op, attrs=dict(op.attrs) | {"value": 2})
            return op

        if self.name == "ssa.decompose_linalg" and kwargs.get("relation") == "split":
            targets = tuple(mutate(op) for op in targets)
        return derive(self, targets, sources, **kwargs)

    monkeypatch.setattr(ProvenancePass, "derive", faulty)
    current = seed_origins(kernel.frontend_program)

    for pass_ in pipeline.passes:
        previous = current
        current = record_pass(previous, pass_.run(previous, context), pass_.name)

        if pass_.name == "ssa.decompose_linalg":
            break

    mappings = current.metadata["provenance"]["passes"][-1]["value_mappings"]
    assert {item["projection"] for item in mappings} >= {
        "lane",
        "matmul_term",
        "matmul_prefix",
    }
    report = compare_programs(
        previous, current, inputs, **options, failure_dir=tmp_path / "failure"
    )
    assert report.equal is (fault == "none")

    if fault == "step":
        assert report.localization.operation.opcode == "scf.for"
        assert report.localization.projection == "lane"
        assert any(
            "iterations must cover K" in issue for issue in report.mapping_issues
        )

        return

    assert report.mapping_issues == ()

    if fault != "none":
        assert report.localization.basis == "mapped_result"
        assert (
            report.localization.operation.opcode
            == {
                "lhs": "tensor.extract",
                "rhs": "tensor.extract",
                "product": "arith.add",
                "accumulator": "arith.sub",
            }[fault]
        )
        assert (
            report.localization.projection
            == {
                "lhs": "matmul_lhs",
                "rhs": "matmul_rhs",
                "product": "matmul_term",
                "accumulator": "matmul_prefix",
            }[fault]
        )
        assert report.localization.operation.iteration == (
            (1,) if fault in {"lhs", "rhs"} else (0,)
        )
        assert report.localization.operation.lane == (0, 0)
        from ninetoothed.interpreter.failure import replay_failure

        replay_failure(report.reproducer)
