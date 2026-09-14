"""Check declared result correspondence against independently known SSA faults."""

import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.frontend.python import from_source
from ninetoothed.interpreter.debugger import check_passes, compare_programs
from ninetoothed.interpreter.failure import replay_failure
from ninetoothed.ir import TensorSpec, ssa
from ninetoothed.ir.provenance import (
    ProvenancePass,
    operation_locations,
    record_pass,
    seed_origins,
)

from .test_interpreter_provenance import (
    _arithmetic,
    _inject_decomposed_fault,
    _real_pipeline_case,
    _value,
)
from .test_interpreter_provenance import (
    real_linalg_case as real_linalg_case,
)


def _mapped_arithmetic(program, *, mode, wrong, mapped=True):
    tracker = ProvenancePass(program, mode)
    zero, two, scale, output = program.blocks[0].operations

    if mode == "replace":
        targets = tracker.derive(
            (replace(scale, opcode="arith.add" if wrong else "arith.mul"),),
            (scale,),
        )
        source, target = scale, targets[0]
        operations = (zero, two, *targets, output)
    elif mode == "split":
        partial_value = _value("%partial")
        targets = tracker.derive(
            (
                ssa.Operation(
                    opcode="arith.add", operands=("x", "x"), results=(partial_value,)
                ),
                ssa.Operation(
                    opcode="arith.add",
                    operands=(partial_value.name, "%two" if wrong else "%zero"),
                    results=scale.results,
                ),
            ),
            (scale,),
            relation="split",
        )
        source, target = scale, targets[-1]
        operations = (zero, two, *targets, output)
    else:
        assert mode == "merge"
        # Fuse x*2+0 into x*2; the injected x+2 fails for several held-out inputs.
        targets = tracker.derive(
            (
                replace(
                    output,
                    opcode="arith.add" if wrong else "arith.mul",
                    operands=("x", "%two"),
                ),
            ),
            (scale, output),
            relation="merge",
        )
        source, target = output, targets[0]
        operations = (zero, two, *targets)

    if mapped:
        tracker.map_result(source, target)

    return tracker.finish(
        replace(program, blocks=(replace(program.blocks[0], operations=operations),))
    )


@pytest.mark.parametrize("mode", ("replace", "split", "merge"))
@pytest.mark.parametrize("wrong", (True, False), ids=("fault", "repair"))
@pytest.mark.parametrize("input_value", (3, 5, -2))
def test_restructured_results_locate_the_producer_and_replay(
    mode, wrong, input_value, tmp_path
):
    original = _arithmetic()

    if mode == "merge":
        block = original.blocks[0]
        last = replace(block.operations[-1], operands=("%scaled", "%zero"))
        original = replace(
            original,
            blocks=(replace(block, operations=(*block.operations[:-1], last)),),
        )

    report = check_passes(
        original,
        ((mode, partial(_mapped_arithmetic, mode=mode, wrong=wrong)),),
        {"x": np.int32(input_value)},
        failure_dir=tmp_path / "failure",
        seed=42,
    )
    assert report.error is None
    assert report.passed is not wrong

    if not wrong:
        assert report.localization is None
        assert not (tmp_path / "failure").exists()

        return

    assert report.first_bad_pass == mode
    location = report.localization
    assert location.basis == "mapped_result"
    assert location.operation.component == "result"
    assert (
        location.operation.location
        == {
            "replace": "entry:2:arith.add",
            "split": "entry:3:arith.add",
            "merge": "entry:2:arith.add",
        }[mode]
    )
    assert (
        location.reference_location
        == {
            "replace": "entry:2:arith.mul",
            "split": "entry:2:arith.mul",
            "merge": "entry:3:arith.add",
        }[mode]
    )
    assert report.dependency_slice.events[-1].location == location.operation.location
    assert report.export_error is None
    assert replay_failure(report.reproducer) is not None


def test_same_names_and_origins_do_not_create_a_value_mapping():
    original = seed_origins(_arithmetic())
    candidate = _mapped_arithmetic(original, mode="split", wrong=True, mapped=False)
    comparison = compare_programs(original, candidate, {"x": np.int32(3)})
    assert not comparison.equal
    assert comparison.mapped_operation is None
    assert comparison.retained_operation.operation.component == "input"


def test_a_stale_program_fingerprint_disables_result_alignment():
    original = seed_origins(_arithmetic())
    candidate = _mapped_arithmetic(original, mode="split", wrong=True)
    operations = list(candidate.blocks[0].operations)
    operations[1] = replace(operations[1], attrs={"value": 4})
    candidate = replace(
        candidate, blocks=(replace(candidate.blocks[0], operations=tuple(operations)),)
    )
    comparison = compare_programs(original, candidate, {"x": np.int32(3)})
    assert not comparison.equal
    assert comparison.mapped_operation is None
    assert comparison.mapping_issues


@pytest.mark.parametrize("problem", ("duplicate", "missing_result", "wrong_type"))
def test_invalid_result_contracts_are_rejected(problem):
    original = seed_origins(_arithmetic())
    tracker = ProvenancePass(original, "invalid")
    zero, two, scale, output = original.blocks[0].operations
    replacement = replace(scale, opcode="arith.add")

    if problem == "wrong_type":
        replacement = replace(replacement, results=(_value("%scaled", "float32"),))

    target = tracker.derive((replacement,), (scale,))[0]

    with pytest.raises(ValueError):
        if problem == "duplicate":
            tracker.map_result(scale, target)

        tracker.map_result(
            scale,
            target,
            source_result="%missing" if problem == "missing_result" else None,
        )
        tracker.finish(
            replace(
                original,
                blocks=(
                    replace(original.blocks[0], operations=(zero, two, target, output)),
                ),
            )
        )


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("wrong", (True, False), ids=("fault", "repair"))
def test_actual_lowered_dot_and_transpose_find_the_changed_instruction(
    backend, wrong, real_linalg_case, tmp_path
):
    kind, kernel, pipeline, context, inputs, _expected, options = _real_pipeline_case(
        backend, real_linalg_case
    )
    observed = []
    injected_name = f"diagnostic.mapped_{kind}"

    def inject(program):
        candidate = _inject_decomposed_fault(
            program, kind=kind, name=injected_name, wrong=wrong, map_values=True
        )
        observed.append(candidate)

        return candidate

    checks = []

    for pass_ in pipeline.passes:
        checks.append((pass_.name, partial(pass_.run, context=context)))

        if pass_.name == "ssa.decompose_linalg":
            checks.append((injected_name, inject))

    report = check_passes(
        kernel.frontend_program,
        checks,
        inputs,
        **options,
        seed=2026,
        failure_dir=tmp_path / "failure",
    )
    assert report.error is None
    assert report.passed is not wrong

    if wrong:
        assert report.first_bad_pass == injected_name
        target = next(
            location
            for location, operation in operation_locations(observed[0])
            if operation.opcode == ("arith.add" if kind == "dot" else "tensor.extract")
            and operation.attrs.get("decomposition")
            == ("matmul" if kind == "dot" else "transpose")
            and (
                kind != "dot"
                or "operand" not in operation.attrs
                and len(operation.operands) == 2
                and "%acc_iter" not in operation.operands
            )
        )
        assert report.localization.basis == "mapped_result"
        assert report.localization.operation.location == target
        assert report.localization.operation.lane is not None
        assert report.export_error is None
        metadata = json.loads((report.reproducer / "failure.json").read_text())
        assert metadata["report"]["dependency_slice"]["events"]
        replay_failure(report.reproducer)


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("wrong", (True, False), ids=("fault", "repair"))
def test_actual_decomposition_checks_tile_and_internal_results(
    backend, wrong, real_linalg_case, monkeypatch
):
    kind, kernel, pipeline, context, inputs, _expected, options = _real_pipeline_case(
        backend, real_linalg_case
    )
    original_derive = ProvenancePass.derive

    def faulty_derive(self, targets, sources, **kwargs):
        def mutate(operation):
            regions = tuple(
                replace(
                    region, operations=tuple(mutate(op) for op in region.operations)
                )
                for region in operation.regions
            )
            result = replace(operation, regions=regions)

            if wrong and operation.opcode == (
                "arith.mul" if kind == "dot" else "tensor.extract"
            ):
                if kind == "dot":
                    result = replace(result, opcode="arith.add")
                else:
                    tensor, row, column = operation.operands
                    result = replace(result, operands=(tensor, column, row))
            return result

        if self.name == "ssa.decompose_linalg" and kwargs.get("relation") == "split":
            targets = tuple(mutate(op) for op in targets)
        return original_derive(self, targets, sources, **kwargs)

    monkeypatch.setattr(ProvenancePass, "derive", faulty_derive)
    current = seed_origins(kernel.frontend_program)

    for pass_ in pipeline.passes:
        previous = current
        current = record_pass(previous, pass_.run(previous, context), pass_.name)

        if pass_.name == "ssa.decompose_linalg":
            break

    comparison = compare_programs(previous, current, inputs, **options)
    assert comparison.equal is not wrong
    assert comparison.mapping_issues == ()

    if wrong:
        assert comparison.localization.basis == "mapped_result"
        assert comparison.localization.operation.opcode == (
            "arith.add" if kind == "dot" else "tensor.extract"
        )
        assert comparison.localization.operation.lane is not None
        assert comparison.dependency_slice.events
    else:
        assert comparison.mapped_operation is None
        assert comparison.localization is None


def _loop_program():
    zero, stop, step, two, induction, carried, term, total, result = map(
        _value,
        (
            "%zero",
            "%stop",
            "%step",
            "%two",
            "%i",
            "%carried",
            "%term",
            "%total",
            "%result",
        ),
    )
    constants = tuple(
        ssa.Operation(
            opcode="arith.constant", results=(value,), attrs={"value": number}
        )
        for value, number in ((zero, 0), (stop, 3), (step, 1), (two, 2))
    )
    loop = ssa.Operation(
        opcode="scf.for",
        operands=(zero.name, stop.name, step.name, zero.name),
        results=(result,),
        regions=(
            ssa.Block(
                args=(induction, carried),
                operations=(
                    ssa.Operation(
                        opcode="arith.mul",
                        operands=(induction.name, two.name),
                        results=(term,),
                    ),
                    ssa.Operation(
                        opcode="arith.add",
                        operands=(carried.name, term.name),
                        results=(total,),
                    ),
                    ssa.Operation(opcode="scf.yield", operands=(total.name,)),
                ),
            ),
        ),
    )

    return ssa.Program(
        kind="mapped_loop",
        outputs=(result,),
        blocks=(ssa.Block(operations=(*constants, loop)),),
    )


@pytest.mark.parametrize("mapping", ("inner", "loop"))
def test_loop_mapping_and_dynamic_carried_dependencies(mapping):
    def change(program):
        tracker = ProvenancePass(program, "loop_fault")
        block = program.blocks[0]
        old = block.operations[-1]
        region = old.regions[0]
        term = region.operations[0]
        new_term = tracker.derive((replace(term, opcode="arith.add"),), (term,))[0]
        remaining = region.operations[1:]

        if mapping == "loop":
            remaining = tuple(
                tracker.derive((replace(op),), (op,))[0] for op in remaining
            )

        new_region = replace(region, operations=(new_term, *remaining))
        new_loop = tracker.derive((replace(old, regions=(new_region,)),), (old,))[0]
        tracker.map_result(
            term, new_term
        ) if mapping == "inner" else tracker.map_result(old, new_loop)

        return tracker.finish(
            replace(
                program,
                blocks=(replace(block, operations=(*block.operations[:-1], new_loop)),),
            )
        )

    report = check_passes(_loop_program(), (("loop_fault", change),), {})
    assert report.first_bad_pass == "loop_fault"
    assert report.localization.basis == "mapped_result"

    if mapping == "inner":
        assert report.localization.operation.iteration == (0,)
        assert report.localization.operation.opcode == "arith.add"
    else:
        assert report.localization.operation.opcode == "scf.for"
        terms = [
            event
            for event in report.dependency_slice.events
            if event.location.endswith("/region0:0:arith.add")
        ]
        assert [event.iteration for event in terms] == [(0,), (1,), (2,)]

    assert report.dependency_slice.boundaries == ()


def test_mapping_order_cannot_hide_an_earlier_bad_producer():
    def change(program):
        tracker = ProvenancePass(program, "two_faults")
        zero, two, scale, output = program.blocks[0].operations
        new_two = tracker.derive((replace(two, attrs={"value": 3}),), (two,))[0]
        new_scale = tracker.derive((replace(scale, opcode="arith.add"),), (scale,))[0]
        tracker.map_result(scale, new_scale)
        tracker.map_result(two, new_two)

        return tracker.finish(
            replace(
                program,
                blocks=(
                    replace(
                        program.blocks[0], operations=(zero, new_two, new_scale, output)
                    ),
                ),
            )
        )

    report = check_passes(_arithmetic(), (("two_faults", change),), {"x": np.int32(5)})
    assert report.localization.operation.location == "entry:1:arith.constant"


def test_result_renaming_needs_an_explicit_contract():
    original = seed_origins(_arithmetic())
    tracker = ProvenancePass(original, "rename")
    zero, two, scale, output = original.blocks[0].operations
    value = _value("%renamed")
    new_scale = tracker.derive(
        (replace(scale, opcode="arith.add", results=(value,)),), (scale,)
    )[0]
    new_output = tracker.derive(
        (replace(output, operands=(value.name, "%two")),), (output,)
    )[0]
    tracker.map_result(scale, new_scale)
    candidate = tracker.finish(
        replace(
            original,
            blocks=(
                replace(
                    original.blocks[0], operations=(zero, two, new_scale, new_output)
                ),
            ),
        )
    )
    comparison = compare_programs(original, candidate, {"x": np.int32(3)})
    assert comparison.localization.operation.result_name == "%renamed"
    assert comparison.localization.reference_location == "entry:2:arith.mul"


@pytest.mark.parametrize("change_header", (False, True))
def test_different_loop_instances_are_not_paired(change_header):
    original = seed_origins(_loop_program())
    tracker = ProvenancePass(original, "different_iterations")
    constants = list(original.blocks[0].operations[:-1])
    old = original.blocks[0].operations[-1]
    region = old.regions[0]
    term = region.operations[0]
    new_term = tracker.derive((replace(term, opcode="arith.add"),), (term,))[0]

    if not change_header:
        constants[1] = tracker.derive(
            (replace(constants[1], attrs={"value": 4}),), (constants[1],)
        )[0]

    operands = ("%zero", "%two", "%step", "%zero") if change_header else old.operands
    new_loop = tracker.derive(
        (
            replace(
                old,
                operands=operands,
                regions=(
                    replace(region, operations=(new_term, *region.operations[1:])),
                ),
            ),
        ),
        (old,),
    )[0]
    tracker.map_result(term, new_term)
    candidate = tracker.finish(
        replace(
            original,
            blocks=(replace(original.blocks[0], operations=(*constants, new_loop)),),
        )
    )
    comparison = compare_programs(original, candidate, {})
    assert not comparison.equal
    assert comparison.mapped_operation is None
    assert comparison.mapping_issues


@pytest.mark.parametrize("problem", ("wrong_source", "missing_target"))
def test_result_mapping_requires_the_recorded_producers(problem):
    original = seed_origins(_arithmetic())
    tracker = ProvenancePass(original, "bad_contract")
    zero, two, scale, output = original.blocks[0].operations
    target = tracker.derive((replace(scale, opcode="arith.add"),), (scale,))[0]
    tracker.map_result(two if problem == "wrong_source" else scale, target)

    with pytest.raises(ValueError, match="explicit operation relation|occur once"):
        tracker.finish(
            replace(
                original,
                blocks=(
                    replace(
                        original.blocks[0],
                        operations=(
                            zero,
                            two,
                            scale if problem == "missing_target" else target,
                            output,
                        ),
                    ),
                ),
            )
        )


def test_one_reference_value_can_explicitly_feed_several_generated_results():
    original = seed_origins(_arithmetic())
    tracker = ProvenancePass(original, "duplicate_value")
    zero, two, scale, output = original.blocks[0].operations
    targets = tracker.derive(
        (replace(two), replace(two, results=(_value("%copy"),))),
        (two,),
        relation="split",
    )

    for target in targets:
        tracker.map_result(two, target)

    candidate = tracker.finish(
        replace(
            original,
            blocks=(
                replace(original.blocks[0], operations=(zero, *targets, scale, output)),
            ),
        )
    )
    comparison = compare_programs(original, candidate, {"x": np.int32(5)})
    assert comparison.equal
    assert comparison.mapping_issues == ()
    assert comparison.mapped_operation is None


@pytest.mark.parametrize(
    "field", ("mapped_operation", "dependency_slice", "pass_localization")
)
def test_replay_checks_new_localization_and_dependency_evidence(field, tmp_path):
    report = check_passes(
        _arithmetic(),
        (("split", partial(_mapped_arithmetic, mode="split", wrong=True)),),
        {"x": np.int32(3)},
        failure_dir=tmp_path / "failure",
    )
    metadata_path = report.reproducer / "failure.json"
    metadata = json.loads(metadata_path.read_text())

    if field == "pass_localization":
        metadata["report"]["localization"] = None
    else:
        metadata["report"]["adjacent_difference"][field] = None

    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="no longer reproduces"):
        replay_failure(report.reproducer)


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize("output_dtype", ("float16", "float64"))
def test_debug_contracts_preserve_mixed_dtype_lowering(
    backend, output_dtype, monkeypatch
):
    tensors = (
        TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
        TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype=output_dtype, name="out"),
    )
    program = from_source(
        "def matmul(a, b, out):\n    out = a @ b\n", tensors, kind="matmul"
    )
    actual = lower_for_target(program, backend=backend)

    with monkeypatch.context() as context:
        context.setattr(ProvenancePass, "map_result", lambda *args, **kwargs: None)
        without_debug_contract = lower_for_target(program, backend=backend)

    assert ssa.render(actual) == ssa.render(without_debug_contract)
    assert not actual.metadata["provenance"]["passes"][-1].get("value_mappings")
