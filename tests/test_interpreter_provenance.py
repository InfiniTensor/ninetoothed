"""CPU checks for conservative trace alignment and explicit SSA provenance."""

import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from ninetoothed import interpret
from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import Context, default_pipeline
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import (
    check_passes,
    compare_programs,
    export_reproducer,
    load_reproducer,
)
from ninetoothed.ir import ir_to_dict, ssa
from ninetoothed.ir.provenance import (
    ProvenancePass,
    operation_locations,
    record_pass,
    seed_origins,
    source_candidates,
)

from .test_interpreter_applications import _descriptor, _transpose, _transpose_tiles
from .test_interpreter_matmul import _operands, _tensors, matrix_dot, matrix_tiles


def _value(name, dtype="int32"):
    return ssa.Value(name=name, type=ssa.Type(kind="scalar", dtype=dtype))


def _branch_case():
    condition, output = _value("%condition", "bool"), _value("%output")
    regions = []

    for index, literal in enumerate((1, 2)):
        value = _value(f"%branch{index}")
        regions.append(
            ssa.Block(
                name=f"branch{index}",
                operations=(
                    ssa.Operation(
                        opcode="arith.constant",
                        results=(value,),
                        attrs={"value": literal},
                    ),
                    ssa.Operation(opcode="scf.yield", operands=(value.name,)),
                ),
            )
        )

    block = ssa.Block(
        operations=(
            ssa.Operation(
                opcode="arith.constant", results=(condition,), attrs={"value": True}
            ),
            ssa.Operation(
                opcode="scf.if",
                operands=(condition.name,),
                results=(output,),
                regions=tuple(regions),
            ),
        )
    )
    reference = ssa.Program(
        kind="branch_provenance", outputs=(output,), blocks=(block,)
    )
    candidate = replace(
        reference,
        blocks=(
            replace(
                block,
                operations=(
                    replace(block.operations[0], attrs={"value": False}),
                    block.operations[1],
                ),
            ),
        ),
    )

    return reference, candidate


def test_branch_divergence_after_an_early_difference_does_not_claim_aligned_traces():
    reference, candidate = _branch_case()
    first = interpret_program(reference, {}, trace=True)
    second = interpret_program(candidate, {}, trace=True)
    assert first.outputs["%output"] == 1
    assert second.outputs["%output"] == 2
    assert len(first.trace) == len(second.trace)
    assert [event.location for event in first.trace] != [
        event.location for event in second.trace
    ]
    difference = compare_programs(reference, candidate, {})
    assert not difference.equal
    assert not difference.traces_aligned
    assert difference.first_operation is None


def test_changed_loop_count_does_not_claim_aligned_traces_after_bound_difference():
    zero, one, bound, induction, carried, updated, output = map(
        _value, ("%zero", "%one", "%bound", "%i", "%acc", "%updated", "%output")
    )
    body = ssa.Block(
        name="loop",
        args=(induction, carried),
        operations=(
            ssa.Operation(
                opcode="arith.add",
                operands=(carried.name, one.name),
                results=(updated,),
            ),
            ssa.Operation(opcode="scf.yield", operands=(updated.name,)),
        ),
    )
    block = ssa.Block(
        operations=(
            ssa.Operation(opcode="arith.constant", results=(zero,), attrs={"value": 0}),
            ssa.Operation(opcode="arith.constant", results=(one,), attrs={"value": 1}),
            ssa.Operation(
                opcode="arith.constant", results=(bound,), attrs={"value": 3}
            ),
            ssa.Operation(
                opcode="scf.for",
                operands=(zero.name, bound.name, one.name, zero.name),
                results=(output,),
                regions=(body,),
            ),
        )
    )
    reference = ssa.Program(kind="loop_provenance", outputs=(output,), blocks=(block,))
    candidate = replace(
        reference,
        blocks=(
            replace(
                block,
                operations=(
                    *block.operations[:2],
                    replace(block.operations[2], attrs={"value": 4}),
                    block.operations[3],
                ),
            ),
        ),
    )
    assert interpret_program(reference, {}).outputs["%output"] == 3
    assert interpret_program(candidate, {}).outputs["%output"] == 4
    difference = compare_programs(reference, candidate, {})
    assert not difference.equal
    assert not difference.traces_aligned
    assert difference.first_operation is None


def _arithmetic():
    x, zero, two, scaled, output = map(
        _value, ("x", "%zero", "%two", "%scaled", "%output")
    )

    return ssa.Program(
        kind="arithmetic_provenance",
        inputs=(x,),
        outputs=(output,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(zero,), attrs={"value": 0}
                    ),
                    ssa.Operation(
                        opcode="arith.constant", results=(two,), attrs={"value": 2}
                    ),
                    ssa.Operation(
                        opcode="arith.mul",
                        operands=(x.name, two.name),
                        results=(scaled,),
                    ),
                    ssa.Operation(
                        opcode="arith.add",
                        operands=(scaled.name, two.name),
                        results=(output,),
                    ),
                )
            ),
        ),
    )


def _split_scale(program, *, wrong=False):
    tracker = ProvenancePass(program, "split_scale")
    zero, two, scale, output = program.blocks[0].operations
    partial = _value("%partial")
    targets = tracker.derive(
        (
            ssa.Operation(opcode="arith.add", operands=("x", "x"), results=(partial,)),
            ssa.Operation(
                opcode="arith.add",
                operands=(partial.name, "%two" if wrong else "%zero"),
                results=scale.results,
            ),
        ),
        (scale,),
        relation="split",
    )

    return tracker.finish(
        replace(
            program,
            blocks=(
                replace(program.blocks[0], operations=(zero, two, *targets, output)),
            ),
        )
    )


def test_seed_is_stable_nonsemantic_and_covers_nested_ssa_locations():
    program, _candidate = _branch_case()
    seeded = seed_origins(program)
    assert seeded == seed_origins(program)
    assert seed_origins(seeded) is seeded
    assert ssa.render(seeded) == ssa.render(program)
    assert seeded.blocks == program.blocks
    sources = seeded.metadata["provenance"]["sources"]
    assert len(sources) == len(operation_locations(program)) == 6
    assert any("/region1:" in source["location"] for source in sources.values())
    assert all(
        sources[operation.origins[0]]["location"] == location
        for location, operation in operation_locations(seeded)
    )


def test_declared_split_has_source_range_but_never_guesses_a_value_correspondence():
    reference = seed_origins(_arithmetic())
    candidate = _split_scale(reference, wrong=True)
    inputs = {"x": np.int32(3)}
    assert interpret_program(reference, inputs).outputs["%output"] == 8
    assert interpret_program(candidate, inputs).outputs["%output"] == 10
    comparison = compare_programs(reference, candidate, inputs)
    assert not comparison.equal
    assert not comparison.traces_aligned
    assert comparison.first_operation is None
    assert len(comparison.source_candidates) == 1
    assert comparison.source_candidates[0].location == "entry:2:arith.mul"
    assert comparison.source_candidates[0].result_names == ("%scaled",)
    # Removing the injected error, while retaining the split, is a real negative control.
    repaired = _split_scale(reference)
    assert compare_programs(reference, repaired, inputs).equal
    assert compare_programs(reference, repaired, inputs).source_candidates == ()


def test_pass_checker_preserves_an_explicit_mapping_and_stops_on_the_fault():
    result = check_passes(
        _arithmetic(),
        (
            ("identity", lambda program: program),
            ("split_scale", lambda program: _split_scale(program, wrong=True)),
        ),
        {"x": np.int32(3)},
    )
    assert not result.passed
    assert result.first_bad_pass == "split_scale"
    assert result.checked_passes == ("identity", "split_scale")
    assert result.difference.first_operation is None
    assert result.difference.source_candidates[0].opcode == "arith.mul"


def test_multiple_passes_keep_original_sources_and_current_relation_locations():
    original = seed_origins(_arithmetic())
    before = _split_scale(original)
    tracker = ProvenancePass(before, "inject_offset")
    operations = list(before.blocks[0].operations)
    old = operations[3]
    operations[3] = tracker.derive(
        (replace(old, operands=("%partial", "%two")),), (old,)
    )[0]
    after = tracker.finish(
        replace(
            before, blocks=(replace(before.blocks[0], operations=tuple(operations)),)
        )
    )
    assert (
        after.metadata["provenance"]["namespace"]
        == original.metadata["provenance"]["namespace"]
    )
    assert len(after.metadata["provenance"]["passes"]) == 2
    relation = after.metadata["provenance"]["passes"][-1]["relations"][0]
    assert relation["inputs"] == ("entry:3:arith.add",)
    candidates = source_candidates(after, ("entry:3:arith.add",))
    assert candidates[0].location == "entry:2:arith.mul"
    difference = compare_programs(before, after, {"x": np.int32(3)})
    # Changing an operand also changes the SSA structure; provenance cannot align it.
    assert difference.first_operation is None
    assert difference.source_candidates == candidates


def test_merge_retains_both_candidates_without_claiming_a_unique_fault_source():
    before = seed_origins(_arithmetic())
    zero, two, scale, output = before.blocks[0].operations
    tracker = ProvenancePass(before, "fuse_wrong")
    fused = tracker.derive(
        (replace(output, operands=("x", "%two")),), (scale, output), relation="merge"
    )
    after = tracker.finish(
        replace(
            before, blocks=(replace(before.blocks[0], operations=(zero, two, *fused)),)
        )
    )
    comparison = compare_programs(before, after, {"x": np.int32(3)})
    assert not comparison.equal
    assert comparison.first_operation is None
    assert {source.location for source in comparison.source_candidates} == {
        "entry:2:arith.mul",
        "entry:3:arith.add",
    }


def test_explicit_deletion_is_distinct_from_an_unmapped_removal():
    before = seed_origins(_arithmetic())
    dead, *remaining = before.blocks[0].operations
    after = replace(
        before, blocks=(replace(before.blocks[0], operations=tuple(remaining)),)
    )
    tracker = ProvenancePass(before, "dead_constant")
    tracker.delete(dead)
    declared = tracker.finish(after)
    assert compare_programs(before, declared, {"x": np.int32(3)}).equal
    assert source_candidates(declared)[0].location == "entry:0:arith.constant"
    unknown = record_pass(before, after, "unknown_deletion")
    assert source_candidates(unknown) == ()
    assert any(
        record["relation"] == "unmapped_removal"
        for record in unknown.metadata["provenance"]["passes"][-1]["relations"]
    )


def test_deleting_mixed_known_unknown_sources_does_not_present_a_partial_candidate_set():
    out = ssa.Value(
        name="out", type=ssa.Type(kind="tensor", shape=("1",), dtype="int32")
    )
    one, unused = _value("%one"), _value("%unused")
    original = seed_origins(
        ssa.Program(
            kind="partial_deletion_sources",
            inputs=(out,),
            outputs=(out,),
            blocks=(
                ssa.Block(
                    operations=(
                        ssa.Operation(
                            opcode="arith.constant", results=(one,), attrs={"value": 1}
                        ),
                        ssa.Operation(
                            opcode="arith.constant",
                            results=(unused,),
                            attrs={"value": 42},
                        ),
                        ssa.Operation(
                            opcode="mem.store", operands=(one.name, out.name)
                        ),
                    )
                ),
            ),
        )
    )
    first, dead, store = original.blocks[0].operations
    before = record_pass(
        original,
        replace(
            original,
            blocks=(
                replace(original.blocks[0], operations=(first, dead, replace(store))),
            ),
        ),
        "unknown_store_clone",
    )
    first, dead, unknown_store = before.blocks[0].operations
    assert dead.origins
    assert not unknown_store.origins
    tracker = ProvenancePass(before, "delete_store_and_dead_constant")
    tracker.delete(unknown_store, dead)
    after = tracker.finish(
        replace(before, blocks=(replace(before.blocks[0], operations=(first,)),))
    )
    np.testing.assert_array_equal(
        interpret_program(before, {"out": np.zeros(1, dtype=np.int32)}).outputs["out"],
        [1],
    )
    np.testing.assert_array_equal(
        interpret_program(after, {"out": np.zeros(1, dtype=np.int32)}).outputs["out"],
        [0],
    )
    comparison = compare_programs(before, after, {"out": np.zeros(1, dtype=np.int32)})
    assert not comparison.equal
    assert comparison.first_operation is None
    # The known unused constant is not a complete source range for the lost write.
    assert comparison.source_candidates == ()
    deletion = after.metadata["provenance"]["passes"][-1]["relations"][0]
    assert deletion["relation"] == "delete"
    assert deletion["inputs"] == ("entry:2:mem.store", "entry:1:arith.constant")
    assert deletion["outputs"] == ()
    assert deletion["origins"] == ()


def test_unknown_clone_does_not_acquire_a_mapping_from_copied_ids_or_names():
    before = seed_origins(_arithmetic())
    block = before.blocks[0]
    after = replace(
        before,
        blocks=(
            replace(
                block,
                operations=tuple(replace(operation) for operation in block.operations),
            ),
        ),
    )
    recorded = record_pass(before, after, "unknown_clone")
    assert compare_programs(before, recorded, {"x": np.int32(3)}).equal
    assert all(
        not operation.origins for _location, operation in operation_locations(recorded)
    )
    assert source_candidates(recorded) == ()
    assert (
        seed_origins(recorded) is recorded
    )  # Never re-label unknown generated nodes as originals.


def test_explicit_record_is_not_duplicated_by_a_pipeline_wrapper():
    before = seed_origins(_arithmetic())
    after = _split_scale(before)
    assert record_pass(before, after, "split_scale") is after
    assert len(after.metadata["provenance"]["passes"]) == 1


def test_declared_targets_must_really_occur_and_deleted_sources_must_be_absent():
    before = seed_origins(_arithmetic())
    source = before.blocks[0].operations[0]
    tracker = ProvenancePass(before, "missing_target")
    tracker.derive((replace(source),), (source,))

    with pytest.raises(ValueError, match="declared target"):
        tracker.finish(before)

    tracker = ProvenancePass(before, "false_deletion")
    tracker.delete(source)

    with pytest.raises(ValueError, match="still present"):
        tracker.finish(before)


def test_recursive_split_includes_generated_region_operations():
    before, _candidate = _branch_case()
    before = seed_origins(before)
    condition, conditional = before.blocks[0].operations
    tracker = ProvenancePass(before, "rebuild_conditional")
    generated = tracker.derive(
        (conditional,), (conditional,), relation="split", recursive=True
    )
    after = tracker.finish(
        replace(
            before,
            blocks=(replace(before.blocks[0], operations=(condition, *generated)),),
        )
    )
    assert compare_programs(before, after, {}).equal
    nested = [
        (location, operation)
        for location, operation in operation_locations(after)
        if "/region" in location
    ]
    assert len(nested) == 4
    assert all(
        operation.origins == conditional.origins for _location, operation in nested
    )
    sources = source_candidates(
        after, tuple(location for location, _operation in nested)
    )
    assert len(sources) == 1
    assert sources[0].location == "entry:1:scf.if"


def test_runtime_errors_retain_declared_source_scope_without_a_fake_exact_location():
    def unsupported(program):
        tracker = ProvenancePass(program, "unsupported")
        operations = list(program.blocks[0].operations)
        source = operations[2]
        operations[2] = tracker.derive(
            (replace(source, opcode="test.unsupported"),), (source,)
        )[0]

        return tracker.finish(
            replace(
                program,
                blocks=(replace(program.blocks[0], operations=tuple(operations)),),
            )
        )

    report = check_passes(
        _arithmetic(), (("unsupported", unsupported),), {"x": np.int32(3)}
    )
    assert not report.passed
    assert report.first_bad_pass == "unsupported"
    assert "test.unsupported" in report.error
    assert report.difference is None
    assert report.source_candidates[0].location == "entry:2:arith.mul"


def test_one_operation_object_duplicated_without_a_declaration_is_unmapped():
    before = seed_origins(_arithmetic())
    operations = before.blocks[0].operations
    terminator = ssa.Operation(opcode="scf.yield")
    # Use a no-result operation so duplicate definitions do not obscure the map test.
    before = record_pass(
        before,
        replace(
            before,
            blocks=(replace(before.blocks[0], operations=(*operations, terminator)),),
        ),
        "insert_unknown",
    )
    duplicate = before.blocks[0].operations[-1]
    after = replace(
        before,
        blocks=(
            replace(
                before.blocks[0], operations=(*before.blocks[0].operations, duplicate)
            ),
        ),
    )
    result = record_pass(before, after, "duplicate_unknown")
    relations = result.metadata["provenance"]["passes"][-1]["relations"]
    assert [
        record["relation"]
        for record in relations
        if record["outputs"] and record["outputs"][0].endswith("scf.yield")
    ] == ["unmapped", "unmapped"]


def test_reproducer_preserves_provenance_and_legacy_json_remains_readable(tmp_path):
    candidate = _split_scale(seed_origins(_arithmetic()), wrong=True)
    directory = export_reproducer(tmp_path / "case", candidate, {"x": np.int32(3)})
    restored, inputs, options = load_reproducer(directory)
    assert ir_to_dict(restored) == ir_to_dict(candidate)
    assert source_candidates(restored) == source_candidates(candidate)
    assert interpret_program(restored, inputs, **options).outputs["%output"] == 10

    # Old schema-1 bundles predate both Operation.origins and provenance metadata.
    program_path = directory / "program.json"
    data = json.loads(program_path.read_text(encoding="utf-8"))

    for operation in data["blocks"][0]["operations"]:
        operation.pop("origins")

    data["metadata"].pop("provenance")
    program_path.write_text(json.dumps(data), encoding="utf-8")
    legacy, legacy_inputs, legacy_options = load_reproducer(directory)
    assert all(
        not operation.origins for _location, operation in operation_locations(legacy)
    )
    assert (
        interpret_program(legacy, legacy_inputs, **legacy_options).outputs["%output"]
        == 10
    )


@pytest.fixture(params=("dot", "transpose"))
def real_linalg_case(request):
    if request.param == "dot":
        a, b = _operands((7, 3, 6), np.float32, "contiguous")
        expected = a @ b
        inputs = {
            "a": a,
            "b": b,
            "out": np.full(expected.shape, -731, dtype=np.float32),
        }
        arrangement, application, tensors = (
            matrix_tiles,
            matrix_dot,
            _tensors(np.float32),
        )
    else:
        # A square, non-symmetric matrix keeps the deliberately swapped indices
        # in bounds, so the injected transpose fault produces a numeric mismatch.
        x = np.arange(9, dtype=np.float32).reshape(3, 3) / 7
        expected = x.T.copy()
        inputs = {"x": x, "out": np.full(expected.shape, -731, dtype=np.float32)}
        arrangement, application, tensors = (
            _transpose_tiles,
            _transpose,
            (_descriptor(2, "x"), _descriptor(2, "out")),
        )
    return request.param, arrangement, application, tensors, inputs, expected


def _real_pipeline_case(backend, case):
    kind, arrangement, application, tensors, inputs, expected = case
    kernel = interpret(arrangement, application, tensors, backend=backend)
    pipeline = default_pipeline(backend)
    context = Context(
        backend=Target(backend),
        compiler_options={},
        kernel_metadata={},
        tensors=kernel.tensors,
        pass_options=pipeline.spec.pass_options,
        pipeline_spec=pipeline.spec,
    )
    options = {"tensors": kernel.tensors, "symbols": kernel.meta}
    # Neither execution is the oracle for the other: check the original frontend
    # and actual default-pipeline result independently against NumPy first.

    for program in (kernel.frontend_program, kernel.program):
        result = interpret_program(
            program, {name: value.copy() for name, value in inputs.items()}, **options
        )
        np.testing.assert_allclose(
            result.outputs["out"], expected, rtol=1e-4, atol=1e-4
        )
    return kind, kernel, pipeline, context, inputs, expected, options


@pytest.mark.parametrize("backend", ("triton", "cuda"))
def test_real_default_linalg_passes_have_complete_nonduplicated_origin_records(
    backend, real_linalg_case
):
    kind, kernel, pipeline, context, inputs, _expected, options = _real_pipeline_case(
        backend, real_linalg_case
    )
    originals = {name: value.copy() for name, value in inputs.items()}
    snapshots = []

    def run_pass(pass_, before):
        after = record_pass(before, pass_.run(before, context), pass_.name)
        snapshots.append(after)

        return after

    checks = tuple((pass_.name, partial(run_pass, pass_)) for pass_ in pipeline.passes)
    report = check_passes(kernel.frontend_program, checks, inputs, **options)
    assert report.passed, report
    assert report.checked_passes == tuple(pass_.name for pass_ in pipeline.passes)
    assert len(snapshots) == len(pipeline.passes)
    assert ssa.render(snapshots[-1]) == ssa.render(kernel.program)
    assert (
        tuple(
            entry["name"] for entry in kernel.program.metadata["provenance"]["passes"]
        )
        == report.checked_passes
    )

    for index, snapshot in enumerate(snapshots):
        history = snapshot.metadata["provenance"]["passes"]
        assert (
            tuple(entry["name"] for entry in history)
            == report.checked_passes[: index + 1]
        )
        relations = history[-1]["relations"]
        assert not any(
            entry["relation"] in {"unmapped", "unmapped_removal"} for entry in relations
        )
        locations = operation_locations(snapshot)
        assert all(operation.origins for _location, operation in locations)
        targets = tuple(
            location for entry in relations for location in entry["outputs"]
        )
        assert len(targets) == len(set(targets)) == len(locations)
        assert set(targets) == {location for location, _operation in locations}

    decomposed = next(
        snapshot
        for snapshot in snapshots
        if snapshot.metadata["provenance"]["passes"][-1]["name"]
        == "ssa.decompose_linalg"
    )
    splits = [
        entry
        for entry in decomposed.metadata["provenance"]["passes"][-1]["relations"]
        if entry["relation"] == "split"
    ]
    assert len(splits) == 1
    assert len(splits[0]["outputs"]) > len(splits[0]["inputs"])
    candidates = source_candidates(decomposed)
    assert {source.opcode for source in candidates} == {f"linalg.{kind}", "mem.store"}
    assert len(candidates) == 2
    comparison = compare_programs(
        kernel.frontend_program, decomposed, inputs, **options
    )
    assert comparison.equal
    assert not comparison.traces_aligned
    assert comparison.first_operation is None

    for name, original in originals.items():
        np.testing.assert_array_equal(inputs[name], original)


def _inject_decomposed_fault(program, *, kind, name, wrong, map_values=False):
    """Inject a diagnostic-only bug into an actual default-pass instruction."""
    tracker = ProvenancePass(program, name)
    opcode = "arith.mul" if kind == "dot" else "tensor.extract"
    decomposition = "matmul" if kind == "dot" else "transpose"
    matching = [
        operation
        for _location, operation in operation_locations(program)
        if operation.opcode == opcode
        and operation.attrs.get("decomposition") == decomposition
    ]
    assert len(matching) == 1
    target = matching[0]
    touched = []

    def transform_block(block):
        operations = []

        for operation in block.operations:
            if operation is target:
                if kind == "dot":
                    replacement = replace(
                        operation, opcode="arith.add" if wrong else operation.opcode
                    )
                else:
                    tensor, row, column = operation.operands
                    replacement = replace(
                        operation,
                        operands=(tensor, column, row) if wrong else operation.operands,
                    )

                derived = tracker.derive((replacement,), (operation,))
                operations.extend(derived)

                if map_values:
                    tracker.map_result(operation, derived[0])

                touched.append(operation)
            else:
                regions = tuple(transform_block(region) for region in operation.regions)

                if all(new is old for new, old in zip(regions, operation.regions)):
                    operations.append(operation)
                else:
                    operations.extend(
                        tracker.derive(
                            (replace(operation, regions=regions),), (operation,)
                        )
                    )

        if all(new is old for new, old in zip(operations, block.operations)):
            return block
        return replace(block, operations=tuple(operations))

    blocks = tuple(transform_block(block) for block in program.blocks)
    assert touched == [target]

    return tracker.finish(replace(program, blocks=blocks))


@pytest.mark.parametrize("backend", ("triton", "cuda"))
@pytest.mark.parametrize(
    "wrong", (True, False), ids=("injected-fault", "negative-control")
)
def test_real_decomposition_fault_reports_injected_pass_and_original_source_range(
    backend, wrong, real_linalg_case
):
    kind, kernel, pipeline, context, inputs, expected, options = _real_pipeline_case(
        backend, real_linalg_case
    )
    originals = {name: value.copy() for name, value in inputs.items()}
    injected_name = f"diagnostic.inject_{kind}_fault"
    mutated = []
    visited_after = []

    def inject(program):
        candidate = _inject_decomposed_fault(
            program, kind=kind, name=injected_name, wrong=wrong
        )
        mutated.append(candidate)

        return candidate

    def after_fault(program):
        visited_after.append(True)

        return program

    checks = []

    for pass_ in pipeline.passes:
        checks.append((pass_.name, partial(pass_.run, context=context)))

        if pass_.name == "ssa.decompose_linalg":
            checks.append((injected_name, inject))

    checks.append(("diagnostic.after_fault", after_fault))
    report = check_passes(kernel.frontend_program, tuple(checks), inputs, **options)
    assert len(mutated) == 1
    executed = interpret_program(
        mutated[0], {name: value.copy() for name, value in inputs.items()}, **options
    )

    if wrong:
        assert not np.allclose(executed.outputs["out"], expected, rtol=1e-4, atol=1e-4)
        assert not report.passed
        assert report.first_bad_pass == injected_name
        assert report.checked_passes[-1] == injected_name
        assert report.error is None
        assert report.difference.output_differences == ("out",)
        assert not report.difference.traces_aligned
        assert report.difference.first_operation is None
        assert report.adjacent_difference is not None
        assert not report.adjacent_difference.equal
        assert report.localization is not None
        assert report.localization.reference == "previous"
        assert report.localization.operation.lane is not None
        assert {source.opcode for source in report.source_candidates} == {
            f"linalg.{kind}",
            "mem.store",
        }
        assert len(report.source_candidates) == 2
        assert report.source_candidates == report.difference.source_candidates
        assert visited_after == []
    else:
        np.testing.assert_allclose(
            executed.outputs["out"], expected, rtol=1e-4, atol=1e-4
        )
        assert report.passed
        assert report.first_bad_pass is None
        assert report.source_candidates == ()
        assert visited_after == [True]

    for name, original in originals.items():
        np.testing.assert_array_equal(inputs[name], original)
