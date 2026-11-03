"""Check observed storage dependencies against independently known writes."""

from dataclasses import replace

import numpy as np
import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import OperationDifference
from ninetoothed.interpreter.localization import backward_slice
from ninetoothed.ir import ssa


def _value(name, *, kind="scalar", shape=()):
    return ssa.Value(name=name, type=ssa.Type(kind=kind, dtype="int32", shape=shape))


def _pointer_program(*, overwrite=False, masked=False, alias=False):
    pointer = _value("buffer", kind="pointer")
    other = _value("other", kind="pointer")
    zero, one, seven, nine, thirteen, second, result = (
        _value(name, kind="pointer" if name == "%second" else "scalar")
        for name in (
            "%zero",
            "%one",
            "%seven",
            "%nine",
            "%thirteen",
            "%second",
            "%result",
        )
    )
    operations = [
        ssa.Operation(opcode="arith.constant", results=(v,), attrs={"value": n})
        for v, n in ((zero, 0), (one, 1), (seven, 7), (nine, 9), (thirteen, 13))
    ]
    operations += [
        ssa.Operation(
            opcode="arith.add", operands=(pointer.name, one.name), results=(second,)
        ),
        ssa.Operation(opcode="mem.store", operands=(seven.name, pointer.name)),
        ssa.Operation(opcode="mem.store", operands=(thirteen.name, second.name)),
    ]

    if overwrite:
        operations.append(
            ssa.Operation(
                opcode="mem.store",
                operands=(nine.name, pointer.name),
                attrs={"mask": not masked},
            )
        )

    operations.append(
        ssa.Operation(
            opcode="mem.load",
            operands=(other.name if alias else pointer.name,),
            results=(result,),
        )
    )

    return ssa.Program(
        kind="memory_history",
        inputs=(pointer, other) if alias else (pointer,),
        outputs=(result,),
        blocks=(ssa.Block(operations=tuple(operations)),),
    )


def _slice(program, inputs, **options):
    result = interpret_program(program, inputs, trace=True, **options)
    last = result.trace[-1]
    observation = OperationDifference(
        last.program_id,
        last.location,
        last.opcode,
        "%result",
        last.iteration,
        last.lane,
    )

    return result, backward_slice(program, result.trace, observation)


@pytest.mark.parametrize(
    "overwrite,masked", ((False, False), (True, False), (True, True))
)
@pytest.mark.parametrize("alias", (False, True))
def test_read_depends_on_the_latest_active_write_to_the_same_bytes(
    overwrite, masked, alias
):
    program = _pointer_program(overwrite=overwrite, masked=masked, alias=alias)
    buffer = np.zeros(2, dtype=np.int32)
    inputs = {"buffer": buffer}

    if alias:
        inputs["other"] = buffer

    result, dependencies = _slice(program, inputs)
    assert result.outputs["%result"] == (9 if overwrite and not masked else 7)
    writes = [event for event in dependencies.events if event.opcode == "mem.store"]
    assert [event.location for event in writes] == [
        "entry:8:mem.store" if overwrite and not masked else "entry:6:mem.store"
    ]
    assert dependencies.boundaries == ()
    assert len(dependencies.memory_dependencies) == 1
    edge = dependencies.memory_dependencies[0]
    assert edge.writer_index == writes[0].trace_index
    assert edge.reader_index == len(result.trace) - 1
    assert edge.byte_ranges == ((0, 4),)


def test_equal_values_in_disjoint_allocations_do_not_create_a_dependency():
    program = _pointer_program(alias=True)
    result, dependencies = _slice(
        program,
        {"buffer": np.zeros(2, dtype=np.int32), "other": np.full(2, 7, dtype=np.int32)},
    )
    assert result.outputs["%result"] == 7
    assert not any(event.opcode == "mem.store" for event in dependencies.events)
    assert dependencies.memory_dependencies == ()
    assert dependencies.boundaries == ()


def test_capture_is_deterministic_and_does_not_serialize_process_addresses():
    program = _pointer_program()
    first, _ = _slice(program, {"buffer": np.zeros(2, dtype=np.int32)})
    second, _ = _slice(program, {"buffer": np.zeros(2, dtype=np.int32)})
    assert first.trace[-1].memory == second.trace[-1].memory
    (access,) = first.trace[-1].memory
    assert access.kind == "read"
    assert access.storage == "storage:buffer"
    assert access.byte_ranges == ((0, 4),)


def test_filtered_trace_marks_a_missing_writer_instead_of_guessing():
    program = _pointer_program()
    result, dependencies = _slice(
        program, {"buffer": np.zeros(2, dtype=np.int32)}, opcodes=("mem.load",)
    )
    assert result.outputs["%result"] == 7
    assert dependencies.memory_dependencies == ()
    assert any("filtered" in boundary for boundary in dependencies.boundaries)


def test_trace_capture_does_not_turn_masked_out_addresses_into_reads():
    program = _pointer_program(overwrite=True, masked=True)
    operations = list(program.blocks[0].operations)
    bad = _value("%bad", kind="pointer")
    operations[-2:-2] = [
        ssa.Operation(
            opcode="arith.add", operands=("buffer", "%thirteen"), results=(bad,)
        )
    ]
    operations[-2] = replace(operations[-2], operands=("%nine", bad.name))
    program = replace(program, blocks=(ssa.Block(operations=tuple(operations)),))
    result, dependencies = _slice(program, {"buffer": np.zeros(2, dtype=np.int32)})
    assert result.outputs["%result"] == 7
    assert [
        event.location for event in dependencies.events if event.opcode == "mem.store"
    ] == ["entry:6:mem.store"]
    assert dependencies.boundaries == ()


@pytest.mark.parametrize("seed", range(5))
def test_interval_history_matches_an_independent_byte_oracle(seed):
    from ninetoothed.interpreter.access import WrittenIntervals

    random = np.random.default_rng(seed)
    history = WrittenIntervals()
    oracle = [-1] * 96

    for writer in range(120):
        start = int(random.integers(0, 96))
        end = int(random.integers(start + 1, 97))
        history.write(start, end, writer)
        oracle[start:end] = [writer] * (end - start)
        observed = [-1] * 96

        for lower, upper, index in history.readers(0, 96):
            observed[lower:upper] = [index] * (upper - lower)

        assert observed == oracle

        for left, right in zip(history.intervals, history.intervals[1:]):
            assert left[1] <= right[0]


@pytest.mark.parametrize("view", ("same", "reverse", "partial", "disjoint"))
@pytest.mark.parametrize("implicit", (False, True))
def test_tensor_views_use_actual_strides_and_only_real_reads(view, implicit):
    allocation = np.zeros(8, dtype=np.int32)
    target = allocation[::2]
    reader = {
        "same": target,
        "reverse": target[::-1],
        "partial": target[1:],
        "disjoint": allocation[1::2],
    }[view]
    x, alias = (
        _value("x", kind="tensor", shape=("4",)),
        _value("alias", kind="tensor", shape=(str(reader.size),)),
    )
    seven, result = (
        _value("%seven"),
        _value("%result", kind="tensor", shape=(str(reader.size),)),
    )
    program = ssa.Program(
        kind="tensor_memory",
        inputs=(x, alias),
        outputs=(result,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(seven,), attrs={"value": 7}
                    ),
                    ssa.Operation(opcode="mem.store", operands=(seven.name, x.name)),
                    ssa.Operation(
                        opcode="arith.add" if implicit else "mem.load",
                        operands=(alias.name, seven.name)
                        if implicit
                        else (alias.name,),
                        results=(result,),
                    ),
                )
            ),
        ),
    )
    actual, dependencies = _slice(
        program, {"x": target, "alias": reader}, watch=("x", "alias")
    )
    np.testing.assert_array_equal(
        actual.outputs[result.name],
        (0 if view == "disjoint" else 7) + (7 if implicit else 0),
    )
    assert [
        event.location for event in dependencies.events if event.opcode == "mem.store"
    ] == ([] if view == "disjoint" else ["entry:1:mem.store"])
    assert dependencies.boundaries == ()
    assert len(actual.trace[1].memory) == 1
    assert actual.trace[1].memory[0].kind == "write"
    assert actual.trace[1].memory[0].byte_ranges == (
        (0, 4),
        (8, 12),
        (16, 20),
        (24, 28),
    )


def test_manual_trace_truncation_cannot_invent_initial_memory():
    program = _pointer_program()
    result, _ = _slice(program, {"buffer": np.zeros(2, dtype=np.int32)})
    trace = result.trace[-1:]
    event = trace[0]
    observation = OperationDifference(
        event.program_id,
        event.location,
        event.opcode,
        "%result",
        event.iteration,
        event.lane,
    )
    dependencies = backward_slice(program, trace, observation)
    assert dependencies.memory_dependencies == ()
    assert any("incomplete trace" in boundary for boundary in dependencies.boundaries)


def test_cross_program_writes_are_not_a_gpu_happens_before_proof():
    program = _pointer_program()
    operations = list(program.blocks[0].operations)
    load = operations.pop()
    operations.insert(6, load)
    program = replace(program, blocks=(ssa.Block(operations=tuple(operations)),))
    result = interpret_program(
        program, {"buffer": np.zeros(2, dtype=np.int32)}, trace=True, grid=(2,)
    )
    event = next(
        event
        for event in result.trace
        if event.opcode == "mem.load" and event.program_id == (1, 0, 0)
    )
    observation = OperationDifference(
        event.program_id,
        event.location,
        event.opcode,
        "%result",
        event.iteration,
        event.lane,
    )
    dependencies = backward_slice(program, result.trace, observation)
    assert result.outputs["%result"] == 7
    assert dependencies.memory_dependencies == ()
    assert any("cross-program" in boundary for boundary in dependencies.boundaries)


def test_overlapping_pointer_write_lanes_remain_an_explicit_boundary():
    pointer, offsets = (
        _value("buffer", kind="pointer"),
        _value("offsets", kind="tensor", shape=("2",)),
    )
    shifted, values, result = (
        _value("%shifted", kind="pointer"),
        _value("values", kind="tensor", shape=("2",)),
        _value("%result"),
    )
    program = ssa.Program(
        kind="overlapping_writes",
        inputs=(pointer, offsets, values),
        outputs=(result,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.add",
                        operands=(pointer.name, offsets.name),
                        results=(shifted,),
                    ),
                    ssa.Operation(
                        opcode="mem.store", operands=(values.name, shifted.name)
                    ),
                    ssa.Operation(
                        opcode="mem.load", operands=(pointer.name,), results=(result,)
                    ),
                )
            ),
        ),
    )
    actual, dependencies = _slice(
        program,
        {
            "buffer": np.zeros(1, dtype=np.int32),
            "offsets": np.zeros(2, dtype=np.int32),
            "values": np.array([7, 9], dtype=np.int32),
        },
    )
    assert actual.trace[1].memory[-1].overlapping_lanes
    assert any(
        "overlapping write lanes" in boundary for boundary in dependencies.boundaries
    )


def test_untraced_execution_does_not_allocate_memory_diagnostics(monkeypatch):
    import ninetoothed.interpreter.runtime as runtime

    def refuse(*args, **kwargs):
        raise AssertionError("Memory recording must be absent when tracing is off.")

    monkeypatch.setattr(runtime, "MemoryRecorder", refuse)
    result = interpret_program(
        _pointer_program(), {"buffer": np.zeros(2, dtype=np.int32)}
    )
    assert result.outputs["%result"] == 7
    assert result.trace == ()


def test_uninstrumented_handler_effects_do_not_claim_complete_memory_history():
    program = _pointer_program()
    operations = program.blocks[0].operations
    program = replace(
        program,
        blocks=(
            ssa.Block(
                operations=(
                    *operations[:-1],
                    ssa.Operation(opcode="example.effect"),
                    operations[-1],
                )
            ),
        ),
    )
    result, dependencies = _slice(
        program,
        {"buffer": np.zeros(2, dtype=np.int32)},
        handlers={"example.effect": lambda op, args: None},
    )
    assert result.outputs["%result"] == 7
    assert dependencies.memory_dependencies == ()
    assert any(
        "uninstrumented handler" in boundary for boundary in dependencies.boundaries
    )


@pytest.mark.parametrize("source_index", (False, True))
def test_scalar_extract_records_only_the_selected_bytes(source_index):
    x = _value("x", kind="tensor", shape=("3",))
    seven, one, result = _value("%seven"), _value("%one"), _value("%result")
    program = ssa.Program(
        kind="scalar_access",
        inputs=(x,),
        outputs=(result,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(seven,), attrs={"value": 7}
                    ),
                    ssa.Operation(
                        opcode="arith.constant", results=(one,), attrs={"value": 1}
                    ),
                    ssa.Operation(opcode="mem.store", operands=(seven.name, x.name)),
                    ssa.Operation(
                        opcode="tensor.extract",
                        operands=(x.name, one.name),
                        results=(result,),
                        attrs={"source": source_index},
                    ),
                )
            ),
        ),
    )
    actual, dependencies = _slice(program, {"x": np.zeros(3, dtype=np.int32)})
    assert actual.outputs[result.name] == 7
    (edge,) = dependencies.memory_dependencies
    assert edge.byte_ranges == ((4, 8),)


def test_memory_dependency_bundle_replays_and_detects_a_changed_writer(tmp_path):
    import json

    from ninetoothed.interpreter.debugger import compare_programs
    from ninetoothed.interpreter.failure import replay_failure
    from ninetoothed.ir.provenance import ProvenancePass, seed_origins

    original = seed_origins(_pointer_program())
    tracker = ProvenancePass(original, "wrong_store")
    operations = list(original.blocks[0].operations)
    old = operations[6]
    (operations[6],) = tracker.derive(
        (replace(old, operands=("%nine", "buffer")),), (old,)
    )
    candidate = tracker.finish(
        replace(original, blocks=(ssa.Block(operations=tuple(operations)),))
    )
    report = compare_programs(
        original,
        candidate,
        {"buffer": np.zeros(2, dtype=np.int32)},
        failure_dir=tmp_path / "failure",
    )
    assert not report.equal
    assert report.localization.operation.opcode == "mem.load"
    assert report.dependency_slice.memory_dependencies[0].writer_index == 6
    replay_failure(report.reproducer)
    path = report.reproducer / "failure.json"
    data = json.loads(path.read_text())
    assert data["diagnostics_version"] == 2
    data["report"]["dependency_slice"]["memory_dependencies"][0]["writer_index"] = 7
    path.write_text(json.dumps(data))

    with pytest.raises(RuntimeError, match="operation difference"):
        replay_failure(report.reproducer)


def test_unsigned_pointer_coordinates_ignore_inactive_huge_offsets():
    from ninetoothed.interpreter.access import MemoryRecorder
    from ninetoothed.interpreter.memory import Pointer

    data = np.array([3, 5], dtype=np.int32)
    recorder = MemoryRecorder({"data": data})
    pointer = Pointer(
        data, np.array([1, np.iinfo(np.uint64).max], dtype=np.uint64), observer=recorder
    )

    with recorder.capture() as events:
        result = pointer.read(mask=np.array([True, False]), other=-1)

    np.testing.assert_array_equal(result, [5, -1])
    assert events[0].byte_ranges == ((4, 8),)


@pytest.mark.parametrize(
    "removed", ("memory_dependencies", "projection", "diagnostics_version")
)
def test_new_failure_bundles_require_their_diagnostic_fields(tmp_path, removed):
    import json

    from ninetoothed.interpreter.debugger import compare_programs
    from ninetoothed.interpreter.failure import replay_failure

    reference = _pointer_program()
    operations = list(reference.blocks[0].operations)
    operations[2] = replace(operations[2], attrs={"value": 8})
    candidate = replace(reference, blocks=(ssa.Block(operations=tuple(operations)),))
    report = compare_programs(
        reference,
        candidate,
        {"buffer": np.zeros(2, dtype=np.int32)},
        failure_dir=tmp_path / "failure",
    )
    metadata_path = report.reproducer / "failure.json"
    metadata = json.loads(metadata_path.read_text())

    if removed == "memory_dependencies":
        del metadata["report"]["dependency_slice"][removed]
    elif removed == "projection":
        del metadata["report"]["localization"][removed]
    else:
        del metadata[removed]

    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises((ValueError, RuntimeError)):
        replay_failure(report.reproducer)
