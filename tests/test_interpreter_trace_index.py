"""Keep mapping analysis linear in trace traversal without changing diagnosis."""

from dataclasses import replace

import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import _same_snapshot
from ninetoothed.interpreter.localization import compare_mapped_results
from ninetoothed.ir import ssa
from ninetoothed.ir.provenance import ProvenancePass, seed_origins


class CountedTrace(tuple):
    """Count complete trace traversals, avoiding timing assertions in tests."""

    def __iter__(self):
        self.traversals = getattr(self, "traversals", 0) + 1
        yield from super().__iter__()


def mapped_constants(count):
    values = tuple(
        ssa.Value(name=f"%v{index}", type=ssa.Type(kind="scalar", dtype="int32"))
        for index in range(count)
    )
    operations = tuple(
        ssa.Operation(opcode="arith.constant", results=(value,), attrs={"value": index})
        for index, value in enumerate(values)
    )
    source = seed_origins(
        ssa.Program(
            kind="trace_index",
            blocks=(ssa.Block(operations=operations),),
            outputs=values,
        )
    )
    tracker = ProvenancePass(source, "constants")
    targets = []

    for index, operation in enumerate(source.blocks[0].operations):
        (target,) = tracker.derive(
            (replace(operation, attrs={"value": index + (index >= count // 2)}),),
            (operation,),
        )
        targets.append(target)

    for left, right in reversed(tuple(zip(source.blocks[0].operations, targets))):
        tracker.map_result(left, right)

    candidate = tracker.finish(
        replace(source, blocks=(ssa.Block(operations=tuple(targets)),))
    )
    first, second = (
        interpret_program(source, {}, trace=True),
        interpret_program(candidate, {}, trace=True),
    )

    return source, candidate, first, second


@pytest.mark.parametrize("count", (8, 64))
def test_many_mappings_do_not_rescan_every_trace_for_every_producer(count):
    source, candidate, first, second = mapped_constants(count)
    first = replace(first, trace=CountedTrace(first.trace))
    second = replace(second, trace=CountedTrace(second.trace))
    mismatch, issues = compare_mapped_results(
        source, candidate, first, second, _same_snapshot, 0, 0
    )
    assert mismatch.candidate_index == count // 2
    assert mismatch.reference_index == count // 2
    assert issues == ()
    assert first.trace.traversals <= 2
    assert second.trace.traversals <= 2


def test_value_only_trace_does_not_build_a_memory_history(monkeypatch):
    from ninetoothed.interpreter import localization
    from ninetoothed.interpreter.debugger import OperationDifference

    _, program, _, result = mapped_constants(32)
    event = result.trace[-1]
    observation = OperationDifference(
        event.program_id,
        event.location,
        event.opcode,
        "%v31",
        event.iteration,
        event.lane,
    )

    def refuse(trace):
        raise AssertionError("A value-only trace has no memory history to analyze.")

    monkeypatch.setattr(localization, "_memory_edges", refuse)
    dependencies = localization.backward_slice(program, result.trace, observation)
    assert [item.trace_index for item in dependencies.events] == [31]
    assert dependencies.memory_dependencies == ()
    assert dependencies.boundaries == ()
