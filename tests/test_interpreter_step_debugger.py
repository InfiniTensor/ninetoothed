"""Deterministic interactive-debugger checks using injected command streams."""

import numpy as np
import pytest

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import DebuggerQuit, StepDebugger
from ninetoothed.ir import ssa


def _program():
    tensor = ssa.Type(kind="tensor", shape=("2",), dtype="float32")
    x, out, plus, square = (
        ssa.Value(name=name, type=tensor) for name in ("x", "out", "%plus", "%square")
    )
    one = ssa.Value(name="%one", type=ssa.Type(kind="scalar", dtype="float32"))

    return ssa.Program(
        kind="debug_steps",
        inputs=(x, out),
        outputs=(out,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(one,), attrs={"value": 1}
                    ),
                    ssa.Operation(
                        opcode="arith.add", operands=("x", "%one"), results=(plus,)
                    ),
                    ssa.Operation(
                        opcode="arith.mul",
                        operands=("%plus", "%plus"),
                        results=(square,),
                    ),
                    ssa.Operation(opcode="mem.store", operands=("%square", "out")),
                )
            ),
        ),
    )


def _inputs():
    return {
        "x": np.array([2, 3], dtype=np.float32),
        "out": np.full(2, -731, dtype=np.float32),
    }


def test_step_continue_dynamic_watch_and_store_pause_timing():
    inputs = _inputs()
    at_command = []
    output = []

    def commands():
        at_command.append(inputs["out"].copy())
        yield "p %one"
        yield "w %plus"
        yield "s"
        at_command.append(inputs["out"].copy())
        yield "p %plus"
        yield "b mem.store"
        yield "c"
        at_command.append(inputs["out"].copy())
        yield "p out"
        yield "c"

    debugger = StepDebugger(commands=commands(), output=output.append)
    result = interpret_program(_program(), inputs, callback=debugger)
    assert [event.opcode for event in debugger.pauses] == [
        "arith.constant",
        "arith.add",
        "mem.store",
    ]
    assert debugger.events_seen == 4
    np.testing.assert_array_equal(at_command[0], [-731, -731])
    np.testing.assert_array_equal(at_command[1], [-731, -731])
    np.testing.assert_array_equal(at_command[2], [9, 16])
    np.testing.assert_array_equal(result.outputs["out"], [9, 16])
    assert debugger.inspect("%plus")["value"] == [3.0, 4.0]
    assert debugger.pauses[1].watched["%plus"]["value"] == [3.0, 4.0]
    assert any("watch added: %plus" in line for line in output)
    assert any(line.startswith("%plus = ") for line in output)


@pytest.mark.parametrize("breakpoint", ("arith.add", "entry:1", "entry:1:arith.add"))
def test_breakpoint_matches_opcode_exact_location_or_location_prefix(breakpoint):
    debugger = StepDebugger(
        commands=("c",),
        output=lambda _line: None,
        breakpoints=(breakpoint,),
        stop_on_entry=False,
    )
    interpret_program(_program(), _inputs(), callback=debugger)
    assert [event.location for event in debugger.pauses] == ["entry:1:arith.add"]


def test_deleting_a_breakpoint_restores_continue_behavior():
    debugger = StepDebugger(
        commands=("b arith.mul", "d arith.mul", "c"), output=lambda _line: None
    )
    interpret_program(_program(), _inputs(), callback=debugger)
    assert len(debugger.pauses) == 1
    assert "arith.mul" not in debugger.breakpoints


@pytest.mark.parametrize("location,written", (("entry:1", False), ("mem.store", True)))
def test_quit_stops_future_operations_and_does_not_rollback_completed_stores(
    location, written
):
    inputs = _inputs()
    debugger = StepDebugger(
        commands=("q",),
        output=lambda _line: None,
        breakpoints=(location,),
        stop_on_entry=False,
    )

    with pytest.raises(DebuggerQuit, match="Debugger stopped"):
        interpret_program(_program(), inputs, callback=debugger)

    np.testing.assert_array_equal(inputs["out"], [9, 16] if written else [-731, -731])


def test_program_and_opcode_filters_bound_debugger_events():
    debugger = StepDebugger(
        commands=("c",), output=lambda _line: None, watch=("%plus",)
    )
    inputs = _inputs()
    result = interpret_program(
        _program(),
        inputs,
        grid=(3,),
        program_ids=((1, 0, 0),),
        opcodes=("arith.mul",),
        callback=debugger,
    )
    assert debugger.events_seen == 1
    assert debugger.pauses[0].program_id == (1, 0, 0)
    assert debugger.pauses[0].opcode == "arith.mul"
    assert debugger.inspect("%plus")["value"] == [3.0, 4.0]
    np.testing.assert_array_equal(result.outputs["out"], [9, 16])


def _arrangement(x, out):
    return x.tile((4,)), out.tile((4,))


def _application(x, out):
    out = x + 1  # noqa: F841


def test_debugger_operates_on_real_tiled_frontend_application():
    debugger = StepDebugger(
        commands=("c", "c"),
        output=lambda _line: None,
        breakpoints=("mem.store",),
        stop_on_entry=False,
    )
    kernel = interpret(
        _arrangement,
        _application,
        (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")),
        callback=debugger,
    )
    x = np.arange(5, dtype=np.float32)
    out = np.full_like(x, -731)
    kernel(x, out)
    np.testing.assert_array_equal(out, x + 1)
    assert [event.program_id for event in debugger.pauses] == [(0, 0, 0), (1, 0, 0)]
    assert debugger.pauses[-1].mask["value"] == [True, False, False, False]


def test_exhausted_script_continues_without_requesting_user_input(monkeypatch):
    def interactive_input_is_forbidden(_prompt):
        raise AssertionError("An exhausted scripted debugger must not call input().")

    monkeypatch.setattr("builtins.input", interactive_input_is_forbidden)
    debugger = StepDebugger(commands=(), output=lambda _line: None)
    result = interpret_program(_program(), _inputs(), callback=debugger)
    assert len(debugger.pauses) == 1
    np.testing.assert_array_equal(result.outputs["out"], [9, 16])
