"""After installation, run python docs/cpu_interpreter_demo.py [--export PATH]."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ninetoothed import Tensor, interpret
from ninetoothed.interpreter.debugger import (
    StepDebugger,
    check_passes,
)


def arrangement(x, out):
    return x.tile((4,)), out.tile((4,))


def application(x, out):
    out = x * 2 + 1  # noqa: F841


def deliberately_bad_pass(program):
    """Inject one visible error to demonstrate automatic localization."""
    block = program.blocks[0]
    operations = list(block.operations)

    for index, operation in enumerate(operations):
        if operation.opcode == "arith.constant" and operation.attrs.get("value") == 2:
            operations[index] = replace(operation, attrs={"value": 3})
            break
    return replace(program, blocks=(replace(block, operations=tuple(operations)),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        help="Save reference/candidate SSA and a differential replay in a new directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Demonstrate scripted stepping and breakpoints.",
    )
    parser.add_argument(
        "--interactive-debug",
        action="store_true",
        help="Read debugger commands from the terminal.",
    )
    arguments = parser.parse_args()
    seed = 2026
    x = np.random.default_rng(seed).normal(size=11).astype(np.float32)
    out = np.zeros_like(x)
    debugger = None

    if arguments.debug or arguments.interactive_debug:
        commands = (
            None
            if arguments.interactive_debug
            else ("watch %0", "print %0", "step", "break mem.store", "continue")
        )
        debugger = StepDebugger(commands=commands)

    kernel = interpret(
        arrangement,
        application,
        (Tensor(1, name="x", dtype="float32"), Tensor(1, name="out", dtype="float32")),
        trace=True,
        backend="triton",
        callback=debugger,
    )
    result = kernel(x, out)
    np.testing.assert_allclose(out, x * 2 + 1, rtol=1e-3, atol=1e-3)
    print(
        f"Correct CPU output: {out.shape}, {out.dtype}; {len(result.trace)} trace events"
    )
    inputs = {"x": x, "out": np.zeros_like(out)}
    report = check_passes(
        kernel.frontend_program,
        (
            ("unchanged", lambda program: program),
            ("injected_bad_constant", deliberately_bad_pass),
        ),
        inputs,
        tensors=kernel.tensors,
        symbols=kernel.meta,
        failure_dir=arguments.export,
        seed=seed,
    )
    assert report.first_bad_pass == "injected_bad_constant"
    print(f"First bad pass: {report.first_bad_pass}")
    print(f"First different operation: {report.difference.first_operation}")
    print("Fault type: deliberately injected constant 2 -> 3; not a historical bug")

    if arguments.export:
        if report.export_error is not None:
            raise RuntimeError(
                f"Failure capture did not complete: {report.export_error}."
            )

        print(f"Differential replay: {report.reproducer / 'replay.py'}")


if __name__ == "__main__":
    main()
