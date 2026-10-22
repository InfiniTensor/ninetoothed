"""Interactive debugging for the CPU reference interpreter.

The interpreter walks the SSA program one operation at a time, which makes an
operation the natural unit of interactive debugging. A ``DebugSession`` pauses
before an operation, either because the caller asked for the next one or because
a breakpoint matched, and hands back what is known at that point: the program
instance, the SSA location, the operands with their current values, the active
mask, and the values on the watch list.

The interpreter is vectorized and runs a whole program instance in one call, so a
session replays the run for every stop. The tensor arguments are restored first,
which keeps a kernel that accumulates into its output consistent with a plain
run; the cost is re-running the operations that precede the stop. Nothing here is
on the path of an ordinary run: a kernel only pays for a session when it is given
one.

.. code-block:: python

    from ninetoothed.interpreter import Breakpoint

    session = kernel.debug(
        lhs,
        rhs,
        output,
        breakpoints=(Breakpoint(program_ids=(3,), opcodes=("mem.",)),),
        watch=("output",),
    )

    while True:
        stop = session.resume()

        if stop is None:
            break

        print(stop.format())
"""

from dataclasses import dataclass, field

import numpy as np

from ninetoothed.interpreter.executor import ExecutionStopped
from ninetoothed.interpreter.trace import (
    DEFAULT_MAX_ELEMENTS,
    matches_pattern,
    summarize,
)

_UNBOUND = "<unbound>"


@dataclass(frozen=True, kw_only=True)
class Breakpoint:
    """A condition that stops a session before an operation runs.

    :param program_ids: The program instances to stop in, or ``None`` for all.
    :param opcodes: The opcodes to stop at, matched exactly or by a prefix that
        ends with a dot, or ``None`` for all.
    :param locations: The SSA locations to stop at, matched the same way, or
        ``None`` for all.
    """

    program_ids: tuple[int, ...] | None = None
    opcodes: tuple[str, ...] | None = None
    locations: tuple[str, ...] | None = None

    def matches(self, *, program_id: int, opcode: str, location: str) -> bool:
        """Return whether one operation is a stop for this breakpoint.

        :param program_id: The program instance of the operation.
        :param opcode: The SSA opcode of the operation.
        :param location: The SSA location of the operation.
        :return: Whether the session should stop before the operation.
        """
        if self.program_ids is not None and program_id not in self.program_ids:
            return False

        return matches_pattern(opcode, self.opcodes) and matches_pattern(
            location, self.locations
        )


@dataclass(frozen=True, kw_only=True)
class Stop:
    """What the interpreter was about to do at one stop.

    :param index: The ordinal of the operation that did not run yet.
    :param program_id: The program instance that was about to run it.
    :param location: The SSA location of the operation.
    :param opcode: The SSA opcode of the operation.
    :param operands: The operand names of the operation.
    :param inputs: A summary of every operand value.
    :param mask: A summary of the mask that was active, if any.
    :param watched: A summary of every watched value.
    """

    index: int
    program_id: int
    location: str
    opcode: str
    operands: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    mask: str | None = None
    watched: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        """Render the stop as a readable multi-line string."""
        line = f"[program {self.program_id}] stop {self.index}: {self.location}"

        if self.operands:
            line = f"{line} ({', '.join(self.operands)})"

        if self.inputs:
            rendered = "; ".join(f"{name}={text}" for name, text in self.inputs.items())
            line = f"{line}\n    in   : {rendered}"

        if self.mask is not None:
            line = f"{line}\n    mask : {self.mask}"

        if self.watched:
            rendered = "; ".join(
                f"{name}={text}" for name, text in self.watched.items()
            )
            line = f"{line}\n    watch: {rendered}"

        return line


class DebugSession:
    """An interactive session over one interpreter kernel.

    :param kernel: The interpreter kernel to run.
    :param arguments: The positional arguments of the kernel.
    :param breakpoints: The breakpoints the session stops at.
    :param watch: The SSA names that are reported at every stop.
    :param max_elements: The maximum number of elements shown per value.
    :param options: The keyword arguments of the kernel, such as the meta symbols.
    """

    def __init__(
        self,
        kernel,
        *arguments,
        breakpoints=(),
        watch=(),
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        **options,
    ):
        self.kernel = kernel
        self.arguments = tuple(arguments)
        self.options = dict(options)
        self.breakpoints = list(breakpoints)
        self.watch_names = tuple(dict.fromkeys(watch))
        self.max_elements = max_elements
        self.index = -1
        self.minimum = 0
        self.step_target = None
        self.stop: Stop | None = None
        self.history: list[Stop] = []
        self.finished = False
        self.output = None
        self.environment: dict = {}
        self._mutable = _mutable_arguments(self.arguments, self.options)
        self._saved = [_snapshot(argument) for argument in self._mutable]

    def __len__(self) -> int:
        """Return the number of stops the session has reported."""
        return len(self.history)

    def watch(self, *names: str) -> None:
        """Add SSA names to the watch list.

        :param names: The SSA names to report at every stop.
        """
        self.watch_names = tuple(dict.fromkeys(self.watch_names + tuple(names)))

    def value(self, name: str):
        """Return the value of one SSA name as it was at the last stop.

        The returned array is a copy, so later replays cannot change it.

        :param name: The SSA name to look up.
        :return: The value, or ``None`` when the name was not bound at the stop.
        """
        return self.environment.get(name)

    def step(self) -> Stop | None:
        """Run to the next operation and report the stop.

        :return: The stop, or ``None`` when the run has ended.
        """
        if self.finished:
            return None

        self.step_target = self.index + 1
        self.minimum = self.step_target

        return self.stop if self._drive() else None

    def resume(self) -> Stop | None:
        """Run to the next breakpoint and report the stop.

        :return: The stop, or ``None`` when the run has ended.
        """
        if self.finished:
            return None

        self.step_target = None
        self.minimum = self.index + 1

        return self.stop if self._drive() else None

    def finish(self):
        """Run to the end without stopping.

        :return: The primary output of the kernel, as a plain run returns it.
        """
        if not self.finished:
            self.step_target = None
            self.minimum = None

            self._drive()

        return self.output

    def reset(self) -> None:
        """Forget every stop so the next one starts from the first operation."""
        self.index = -1
        self.minimum = 0
        self.step_target = None
        self.stop = None
        self.history = []
        self.finished = False
        self.output = None
        self.environment = {}

    def format_history(self) -> str:
        """Render every stop of the session, one operation per line."""
        return "\n".join(stop.format() for stop in self.history)

    def _drive(self) -> bool:
        """Replay the kernel from the restored arguments.

        :return: Whether the run stopped instead of reaching the end.
        """
        for argument, saved in zip(self._mutable, self._saved):
            argument[...] = saved

        try:
            self.output = self.kernel.run(
                *self.arguments, inspector=self._inspect, **self.options
            )
        except ExecutionStopped:
            return True

        self.finished = True

        return False

    def _inspect(self, *, operation, location: str, state, index: int) -> bool:
        """Decide whether the interpreter stops before one operation."""
        if self.minimum is None or index < self.minimum:
            return False

        if self.step_target is not None:
            if index < self.step_target:
                return False
        elif not any(
            breakpoint.matches(
                program_id=state.program_id, opcode=operation.opcode, location=location
            )
            for breakpoint in self.breakpoints
        ):
            return False

        self.index = index
        self.environment = {
            name: _copy(value) for name, value in state.environment.items()
        }
        self.stop = Stop(
            index=index,
            program_id=state.program_id,
            location=location,
            opcode=operation.opcode,
            operands=tuple(operation.operands),
            inputs={
                name: self._summarize(state.environment.get(name))
                for name in operation.operands
            },
            mask=(
                self._summarize(state.mask)
                if operation.opcode.startswith("mem.") and state.mask is not None
                else None
            ),
            watched={
                name: (
                    _UNBOUND
                    if name not in self.environment
                    else self._summarize(self.environment[name])
                )
                for name in self.watch_names
            },
        )
        self.history.append(self.stop)

        return True

    def _summarize(self, value) -> str:
        """Return a compact description of one value."""
        return summarize(value, max_elements=self.max_elements)


def _mutable_arguments(arguments, options) -> list:
    """Return the arguments the interpreter writes into."""
    found = []

    for container in (arguments, options.values()):
        for value in container:
            shape = getattr(value, "shape", None)

            if shape is not None and len(tuple(shape)) > 0 and hasattr(value, "dtype"):
                found.append(value)

    return found


def _snapshot(value):
    """Return a copy of one mutable argument that assignment restores.

    The snapshot keeps the container of the argument, so a kernel that is handed
    either a NumPy array or a PyTorch CPU tensor replays from the same state.

    :param value: The argument to copy.
    :return: A copy that ``argument[...] = snapshot`` restores.
    """
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)

    return value.detach().clone()


def _copy(value):
    """Return a snapshot of one value that later writes cannot change."""
    if isinstance(value, np.ndarray):
        return value.copy()

    return value


__all__ = ["Breakpoint", "DebugSession", "Stop"]
