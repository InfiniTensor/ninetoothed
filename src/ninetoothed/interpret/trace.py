"""Structured execution traces for the CPU reference interpreter.

A trace records, for every executed SSA operation, the program instance, the SSA
location, the operand and result values, and the mask applied to memory accesses.
"""

import numpy as np

from .values import ARRAY, POINTER, SCALAR, SYMBOL, TUPLE, VIEW

#: Values shorter than this are rendered element by element.
_FULL_VALUE_LIMIT = 8


class ValueSnapshot:
    """A recorded view of one operand or result value."""

    __slots__ = ("dtype", "kind", "name", "shape", "summary", "values")

    def __init__(self, name, kind, dtype, shape, summary, values=None):
        self.name = name
        self.kind = kind
        self.dtype = dtype
        self.shape = tuple(shape)
        self.summary = summary
        self.values = values

    def __repr__(self):
        return f"{self.name}={self.summary}"

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": tuple(self.shape),
            "summary": self.summary,
        }


class TraceEvent:
    """One executed SSA operation inside one program instance."""

    __slots__ = (
        "depth",
        "location",
        "mask",
        "opcode",
        "operands",
        "program_id",
        "results",
        "sequence",
    )

    def __init__(
        self,
        *,
        sequence,
        program_id,
        depth,
        opcode,
        location,
        operands=(),
        results=(),
        mask=None,
    ):
        self.sequence = sequence
        self.program_id = program_id
        self.depth = depth
        self.opcode = opcode
        self.location = location
        self.operands = tuple(operands)
        self.results = tuple(results)
        self.mask = mask

    def __repr__(self):
        return (
            f"[p{self.program_id}] {self.location} {self.opcode}"
            f"({', '.join(repr(item) for item in self.operands)})"
            f" -> {', '.join(repr(item) for item in self.results)}"
        )

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "sequence": self.sequence,
            "program_id": self.program_id,
            "depth": self.depth,
            "opcode": self.opcode,
            "location": self.location,
            "operands": [item.to_dict() for item in self.operands],
            "results": [item.to_dict() for item in self.results],
            "mask": self.mask,
        }


class Tracer:
    """Collects :class:`TraceEvent` objects during interpretation.

    :param program_ids: Only record events for these program instances.
    :param opcodes: Only record events for these opcodes.
    :param watch: Value names whose full content is recorded.
    :param limit: Stop recording after this many events.
    :param breakpoints: Mapping of opcode or SSA location to a callback invoked
        before the operation executes.  Raising from the callback stops
        execution.
    :param on_event: Callback invoked for every recorded event (single stepping).
    """

    def __init__(
        self,
        *,
        program_ids=None,
        opcodes=None,
        watch=(),
        limit=None,
        breakpoints=None,
        on_event=None,
        record_operands=True,
    ):
        self.program_ids = None if program_ids is None else frozenset(program_ids)
        self.opcodes = None if opcodes is None else frozenset(opcodes)
        self.watch = frozenset(watch)
        self.limit = limit
        self.breakpoints = dict(breakpoints or {})
        self.on_event = on_event
        self.record_operands = record_operands
        self.events: list[TraceEvent] = []
        self._sequence = 0

    def accepts(self, program_id, opcode):
        """Return whether an event should be recorded."""
        if self.limit is not None and len(self.events) >= self.limit:
            return False

        if self.program_ids is not None and program_id not in self.program_ids:
            return False

        if self.opcodes is not None and opcode not in self.opcodes:
            return False

        return True

    def breakpoint_for(self, opcode, location):
        """Return the breakpoint callback registered for an operation."""
        return self.breakpoints.get(location) or self.breakpoints.get(opcode)

    def record(self, *, program_id, depth, operation, state, mask=None):
        """Record one executed operation.

        :param program_id: The program instance id.
        :param depth: The region nesting depth.
        :param operation: The executed :class:`ninetoothed.ir.ssa.Operation`.
        :param state: The interpreter state, used to snapshot values.
        :param mask: A textual description of the applied access mask.
        :return: The recorded :class:`TraceEvent`.
        """
        operands = ()

        if self.record_operands:
            operands = tuple(
                snapshot(name, state)
                for name in operation.operands
                if name in state.values
            )

        results = tuple(
            snapshot(result.name, state)
            for result in operation.results
            if result.name in state.values
        )
        event = TraceEvent(
            sequence=self._sequence,
            program_id=program_id,
            depth=depth,
            opcode=operation.opcode,
            location=state.location,
            operands=operands,
            results=results,
            mask=mask,
        )
        self._sequence += 1
        self.events.append(event)

        if self.on_event is not None:
            self.on_event(event)

        return event

    def events_for(self, *, program_id=None, opcode=None):
        """Return recorded events matching an optional filter."""
        result = self.events

        if program_id is not None:
            result = [event for event in result if event.program_id == program_id]

        if opcode is not None:
            result = [event for event in result if event.opcode == opcode]

        return tuple(result)

    def values_of(self, name):
        """Return every recorded snapshot of ``name`` in execution order."""
        found = []

        for event in self.events:
            for item in (*event.operands, *event.results):
                if item.name == name:
                    found.append(item)

        return tuple(found)

    def to_dicts(self):
        """Return the whole trace as JSON-friendly dictionaries."""
        return [event.to_dict() for event in self.events]

    def render(self, *, limit=None):
        """Render the trace as readable text."""
        lines = []

        for event in self.events[: limit if limit is not None else len(self.events)]:
            indent = "  " * event.depth
            operands = ", ".join(repr(item) for item in event.operands)
            results = ", ".join(repr(item) for item in event.results)
            mask = f" mask={event.mask}" if event.mask else ""
            lines.append(
                f"[p{event.program_id}]{indent} {event.location} {event.opcode}"
                f"({operands}){mask}" + (f" -> {results}" if results else "")
            )
        return "\n".join(lines)


def snapshot(name, state):
    """Build a :class:`ValueSnapshot` for one runtime value."""
    value = state.values[name]
    full = name in getattr(state.tracer, "watch", ())

    if value.kind == VIEW:
        tensor = value.data.tensor
        data = state.context.read(value.data) if full else None
        summary = f"view{tensor.name}<{'x'.join(str(dim) for dim in value.data.shape)}>"

        return ValueSnapshot(
            name,
            VIEW,
            value.dtype,
            value.data.shape,
            summary if data is None else f"{summary} {_render(data)}",
            data,
        )

    if value.kind == ARRAY:
        return ValueSnapshot(
            name,
            ARRAY,
            str(value.data.dtype),
            value.data.shape,
            _render(value.data)
            if full or value.data.size <= _FULL_VALUE_LIMIT
            else _stats(value.data),
            value.data if full else None,
        )

    if value.kind == SCALAR:
        return ValueSnapshot(
            name, SCALAR, value.dtype, (), repr(np.asarray(value.data).item())
        )

    if value.kind == POINTER:
        return ValueSnapshot(
            name,
            POINTER,
            value.dtype,
            (),
            f"&{value.data.tensor.name}+{value.data.offset}",
        )

    if value.kind == TUPLE:
        return ValueSnapshot(
            name, TUPLE, None, (len(value.data),), f"tuple({len(value.data)})"
        )

    if value.kind == SYMBOL:
        return ValueSnapshot(name, SYMBOL, None, (), str(value.data))

    return ValueSnapshot(name, value.kind, value.dtype, value.shape, repr(value.data))


def _render(data):
    data = np.asarray(data)

    if data.ndim == 0:
        return repr(data.item())

    if data.size <= _FULL_VALUE_LIMIT:
        return np.array2string(data, separator=", ")

    return _stats(data)


def _stats(data):
    data = np.asarray(data)

    if data.size == 0:
        return f"{data.dtype}[0] <empty>"

    if data.dtype.kind in "biu":
        return (
            f"{data.dtype}[{data.size}] min={data.min()} max={data.max()} "
            f"sum={data.sum()}"
        )

    with np.errstate(all="ignore"):
        finite = data[np.isfinite(data)]

    if finite.size == 0:
        return f"{data.dtype}[{data.size}] <non-finite>"

    return (
        f"{data.dtype}[{data.size}] min={finite.min():.6g} "
        f"max={finite.max():.6g} mean={finite.mean():.6g}"
    )


__all__ = ["TraceEvent", "Tracer", "ValueSnapshot", "snapshot"]
