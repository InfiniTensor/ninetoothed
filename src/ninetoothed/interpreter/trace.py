"""Optional execution traces for the CPU reference interpreter."""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_ELEMENTS = 8


def matches_pattern(text: str, patterns) -> bool:
    """Return whether an opcode or SSA location matches a pattern set.

    A pattern either equals the text or, when it ends with a dot, is a prefix of
    it, so ``"mem."`` matches every memory operation.

    :param text: The opcode or SSA location to match.
    :param patterns: The patterns to match against, or ``None`` for everything.
    :return: Whether the text matches.
    """
    if patterns is None:
        return True

    return any(
        text == pattern or (pattern.endswith(".") and text.startswith(pattern))
        for pattern in patterns
    )


def summarize(value, *, max_elements: int = DEFAULT_MAX_ELEMENTS) -> str:
    """Render a compact single-line description of an interpreter value.

    :param value: The interpreter value, such as a scalar, a tuple, or an array.
    :param max_elements: The maximum number of elements shown per container.
    :return: A compact textual summary.
    """
    if isinstance(value, np.ndarray):
        return _summarize_array(value, max_elements=max_elements)

    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return repr(value.item() if isinstance(value, np.generic) else value)

    if isinstance(value, tuple):
        items = ", ".join(
            summarize(item, max_elements=max_elements) for item in value[:max_elements]
        )

        if len(value) > max_elements:
            items = f"{items}, ..."

        return f"({items})"

    return repr(value)


def _summarize_array(value, *, max_elements: int) -> str:
    flat = value.reshape(-1)
    head = ", ".join(repr(item) for item in flat[:max_elements].tolist())

    if flat.size > max_elements:
        head = f"{head}, ..."

    return f"array(shape={tuple(value.shape)}, dtype={value.dtype}, [{head}])"


@dataclass(frozen=True, kw_only=True)
class TraceOptions:
    """Filtering and detail settings for one interpreter run.

    :param program_ids: The program instances to trace, or ``None`` for all.
    :param opcodes: The SSA opcodes to trace, matched exactly or by a prefix
        ending with a dot, or ``None`` for all.
    :param max_elements: The maximum number of elements shown per traced value.
    :param keep_values: Whether traced values are copied into the event.
    """

    program_ids: tuple[int, ...] | None = None
    opcodes: tuple[str, ...] | None = None
    max_elements: int = DEFAULT_MAX_ELEMENTS
    keep_values: bool = False

    def matches(self, *, program_id: int, opcode: str) -> bool:
        """Return whether one operation of one program instance is traced."""
        if self.program_ids is not None and program_id not in self.program_ids:
            return False

        return matches_pattern(opcode, self.opcodes)


@dataclass(frozen=True, kw_only=True)
class TraceEvent:
    """One traced SSA operation.

    :param program_id: The program instance that executed the operation.
    :param location: The SSA location, for example ``entry:4:mem.store``.
    :param opcode: The SSA opcode.
    :param operands: The operand names of the operation.
    :param inputs: A summary of every operand value.
    :param outputs: A summary of every result value.
    :param mask: A summary of the mask active for the operation, if any.
    :param values: The copied operand and result values, when they are kept.
    """

    program_id: int
    location: str
    opcode: str
    operands: tuple[str, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    mask: str | None = None
    values: dict[str, object] | None = None

    def format(self) -> str:
        """Render the event as one readable line."""
        line = f"[program {self.program_id}] {self.location} {self.opcode}"

        if self.operands:
            line = f"{line} ({', '.join(self.operands)})"

        if self.inputs:
            rendered = "; ".join(f"{name}={text}" for name, text in self.inputs.items())
            line = f"{line}\n    in : {rendered}"

        if self.outputs:
            rendered = "; ".join(
                f"{name}={text}" for name, text in self.outputs.items()
            )
            line = f"{line}\n    out: {rendered}"

        if self.mask is not None:
            line = f"{line}\n    mask: {self.mask}"

        return line


class ExecutionTrace:
    """The ordered trace events collected during one interpreter run."""

    def __init__(self):
        self.events: list[TraceEvent] = []

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def append(self, event: TraceEvent) -> None:
        """Append one traced operation."""
        self.events.append(event)

    def filter(self, *, program_ids=None, opcodes=None) -> "ExecutionTrace":
        """Return a new trace restricted to the requested program ids and opcodes.

        :param program_ids: The program instances to keep, or ``None`` for all.
        :param opcodes: The opcodes to keep, matched exactly or by prefix.
        :return: The filtered trace.
        """
        options = TraceOptions(program_ids=program_ids, opcodes=opcodes)

        filtered = ExecutionTrace()

        for event in self.events:
            if options.matches(program_id=event.program_id, opcode=event.opcode):
                filtered.append(event)

        return filtered

    def format(self) -> str:
        """Render every event as a readable multi-line string."""
        return "\n".join(event.format() for event in self.events)


class TraceRecorder:
    """Collect trace events according to a set of trace options."""

    def __init__(self, options: TraceOptions | None = None):
        self.options = options if options is not None else TraceOptions()
        self.trace = ExecutionTrace()

    def enabled(self, *, program_id: int, opcode: str) -> bool:
        """Return whether the given operation should be traced."""
        return self.options.matches(program_id=program_id, opcode=opcode)

    def record(
        self,
        *,
        program_id: int,
        location: str,
        opcode: str,
        operands=(),
        inputs=(),
        outputs=(),
        mask=None,
    ) -> None:
        """Record one executed SSA operation.

        :param program_id: The program instance that executed the operation.
        :param location: The SSA location of the operation.
        :param opcode: The SSA opcode.
        :param operands: Pairs of operand name and value.
        :param outputs: Pairs of result name and value.
        :param mask: The mask active for the operation, if any.
        """
        options = self.options
        inputs_ = {
            name: summarize(value, max_elements=options.max_elements)
            for name, value in inputs
        }
        outputs_ = {
            name: summarize(value, max_elements=options.max_elements)
            for name, value in outputs
        }
        values = None

        if options.keep_values:
            values = {name: _copy_value(value) for name, value in inputs}
            values.update({name: _copy_value(value) for name, value in outputs})

        self.trace.append(
            TraceEvent(
                program_id=program_id,
                location=location,
                opcode=opcode,
                operands=tuple(operands),
                inputs=inputs_,
                outputs=outputs_,
                mask=(
                    None
                    if mask is None
                    else summarize(mask, max_elements=options.max_elements)
                ),
                values=values,
            )
        )


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()

    return value


def summarize_trace(trace: ExecutionTrace, *, head: int | None = None) -> str:
    """Render a trace, optionally limited to the first events.

    :param trace: The trace to render.
    :param head: The number of leading events to render, or ``None`` for all.
    :return: The rendered trace.
    """
    if head is None or head >= len(trace.events):
        return trace.format()

    return "\n".join(event.format() for event in trace.events[:head])


__all__ = [
    "DEFAULT_MAX_ELEMENTS",
    "ExecutionTrace",
    "TraceEvent",
    "TraceOptions",
    "TraceRecorder",
    "matches_pattern",
    "summarize",
    "summarize_trace",
]
