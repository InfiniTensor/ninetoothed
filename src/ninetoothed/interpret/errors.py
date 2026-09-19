"""Errors raised by the CPU reference interpreter.

Every diagnostic carries the SSA operation name and the SSA location of the
offending operation so that a failing interpretation can be traced back to a
single node of the ``ssa.Program`` without reading generated backend code.
"""


class InterpreterError(RuntimeError):
    """Base class for every interpreter failure.

    :param message: The human readable description of the failure.
    :param opcode: The SSA opcode of the offending operation, if known.
    :param location: The SSA location (``block:index:opcode``) of the operation.
    """

    def __init__(self, message, *, opcode=None, location=None):
        self.message = message
        self.opcode = opcode
        self.location = location

        super().__init__(self._render())

    def _render(self):
        if self.opcode is None and self.location is None:
            return self.message

        return f"{self.message} (operation `{self.opcode}` at `{self.location}`)"


class UnsupportedOperationError(InterpreterError):
    """Raised when an SSA opcode has no registered interpreter implementation."""


class UnsupportedDTypeError(InterpreterError):
    """Raised when an SSA dtype cannot be represented by the CPU memory model."""


class UnsupportedAccessError(InterpreterError):
    """Raised for memory access patterns the CPU memory model cannot resolve."""


class UnsupportedControlFlowError(InterpreterError):
    """Raised when a control-flow region cannot be executed deterministically."""


class MissingSymbolError(InterpreterError):
    """Raised when a symbolic dimension has no runtime value."""


class ProgramDomainError(InterpreterError):
    """Raised when the launch domain of a program cannot be derived."""


class TraceStop(InterpreterError):
    """Raised internally to stop execution at a breakpoint."""

    def __init__(self, event):
        self.event = event

        super().__init__(f"Execution stopped at breakpoint {event.location}.")


__all__ = [
    "InterpreterError",
    "MissingSymbolError",
    "ProgramDomainError",
    "TraceStop",
    "UnsupportedAccessError",
    "UnsupportedControlFlowError",
    "UnsupportedDTypeError",
    "UnsupportedOperationError",
]
