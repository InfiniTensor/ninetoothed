"""Errors raised by the CPU reference interpreter."""


class InterpretationError(RuntimeError):
    """Base class for every CPU interpreter failure."""


class UnsupportedOperationError(InterpretationError):
    """Report an SSA operation the CPU interpreter cannot execute.

    :param opcode: The unsupported SSA opcode, for example ``mem.atomic_add``.
    :param location: The SSA location, for example ``entry:3:mem.atomic_add``.
    :param detail: An optional explanation of why the operation is unsupported.
    """

    def __init__(self, opcode, location, detail=None):
        message = f"Unsupported SSA operation `{opcode}` at `{location}`"

        if detail:
            message = f"{message}: {detail}."
        else:
            message = f"{message}."

        super().__init__(message)

        self.opcode = opcode
        self.location = location
        self.detail = detail


class UnsupportedDTypeError(InterpretationError):
    """Report an SSA dtype the CPU interpreter cannot represent."""


class UnsupportedAccessError(InterpretationError):
    """Report a memory access pattern the CPU interpreter cannot model."""


class InvalidProgramError(InterpretationError):
    """Report an SSA program that violates the interpreter's execution contract."""


class InvalidArgumentError(InterpretationError):
    """Report a runtime argument mismatch for an interpreter kernel."""


__all__ = [
    "InterpretationError",
    "InvalidArgumentError",
    "InvalidProgramError",
    "UnsupportedAccessError",
    "UnsupportedDTypeError",
    "UnsupportedOperationError",
]
