"""Opcode registry for the CPU reference interpreter.

Operations are registered by opcode so that new operations can be added by
importing an extra module and calling :func:`register` — no interpreter change
is required.  The registry doubles as the machine-readable support matrix
exposed through :func:`support_matrix`.
"""

from dataclasses import dataclass
from typing import Callable

from .errors import UnsupportedOperationError


@dataclass(frozen=True)
class OperationSpec:
    """Registration record for one opcode."""

    opcode: str
    category: str
    handler: Callable
    summary: str


_HANDLERS: dict[str, OperationSpec] = {}

#: Opcodes the interpreter knowingly refuses, mapped to the reason.
UNSUPPORTED_OPERATIONS: dict[str, str] = {
    "mem.atomic_add": "Atomic accumulation is order-dependent and out of scope.",
    "math.rand": "Random number generation is not reproducible across backends.",
}


def register(*opcodes, category, summary):
    """Register an operation handler.

    :param opcodes: The SSA opcodes handled by the decorated function.
    :param category: The operation family (for example ``"arith"``).
    :param summary: A one-line description used by the support matrix.
    """

    def decorator(func):
        for opcode in opcodes:
            if opcode in _HANDLERS:
                raise ValueError(f"Operation `{opcode}` is already registered.")

            _HANDLERS[opcode] = OperationSpec(
                opcode=opcode,
                category=category,
                handler=func,
                summary=summary,
            )
        return func

    return decorator


def handler_for(opcode):
    """Return the registered handler for ``opcode`` or ``None``."""
    spec = _HANDLERS.get(opcode)

    return None if spec is None else spec.handler


def spec_for(opcode):
    """Return the registration record for ``opcode`` or ``None``."""
    return _HANDLERS.get(opcode)


def require_handler(opcode, *, location=None):
    """Return the handler for ``opcode``, raising a rich error when missing.

    :param opcode: The SSA opcode.
    :param location: The SSA location used in the diagnostic.
    :raises UnsupportedOperationError: If the opcode has no implementation.
    """
    handler = handler_for(opcode)

    if handler is not None:
        return handler

    if opcode.startswith("call."):
        raise UnsupportedOperationError(
            f"Target intrinsic `{opcode[len('call.') :]}` has no CPU reference "
            "implementation. Intrinsics such as `triton.cdiv` are backend-specific; "
            "replace them with an equivalent arithmetic expression to interpret the "
            "program on the CPU.",
            opcode=opcode,
            location=location,
        )

    reason = UNSUPPORTED_OPERATIONS.get(opcode)
    detail = f" {reason}" if reason else ""

    raise UnsupportedOperationError(
        f"Unsupported SSA operation `{opcode}`." + detail,
        opcode=opcode,
        location=location,
    )


def supported_opcodes() -> tuple:
    """Return every registered opcode, sorted."""
    return tuple(sorted(_HANDLERS))


def support_matrix() -> dict:
    """Return a category-keyed mapping of the supported operations.

    :return: ``{category: (OperationSpec, ...)}``.
    """
    matrix: dict[str, list[OperationSpec]] = {}

    for spec in _HANDLERS.values():
        matrix.setdefault(spec.category, []).append(spec)

    return {
        category: tuple(sorted(specs, key=lambda spec: spec.opcode))
        for category, specs in sorted(matrix.items())
    }


def format_support_matrix() -> str:
    """Render the support matrix as a plain-text table."""
    lines = []

    for category, specs in support_matrix().items():
        lines.append(f"[{category}]")

        for spec in specs:
            lines.append(f"  {spec.opcode:<24} {spec.summary}")

    if UNSUPPORTED_OPERATIONS:
        lines.append("[unsupported]")

        for opcode, reason in sorted(UNSUPPORTED_OPERATIONS.items()):
            lines.append(f"  {opcode:<24} {reason}")
    return "\n".join(lines)


__all__ = [
    "UNSUPPORTED_OPERATIONS",
    "OperationSpec",
    "format_support_matrix",
    "handler_for",
    "register",
    "require_handler",
    "spec_for",
    "support_matrix",
    "supported_opcodes",
]
